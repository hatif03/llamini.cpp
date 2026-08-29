#include "gguf.h"

// ================================================================
// Bounds-checked cursor over the mapped file. Every multi-byte read
// advances `pos` and checks it against `size` first; once a read runs
// past the file, `ok` goes to 0 and every later read on this cursor is a
// silent no-op, so a truncated/malformed file fails the whole parse
// instead of reading garbage.
// ================================================================
typedef struct {
    const u8* base;
    u64 size;
    u64 pos;
    int ok;
} Cur;

static void cur_init(Cur* c, const u8* base, u64 size, u64 pos) {
    c->base = base; c->size = size; c->pos = pos; c->ok = 1;
}

static void cur_bytes(Cur* c, void* out, u64 n) {
    if (!c->ok || c->pos > c->size || n > c->size - c->pos) { c->ok = 0; return; }
    if (out) memcpy(out, c->base + c->pos, n);
    c->pos += n;
}

static u32 cur_u32(Cur* c) { u32 v = 0; cur_bytes(c, &v, 4); return v; }
static u64 cur_u64(Cur* c) { u64 v = 0; cur_bytes(c, &v, 8); return v; }

// Reads a GGUF string (u64 length prefix + raw bytes, not null-terminated
// on disk) into a freshly malloc'd null-terminated buffer.
static char* cur_str(Cur* c) {
    u64 len = cur_u64(c);
    if (!c->ok) return NULL;
    char* s = (char*)malloc(len + 1);
    if (!s) { c->ok = 0; return NULL; }
    cur_bytes(c, s, len);
    if (!c->ok) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

// Skips a string's bytes without allocating.
static void cur_skip_str(Cur* c) {
    u64 len = cur_u64(c);
    cur_bytes(c, NULL, len);
}

// Byte size of a fixed-width scalar type; 0 for STRING/ARRAY (variable
// size, handled by their callers).
static u64 gguf_scalar_size(u32 type) {
    switch (type) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static void cur_skip_scalar_or_string(Cur* c, u32 type) {
    if (type == GGUF_TYPE_STRING) { cur_skip_str(c); return; }
    u64 sz = gguf_scalar_size(type);
    if (sz == 0) { c->ok = 0; return; } // unknown/unsupported type tag
    cur_bytes(c, NULL, sz);
}

// ================================================================
// Parsing: header -> metadata KV table -> tensor-info table -> aligned
// tensor data section. Real GGUF layout (see README/STDLIB.md) -- this
// replaces the old "assume raw f32 right after the header" placeholder.
// ================================================================

static void free_parsed(GGUFFile* gf) {
    if (gf->kv) {
        for (u64 i = 0; i < gf->n_kv; i++) free(gf->kv[i].key);
        free(gf->kv);
        gf->kv = NULL;
    }
    if (gf->tensors) {
        for (u64 i = 0; i < gf->n_tensors; i++) free(gf->tensors[i].name);
        free(gf->tensors);
        gf->tensors = NULL;
    }
}

static int raw_read(GGUFFile* gf, u64 off, void* out, u64 n) {
    if (off > gf->size || n > gf->size - off) return -1;
    memcpy(out, (const u8*)gf->data + off, n);
    return 0;
}

static const GGUFMeta* find_kv(GGUFFile* gf, const char* key) {
    for (u64 i = 0; i < gf->n_kv; i++)
        if (gf->kv[i].key && strcmp(gf->kv[i].key, key) == 0) return &gf->kv[i];
    return NULL;
}

int gguf_open(const char* path, GGUFFile* gf) {
    memset(gf, 0, sizeof(GGUFFile));
    gf->fd = open(path, O_RDONLY);
    if (gf->fd < 0) return -1;

    struct stat st;
    fstat(gf->fd, &st);
    gf->size = st.st_size;

    gf->data = mmap(NULL, gf->size, PROT_READ, MAP_PRIVATE, gf->fd, 0);
    if (gf->data == MAP_FAILED) { close(gf->fd); return -1; }

    Cur c; cur_init(&c, (const u8*)gf->data, gf->size, 0);
    gf->hdr.magic   = cur_u32(&c);
    gf->hdr.version = cur_u32(&c);
    gf->hdr.n_tensors  = cur_u64(&c);
    gf->hdr.n_metadata = cur_u64(&c);
    if (!c.ok || gf->hdr.magic != GGUF_MAGIC) goto fail;

    // Metadata KV table
    gf->n_kv = gf->hdr.n_metadata;
    gf->kv = (GGUFMeta*)calloc(gf->n_kv ? gf->n_kv : 1, sizeof(GGUFMeta));
    if (!gf->kv) goto fail;
    for (u64 i = 0; i < gf->n_kv; i++) {
        GGUFMeta* m = &gf->kv[i];
        m->key = cur_str(&c);
        m->type = cur_u32(&c);
        if (!c.ok) goto fail;
        if (m->type == GGUF_TYPE_ARRAY) {
            m->arr_elem_type = cur_u32(&c);
            m->arr_len = cur_u64(&c);
            m->value_off = c.pos; // first element starts right after this sub-header
            if (!c.ok) goto fail;
            if (m->arr_elem_type == GGUF_TYPE_STRING) {
                for (u64 j = 0; j < m->arr_len && c.ok; j++) cur_skip_str(&c);
            } else {
                u64 esz = gguf_scalar_size(m->arr_elem_type);
                if (esz == 0) { c.ok = 0; }
                else cur_bytes(&c, NULL, esz * m->arr_len);
            }
        } else {
            m->value_off = c.pos; // scalar/string value starts here
            cur_skip_scalar_or_string(&c, m->type);
        }
        if (!c.ok) goto fail;
    }

    // Tensor-info table
    gf->n_tensors = gf->hdr.n_tensors;
    gf->tensors = (GGUFTensorInfo*)calloc(gf->n_tensors ? gf->n_tensors : 1, sizeof(GGUFTensorInfo));
    if (!gf->tensors) goto fail;
    for (u64 i = 0; i < gf->n_tensors; i++) {
        GGUFTensorInfo* t = &gf->tensors[i];
        t->name = cur_str(&c);
        t->n_dims = cur_u32(&c);
        if (!c.ok || t->n_dims > 4) goto fail;
        for (u32 d = 0; d < t->n_dims; d++) t->dims[d] = cur_u64(&c);
        for (u32 d = t->n_dims; d < 4; d++) t->dims[d] = 1;
        t->ggml_type = cur_u32(&c);
        t->offset = cur_u64(&c);
        if (!c.ok) goto fail;
    }

    // Tensor data starts here, padded up to general.alignment (default 32).
    {
        u32 align = gguf_get_u32(gf, "general.alignment", 32);
        if (align == 0) align = 32;
        u64 unaligned = c.pos;
        gf->tensor_offset = (unaligned + align - 1) / align * align;
    }
    return 0;

fail:
    free_parsed(gf);
    munmap(gf->data, gf->size);
    close(gf->fd);
    memset(gf, 0, sizeof(GGUFFile));
    gf->fd = -1;
    return -1;
}

void gguf_close(GGUFFile* gf) {
    free_parsed(gf);
    if (gf->data) munmap(gf->data, gf->size);
    if (gf->fd >= 0) close(gf->fd);
}

int gguf_read_f32(GGUFFile* gf, u64 offset, f32* out, u64 elem_cnt) {
    u64 need = elem_cnt * sizeof(f32);
    if (gf->tensor_offset > gf->size) return -1;
    u64 avail = gf->size - gf->tensor_offset;
    if (offset > avail) return -1;
    avail -= offset;
    if (need > avail) return -1;

    u8* src = (u8*)gf->data + gf->tensor_offset + offset;
    memcpy(out, src, need);
    return 0;
}

// ================================================================
// Metadata accessors
// ================================================================

u32 gguf_get_u32(GGUFFile* gf, const char* key, u32 def_val) {
    const GGUFMeta* m = find_kv(gf, key);
    if (!m) return def_val;
    switch (m->type) {
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: { u32 v; return raw_read(gf, m->value_off, &v, 4) == 0 ? v : def_val; }
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: { u16 v; return raw_read(gf, m->value_off, &v, 2) == 0 ? v : def_val; }
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  { u8  v; return raw_read(gf, m->value_off, &v, 1) == 0 ? v : def_val; }
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: { u64 v; return raw_read(gf, m->value_off, &v, 8) == 0 ? (u32)v : def_val; }
        default: return def_val;
    }
}

f32 gguf_get_f32(GGUFFile* gf, const char* key, f32 def_val) {
    const GGUFMeta* m = find_kv(gf, key);
    if (!m) return def_val;
    if (m->type == GGUF_TYPE_FLOAT32) { f32 v; return raw_read(gf, m->value_off, &v, 4) == 0 ? v : def_val; }
    if (m->type == GGUF_TYPE_FLOAT64) { double v; return raw_read(gf, m->value_off, &v, 8) == 0 ? (f32)v : def_val; }
    return def_val;
}

u64 gguf_meta_array_len(GGUFFile* gf, const char* key, u32* elem_type_out) {
    const GGUFMeta* m = find_kv(gf, key);
    if (!m || m->type != GGUF_TYPE_ARRAY) return 0;
    if (elem_type_out) *elem_type_out = m->arr_elem_type;
    return m->arr_len;
}

char** gguf_get_string_array(GGUFFile* gf, const char* key, u64* count_out) {
    if (count_out) *count_out = 0;
    const GGUFMeta* m = find_kv(gf, key);
    if (!m || m->type != GGUF_TYPE_ARRAY || m->arr_elem_type != GGUF_TYPE_STRING) return NULL;

    char** arr = (char**)calloc(m->arr_len ? m->arr_len : 1, sizeof(char*));
    if (!arr) return NULL;
    Cur c; cur_init(&c, (const u8*)gf->data, gf->size, m->value_off);
    for (u64 i = 0; i < m->arr_len; i++) {
        arr[i] = cur_str(&c);
        if (!c.ok) {
            for (u64 j = 0; j <= i; j++) free(arr[j]);
            free(arr);
            return NULL;
        }
    }
    if (count_out) *count_out = m->arr_len;
    return arr;
}

f32* gguf_get_f32_array(GGUFFile* gf, const char* key, u64* count_out) {
    if (count_out) *count_out = 0;
    const GGUFMeta* m = find_kv(gf, key);
    if (!m || m->type != GGUF_TYPE_ARRAY || m->arr_elem_type != GGUF_TYPE_FLOAT32) return NULL;
    f32* arr = (f32*)malloc((m->arr_len ? m->arr_len : 1) * sizeof(f32));
    if (!arr) return NULL;
    if (raw_read(gf, m->value_off, arr, m->arr_len * sizeof(f32)) != 0) { free(arr); return NULL; }
    if (count_out) *count_out = m->arr_len;
    return arr;
}

i32* gguf_get_i32_array(GGUFFile* gf, const char* key, u64* count_out) {
    if (count_out) *count_out = 0;
    const GGUFMeta* m = find_kv(gf, key);
    if (!m || m->type != GGUF_TYPE_ARRAY || m->arr_elem_type != GGUF_TYPE_INT32) return NULL;
    i32* arr = (i32*)malloc((m->arr_len ? m->arr_len : 1) * sizeof(i32));
    if (!arr) return NULL;
    if (raw_read(gf, m->value_off, arr, m->arr_len * sizeof(i32)) != 0) { free(arr); return NULL; }
    if (count_out) *count_out = m->arr_len;
    return arr;
}

// ================================================================
// Tensor lookup
// ================================================================

const GGUFTensorInfo* gguf_find_tensor(GGUFFile* gf, const char* name) {
    for (u64 i = 0; i < gf->n_tensors; i++)
        if (gf->tensors[i].name && strcmp(gf->tensors[i].name, name) == 0) return &gf->tensors[i];
    return NULL;
}

u64 gguf_tensor_count(const GGUFTensorInfo* info) {
    u64 n = 1;
    for (u32 d = 0; d < info->n_dims; d++) n *= info->dims[d];
    return n;
}

// ================================================================
// GGML dequantization. Formats implemented: F32, F16, Q4_0, Q4_1, Q8_0
// (simple per-32-element blocks) and Q4_K, Q6_K (the "K-quant" super-block
// format an actual Q4_K_M file uses for most and some tensors respectively).
// Anything else falls through to the `default: return -1` case rather
// than being silently misread -- see gguf.h.
// ================================================================

static f32 f16_to_f32(u16 h) {
    u32 sign = (u32)(h & 0x8000) << 16;
    u32 exp  = (h >> 10) & 0x1F;
    u32 mant = h & 0x3FF;
    u32 bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    f32 out;
    memcpy(&out, &bits, 4);
    return out;
}

// Q4_K's 12-byte `scales` field packs six-bit (scale, min) pairs for 8
// sub-blocks of 32 elements each. Sub-blocks 0-3 store their scale/min
// directly; 4-7 reuse the low bits of 0-3's bytes for their high bits.
static void get_scale_min_k4(u32 j, const u8* q, u8* d_out, u8* m_out) {
    if (j < 4) {
        *d_out = q[j] & 63;
        *m_out = q[j + 4] & 63;
    } else {
        *d_out = (u8)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m_out = (u8)((q[j + 4] >> 4)  | ((q[j]     >> 6) << 4));
    }
}

int gguf_dequantize_tensor(GGUFFile* gf, const GGUFTensorInfo* info, f32* out, u64 out_count) {
    u64 n = gguf_tensor_count(info);
    if (n != out_count) return -1;
    u64 abs_off = gf->tensor_offset + info->offset;
    if (abs_off < info->offset) return -1; // overflow guard

    switch (info->ggml_type) {
    case GGML_TYPE_F32: {
        return raw_read(gf, abs_off, out, n * sizeof(f32));
    }
    case GGML_TYPE_F16: {
        if (abs_off > gf->size || n * 2 > gf->size - abs_off) return -1;
        const u16* src = (const u16*)((const u8*)gf->data + abs_off);
        for (u64 i = 0; i < n; i++) out[i] = f16_to_f32(src[i]);
        return 0;
    }
    case GGML_TYPE_Q4_0: {
        if (n % 32 != 0) return -1;
        u64 nb = n / 32;
        if (abs_off > gf->size || nb * 18 > gf->size - abs_off) return -1;
        const u8* p = (const u8*)gf->data + abs_off;
        for (u64 b = 0; b < nb; b++) {
            u16 dh; memcpy(&dh, p, 2);
            f32 d = f16_to_f32(dh);
            const u8* qs = p + 2;
            f32* y = out + b * 32;
            for (u32 i = 0; i < 16; i++) {
                y[i]      = ((f32)(qs[i] & 0xF) - 8.0f) * d;
                y[i + 16] = ((f32)(qs[i] >> 4)  - 8.0f) * d;
            }
            p += 18;
        }
        return 0;
    }
    case GGML_TYPE_Q4_1: {
        if (n % 32 != 0) return -1;
        u64 nb = n / 32;
        if (abs_off > gf->size || nb * 20 > gf->size - abs_off) return -1;
        const u8* p = (const u8*)gf->data + abs_off;
        for (u64 b = 0; b < nb; b++) {
            u16 dh, mh; memcpy(&dh, p, 2); memcpy(&mh, p + 2, 2);
            f32 d = f16_to_f32(dh), m = f16_to_f32(mh);
            const u8* qs = p + 4;
            f32* y = out + b * 32;
            for (u32 i = 0; i < 16; i++) {
                y[i]      = (f32)(qs[i] & 0xF) * d + m;
                y[i + 16] = (f32)(qs[i] >> 4)  * d + m;
            }
            p += 20;
        }
        return 0;
    }
    case GGML_TYPE_Q8_0: {
        if (n % 32 != 0) return -1;
        u64 nb = n / 32;
        if (abs_off > gf->size || nb * 34 > gf->size - abs_off) return -1;
        const u8* p = (const u8*)gf->data + abs_off;
        for (u64 b = 0; b < nb; b++) {
            u16 dh; memcpy(&dh, p, 2);
            f32 d = f16_to_f32(dh);
            const i8* qs = (const i8*)(p + 2);
            f32* y = out + b * 32;
            for (u32 i = 0; i < 32; i++) y[i] = (f32)qs[i] * d;
            p += 34;
        }
        return 0;
    }
    case GGML_TYPE_Q4_K: {
        if (n % 256 != 0) return -1;
        u64 nb = n / 256;
        if (abs_off > gf->size || nb * 144 > gf->size - abs_off) return -1;
        const u8* p = (const u8*)gf->data + abs_off;
        for (u64 b = 0; b < nb; b++) {
            u16 dh, dminh; memcpy(&dh, p, 2); memcpy(&dminh, p + 2, 2);
            f32 d = f16_to_f32(dh), dmin = f16_to_f32(dminh);
            const u8* scales = p + 4;
            const u8* q = p + 16;
            f32* y = out + b * 256;
            u32 is = 0;
            for (u32 j = 0; j < 256; j += 64) {
                u8 sc1, m1, sc2, m2;
                get_scale_min_k4(is + 0, scales, &sc1, &m1);
                get_scale_min_k4(is + 1, scales, &sc2, &m2);
                f32 d1 = d * sc1, mm1 = dmin * m1;
                f32 d2 = d * sc2, mm2 = dmin * m2;
                for (u32 l = 0; l < 32; l++) y[l]      = d1 * (f32)(q[l] & 0xF) - mm1;
                for (u32 l = 0; l < 32; l++) y[l + 32] = d2 * (f32)(q[l] >> 4)  - mm2;
                y += 64; q += 32; is += 2;
            }
            p += 144;
        }
        return 0;
    }
    case GGML_TYPE_Q6_K: {
        if (n % 256 != 0) return -1;
        u64 nb = n / 256;
        if (abs_off > gf->size || nb * 210 > gf->size - abs_off) return -1;
        const u8* p = (const u8*)gf->data + abs_off;
        for (u64 b = 0; b < nb; b++) {
            const u8* ql = p;
            const u8* qh = p + 128;
            const i8* sc = (const i8*)(p + 192);
            u16 dh; memcpy(&dh, p + 208, 2);
            f32 d = f16_to_f32(dh);
            f32* y = out + b * 256;
            for (u32 blk = 0; blk < 256; blk += 128) {
                for (u32 l = 0; l < 32; l++) {
                    u32 is = l / 16;
                    i8 q1 = (i8)((ql[l + 0]  & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    i8 q2 = (i8)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    i8 q3 = (i8)((ql[l + 0]  >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                    i8 q4 = (i8)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l + 0]  = d * (f32)sc[is + 0] * (f32)q1;
                    y[l + 32] = d * (f32)sc[is + 2] * (f32)q2;
                    y[l + 64] = d * (f32)sc[is + 4] * (f32)q3;
                    y[l + 96] = d * (f32)sc[is + 6] * (f32)q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
            p += 210;
        }
        return 0;
    }
    default:
        return -1; // unsupported ggml_type
    }
}
