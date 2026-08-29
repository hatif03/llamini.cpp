#include "tokenizer.h"

// ================================================================
// Small open-addressing string -> token-id table. The real GGUF vocab
// has ~32000 entries and BPE merging does many lookups per input word,
// so (unlike the old 15-word demo table) a linear scan is genuinely too
// slow here -- this is the "no stdlib hashmap" substitution documented
// in STDLIB.md.
// ================================================================
typedef struct { const char* key; u32 len; u32 id; int used; } HEntry;
typedef struct { HEntry* table; u32 cap; } StrMap;

static u32 fnv1a(const char* s, u32 len) {
    u32 h = 2166136261u;
    for (u32 i = 0; i < len; i++) { h ^= (u8)s[i]; h *= 16777619u; }
    return h;
}

static void strmap_init(StrMap* m, u32 min_cap) {
    u32 cap = 16;
    while (cap < min_cap) cap <<= 1;
    m->table = (HEntry*)calloc(cap, sizeof(HEntry));
    m->cap = cap;
}

static void strmap_free(StrMap* m) { free(m->table); m->table = NULL; m->cap = 0; }

static void strmap_put(StrMap* m, const char* key, u32 len, u32 id) {
    u32 h = fnv1a(key, len) & (m->cap - 1);
    while (m->table[h].used) h = (h + 1) & (m->cap - 1);
    m->table[h].key = key; m->table[h].len = len; m->table[h].id = id; m->table[h].used = 1;
}

static int strmap_get(const StrMap* m, const char* key, u32 len, u32* id_out) {
    u32 h = fnv1a(key, len) & (m->cap - 1);
    u32 start = h;
    while (m->table[h].used) {
        if (m->table[h].len == len && memcmp(m->table[h].key, key, len) == 0) {
            *id_out = m->table[h].id;
            return 1;
        }
        h = (h + 1) & (m->cap - 1);
        if (h == start) break;
    }
    return 0;
}

// ================================================================
// Vocab loading
// ================================================================

int vocab_load_from_gguf(GGUFFile* gf, Vocab* v) {
    memset(v, 0, sizeof(Vocab));
    v->tokens = gguf_get_string_array(gf, "tokenizer.ggml.tokens", &v->n);
    if (!v->tokens || v->n == 0) return -1;

    u64 scores_n = 0, types_n = 0;
    v->scores = gguf_get_f32_array(gf, "tokenizer.ggml.scores", &scores_n);
    v->types  = gguf_get_i32_array(gf, "tokenizer.ggml.token_type", &types_n);
    if (!v->scores || scores_n != v->n) { free(v->scores); v->scores = (f32*)calloc(v->n, sizeof(f32)); }
    if (!v->types  || types_n  != v->n) { free(v->types);  v->types  = (i32*)calloc(v->n, sizeof(i32)); }

    v->bos_id = gguf_get_u32(gf, "tokenizer.ggml.bos_token_id", TOKEN_BOS);
    v->eos_id = gguf_get_u32(gf, "tokenizer.ggml.eos_token_id", TOKEN_EOS);
    v->unk_id = gguf_get_u32(gf, "tokenizer.ggml.unknown_token_id", 0);

    StrMap* map = (StrMap*)malloc(sizeof(StrMap));
    strmap_init(map, (u32)(v->n * 2));
    for (u64 i = 0; i < v->n; i++)
        strmap_put(map, v->tokens[i], (u32)strlen(v->tokens[i]), (u32)i);
    v->hash = map;
    return 0;
}

void vocab_free(Vocab* v) {
    if (!v) return;
    if (v->tokens) {
        for (u64 i = 0; i < v->n; i++) free(v->tokens[i]);
        free(v->tokens);
    }
    free(v->scores);
    free(v->types);
    if (v->hash) { strmap_free((StrMap*)v->hash); free(v->hash); }
    memset(v, 0, sizeof(Vocab));
}

// ================================================================
// Encode: SentencePiece-style greedy BPE merge by vocab score.
// ================================================================

// One symbol in the merge working set: a byte-span of the preprocessed
// buffer, linked into the remaining sequence via prev/next (an index, or
// -1). Merging keeps the left symbol's index and retires the right one,
// so the very first symbol (no prev) is never retired and is always the
// walk's starting point.
typedef struct { i32 prev, next; const char* text; u32 len; int alive; } Sym;

#define MAX_CHARS (MAX_PROMPT_LEN * 3 + 8)

static u32 utf8_len(u8 c) {
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; // not a valid leading byte -- treat as one raw byte
}

u32 text_to_tokens(const Vocab* v, const char* text, u32* tokens, u32 max_tokens) {
    u32 n_out = 0;
    if (n_out < max_tokens) tokens[n_out++] = v->bos_id;

    // Encode spaces (plus one leading boundary marker) as U+2581 "LOWER
    // ONE EIGHTH BLOCK" (▁, UTF-8 bytes E2 96 81) -- SentencePiece's word
    // boundary convention, which is how this vocab's pieces are written.
    static char buf[MAX_CHARS];
    u32 blen = 0;
    buf[blen++] = (char)0xE2; buf[blen++] = (char)0x96; buf[blen++] = (char)0x81;
    for (u32 i = 0; text[i] != '\0' && blen + 4 < sizeof(buf); i++) {
        if (text[i] == ' ') { buf[blen++] = (char)0xE2; buf[blen++] = (char)0x96; buf[blen++] = (char)0x81; }
        else buf[blen++] = text[i];
    }

    static Sym syms[MAX_CHARS];
    u32 n_syms = 0;
    for (u32 i = 0; i < blen; ) {
        u32 l = utf8_len((u8)buf[i]);
        if (i + l > blen) l = blen - i;
        syms[n_syms].text = buf + i;
        syms[n_syms].len = l;
        syms[n_syms].prev = (i32)n_syms - 1;
        syms[n_syms].next = -1;
        syms[n_syms].alive = 1;
        if (n_syms > 0) syms[n_syms - 1].next = (i32)n_syms;
        n_syms++;
        i += l;
    }

    // Repeatedly merge the best-scoring adjacent pair whose concatenation
    // exists in the vocab. ponytail: O(n^2) full rescan per merge instead
    // of a priority queue -- fine for a chat-length prompt (tens of
    // symbols), would need a heap for document-length input.
    for (;;) {
        i32 best_i = -1; f32 best_score = -1e30f; u32 best_len = 0;
        for (u32 i = 0; i < n_syms; i++) {
            if (!syms[i].alive) continue;
            i32 j = syms[i].next;
            if (j < 0) continue;
            u32 cand_len = syms[i].len + syms[j].len;
            u32 id;
            if (strmap_get((const StrMap*)v->hash, syms[i].text, cand_len, &id) && v->scores[id] > best_score) {
                best_score = v->scores[id]; best_i = (i32)i; best_len = cand_len;
            }
        }
        if (best_i < 0) break;
        i32 j = syms[best_i].next;
        syms[best_i].len = best_len;
        syms[best_i].next = syms[j].next;
        if (syms[j].next >= 0) syms[syms[j].next].prev = best_i;
        syms[j].alive = 0;
    }

    for (i32 i = 0; i >= 0 && n_out < max_tokens; i = syms[i].next) {
        u32 id;
        if (strmap_get((const StrMap*)v->hash, syms[i].text, syms[i].len, &id)) {
            tokens[n_out++] = id;
        } else {
            for (u32 b = 0; b < syms[i].len && n_out < max_tokens; b++) {
                char piece[8];
                snprintf(piece, sizeof(piece), "<0x%02X>", (u8)syms[i].text[b]);
                u32 bid;
                tokens[n_out++] = strmap_get((const StrMap*)v->hash, piece, (u32)strlen(piece), &bid) ? bid : v->unk_id;
            }
        }
    }
    return n_out;
}

// ================================================================
// Decode
// ================================================================

u32 token_to_text(const Vocab* v, u32 token, char* out_str, u32 str_len) {
    if (str_len > 0) out_str[0] = '\0';
    if (token >= v->n || str_len == 0) return 0;
    if (v->types[token] == 3) return 0; // CONTROL (<s>, </s>, ...): don't print

    const char* piece = v->tokens[token];
    u32 plen = (u32)strlen(piece);

    // Byte-fallback piece "<0xXX>" -> the single raw byte it encodes.
    if (plen == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        u32 byte_val = 0;
        sscanf(piece + 3, "%2x", &byte_val);
        if (str_len >= 2) { out_str[0] = (char)byte_val; out_str[1] = '\0'; return 1; }
        return 0;
    }

    u32 oi = 0;
    for (u32 i = 0; piece[i] != '\0' && oi + 1 < str_len; ) {
        if ((u8)piece[i] == 0xE2 && (u8)piece[i + 1] == 0x96 && (u8)piece[i + 2] == 0x81) {
            out_str[oi++] = ' ';
            i += 3;
        } else {
            out_str[oi++] = piece[i++];
        }
    }
    out_str[oi] = '\0';
    return oi;
}
