#include "tokenizer.h"

// ================================================================
// Small open-addressing string -> u32 table (reused for both piece->id
// and, for BPE, "left right"->rank lookups). The real GGUF vocab has tens
// of thousands of entries and BPE merging does many lookups per input
// word, so (unlike the old 15-word demo table) a linear scan is genuinely
// too slow here -- this is the "no stdlib hashmap" substitution documented
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
// GPT-2-style byte<->unicode remapping. Byte-level BPE never sees raw
// bytes directly -- every byte is first mapped through this fixed table to
// a printable-or-safe unicode codepoint (so control bytes, whitespace,
// etc. can't confuse a text-based vocab file), and the vocab/merges files
// are themselves written in that remapped alphabet. This is GPT-2's own
// public convention (see gpt2's encoder.py bytes_to_unicode()), not
// invented here -- but it's implemented from scratch, not copied.
// ================================================================
typedef struct { u32 byte_to_cp[256]; i32 cp_to_byte[512]; } ByteMap;

static void bytemap_init(ByteMap* bm) {
    int is_base[256] = {0};
    for (int b = 33;  b <= 126; b++) is_base[b] = 1; // printable ASCII (excl. space)
    for (int b = 161; b <= 172; b++) is_base[b] = 1; // Latin-1 punctuation block
    for (int b = 174; b <= 255; b++) is_base[b] = 1; // Latin-1 letters block
    for (int i = 0; i < 512; i++) bm->cp_to_byte[i] = -1;
    u32 n = 0;
    for (int b = 0; b < 256; b++) {
        u32 cp = is_base[b] ? (u32)b : (256 + n++); // remaining bytes get sequential codepoints >= 256
        bm->byte_to_cp[b] = cp;
        if (cp < 512) bm->cp_to_byte[cp] = b;
    }
}

static u32 utf8_encode_cp(u32 cp, char* out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

// Decodes one UTF-8 codepoint at s (len bytes available); returns bytes
// consumed (>=1; malformed sequences fall back to 1 raw byte).
static u32 utf8_decode_cp(const char* s, u32 len, u32* cp_out) {
    u8 c0 = (u8)s[0];
    if (c0 < 0x80) { *cp_out = c0; return 1; }
    if ((c0 & 0xE0) == 0xC0 && len >= 2) { *cp_out = ((u32)(c0 & 0x1F) << 6) | ((u8)s[1] & 0x3F); return 2; }
    if ((c0 & 0xF0) == 0xE0 && len >= 3) { *cp_out = ((u32)(c0 & 0x0F) << 12) | (((u8)s[1] & 0x3F) << 6) | ((u8)s[2] & 0x3F); return 3; }
    *cp_out = c0;
    return 1;
}

// ================================================================
// Vocab loading
// ================================================================

int vocab_load_from_gguf(GGUFFile* gf, Vocab* v) {
    memset(v, 0, sizeof(Vocab));
    v->tokens = gguf_get_string_array(gf, "tokenizer.ggml.tokens", &v->n);
    if (!v->tokens || v->n == 0) return -1;

    char* model = gguf_get_str(gf, "tokenizer.ggml.model");
    v->is_bpe = (model && !strcmp(model, "gpt2"));
    free(model);

    u64 types_n = 0;
    v->types = gguf_get_i32_array(gf, "tokenizer.ggml.token_type", &types_n);
    if (!v->types || types_n != v->n) { free(v->types); v->types = (i32*)calloc(v->n, sizeof(i32)); }

    if (!v->is_bpe) {
        u64 scores_n = 0;
        v->scores = gguf_get_f32_array(gf, "tokenizer.ggml.scores", &scores_n);
        if (!v->scores || scores_n != v->n) { free(v->scores); v->scores = (f32*)calloc(v->n, sizeof(f32)); }
    }

    v->bos_id = gguf_get_u32(gf, "tokenizer.ggml.bos_token_id", TOKEN_BOS);
    v->eos_id = gguf_get_u32(gf, "tokenizer.ggml.eos_token_id", TOKEN_EOS);
    v->unk_id = gguf_get_u32(gf, "tokenizer.ggml.unknown_token_id", 0);

    StrMap* map = (StrMap*)malloc(sizeof(StrMap));
    strmap_init(map, (u32)(v->n * 2));
    for (u64 i = 0; i < v->n; i++)
        strmap_put(map, v->tokens[i], (u32)strlen(v->tokens[i]), (u32)i);
    v->hash = map;

    if (v->is_bpe) {
        u64 n_merges = 0;
        char** merges = gguf_get_string_array(gf, "tokenizer.ggml.merges", &n_merges);
        if (!merges) { vocab_free(v); return -1; } // BPE vocab with no merges list can't tokenize anything
        StrMap* ranks = (StrMap*)malloc(sizeof(StrMap));
        strmap_init(ranks, (u32)(n_merges * 2 + 1));
        for (u64 i = 0; i < n_merges; i++)
            strmap_put(ranks, merges[i], (u32)strlen(merges[i]), (u32)i); // owns merges[i] now, don't free it below
        v->merge_ranks = ranks;
        free(merges); // frees the array, not the (now hash-table-owned) strings

        ByteMap* bm = (ByteMap*)malloc(sizeof(ByteMap));
        bytemap_init(bm);
        v->bytemap = bm;
    }

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
    if (v->merge_ranks) {
        // The rank table's keys are the merges[] strings themselves
        // (strmap_put stores the pointer, not a copy) -- free each once.
        StrMap* m = (StrMap*)v->merge_ranks;
        for (u32 i = 0; i < m->cap; i++) if (m->table[i].used) free((void*)m->table[i].key);
        strmap_free(m);
        free(m);
    }
    free(v->bytemap);
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

static u32 text_to_tokens_spm(const Vocab* v, const char* text, u32* tokens, u32 max_tokens) {
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

static u32 token_to_text_spm(const Vocab* v, u32 token, char* out_str, u32 str_len) {
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

// ================================================================
// Encode: GPT-2-style byte-level BPE, merge priority = rank (position in
// the merges list; lower rank merges first), not a per-token score.
// ================================================================

// ponytail: pre-tokenization here is "split on whitespace, keep a leading
// space attached to the following run" -- not GPT-2's real Unicode-aware
// regex splitter (which needs \p{L}/\p{N} category tables this project
// doesn't have). Good enough for ordinary English chat input; a token
// like a contraction or a digit run butting against punctuation may split
// differently than a real GPT-2 tokenizer would. See README limits.
static u32 text_to_tokens_bpe(const Vocab* v, const char* text, u32* tokens, u32 max_tokens) {
    const ByteMap* bm = (const ByteMap*)v->bytemap;
    const StrMap* ranks = (const StrMap*)v->merge_ranks;
    u32 n_out = 0;
    if (n_out < max_tokens) tokens[n_out++] = v->bos_id;

    static char mapped[MAX_CHARS];
    static Sym syms[MAX_CHARS];

    u32 ti = 0;
    while (text[ti] != '\0' && n_out < max_tokens) {
        // One "word" chunk: an optional single leading space, then a run
        // of non-space bytes (or, if the text starts mid-run, just that
        // run). Each raw byte of the chunk is remapped independently.
        u32 start = ti;
        if (text[ti] == ' ') ti++;
        while (text[ti] != '\0' && text[ti] != ' ') ti++;
        if (ti == start) break; // shouldn't happen, but never spin forever

        u32 mlen = 0;
        for (u32 i = start; i < ti && mlen + 4 < sizeof(mapped); i++)
            mlen += utf8_encode_cp(bm->byte_to_cp[(u8)text[i]], mapped + mlen);

        u32 n_syms = 0;
        for (u32 i = 0; i < mlen; ) {
            u32 l = utf8_len((u8)mapped[i]);
            if (i + l > mlen) l = mlen - i;
            syms[n_syms].text = mapped + i;
            syms[n_syms].len = l;
            syms[n_syms].prev = (i32)n_syms - 1;
            syms[n_syms].next = -1;
            syms[n_syms].alive = 1;
            if (n_syms > 0) syms[n_syms - 1].next = (i32)n_syms;
            n_syms++;
            i += l;
        }

        // Best (lowest-rank) adjacent pair wins, same O(n^2)-rescan
        // structure as the SPM path above, just ranked the opposite way
        // (lower is higher priority) and keyed on the pair itself (built
        // as "left right" with a literal space -- never ambiguous, since
        // no remapped byte can itself be a raw 0x20).
        char pairbuf[64];
        for (;;) {
            i32 best_i = -1; u32 best_rank = 0xFFFFFFFFu; u32 best_len = 0;
            for (u32 i = 0; i < n_syms; i++) {
                if (!syms[i].alive) continue;
                i32 j = syms[i].next;
                if (j < 0) continue;
                u32 plen = syms[i].len + 1 + syms[j].len;
                if (plen >= sizeof(pairbuf)) continue;
                memcpy(pairbuf, syms[i].text, syms[i].len);
                pairbuf[syms[i].len] = ' ';
                memcpy(pairbuf + syms[i].len + 1, syms[j].text, syms[j].len);
                u32 rank;
                if (strmap_get(ranks, pairbuf, plen, &rank) && rank < best_rank) {
                    best_rank = rank; best_i = (i32)i; best_len = syms[i].len + syms[j].len;
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
            // Every base remapped byte is itself a vocab entry, so unlike
            // SPM there's no separate fallback path for "not found."
            tokens[n_out++] = strmap_get((const StrMap*)v->hash, syms[i].text, syms[i].len, &id) ? id : v->unk_id;
        }
    }
    return n_out;
}

static u32 token_to_text_bpe(const Vocab* v, u32 token, char* out_str, u32 str_len) {
    const ByteMap* bm = (const ByteMap*)v->bytemap;
    const char* piece = v->tokens[token];
    u32 plen = (u32)strlen(piece);

    u32 oi = 0;
    for (u32 i = 0; i < plen && oi + 1 < str_len; ) {
        u32 cp;
        u32 used = utf8_decode_cp(piece + i, plen - i, &cp);
        i32 b = (cp < 512) ? bm->cp_to_byte[cp] : -1;
        out_str[oi++] = (b >= 0) ? (char)b : '?'; // shouldn't happen for a well-formed BPE vocab
        i += used;
    }
    out_str[oi] = '\0';
    return oi;
}

// ================================================================
// Public API: dispatch on vocab->is_bpe
// ================================================================

u32 text_to_tokens(const Vocab* v, const char* text, u32* tokens, u32 max_tokens) {
    return v->is_bpe ? text_to_tokens_bpe(v, text, tokens, max_tokens)
                       : text_to_tokens_spm(v, text, tokens, max_tokens);
}

u32 token_to_text(const Vocab* v, u32 token, char* out_str, u32 str_len) {
    if (str_len > 0) out_str[0] = '\0';
    if (token >= v->n || str_len == 0) return 0;
    if (v->types[token] == 3) return 0; // CONTROL (<s>, </s>, <|endoftext|>, ...): don't print
    return v->is_bpe ? token_to_text_bpe(v, token, out_str, str_len)
                       : token_to_text_spm(v, token, out_str, str_len);
}
