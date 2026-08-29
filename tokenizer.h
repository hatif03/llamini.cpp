#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "common.h"
#include "gguf.h"

// A llama-style SentencePiece BPE vocabulary, loaded from a GGUF file's
// tokenizer.ggml.* metadata (tokens, per-token merge scores, per-token
// type, and special token ids). `hash` is an opaque hand-rolled
// string->id table (see tokenizer.c) -- with a real ~32000-entry vocab, a
// linear scan per lookup is too slow for interactive use, so this is one
// piece of "stdlib has no hashmap" infrastructure this project does need.
typedef struct {
    char** tokens; // n entries, owned
    f32* scores;    // n entries, BPE merge priority (higher merges first)
    i32* types;      // n entries, llama.cpp token_type enum (3 == control)
    u64 n;
    u32 bos_id, eos_id, unk_id;
    void* hash;       // StrMap*, internal to tokenizer.c
} Vocab;

// Loads tokenizer.ggml.tokens/scores/token_type and the bos/eos/unk ids.
// Returns -1 if the file has no (or a malformed) tokenizer vocab.
int vocab_load_from_gguf(GGUFFile* gf, Vocab* vocab);
void vocab_free(Vocab* vocab);

// Encodes text into token ids (BOS-prefixed). SentencePiece-style: leading
// space is treated as a word boundary, spaces become the SPACE marker
// U+2581, then adjacent symbols are greedily merged by vocab score until
// no further merge exists in the vocab; anything left over that still
// isn't a vocab entry is byte-fallback-encoded as "<0xXX>" pieces.
u32 text_to_tokens(const Vocab* vocab, const char* text, u32* tokens, u32 max_tokens);

// Decodes one token id into human-readable UTF-8 bytes in out_str
// (SPACE marker -> ' ', "<0xXX>" byte-fallback pieces -> the raw byte).
// Control tokens (<s>, </s>, ...) are suppressed (write nothing) since a
// chat transcript shouldn't show them. Returns the number of bytes
// written (0 for a suppressed or out-of-range token).
u32 token_to_text(const Vocab* vocab, u32 token, char* out_str, u32 str_len);

#endif
