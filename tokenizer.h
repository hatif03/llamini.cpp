#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "common.h"
#include "gguf.h"

// Two incompatible vocab/BPE conventions both show up as "tokenizer.ggml.*"
// metadata in real GGUF files, detected via tokenizer.ggml.model:
//  - SPM ("llama", "llama-spm", ...): SentencePiece-style. Merge priority
//    is a per-token float score; words get a literal SPACE marker (U+2581).
//  - BPE ("gpt2"): byte-level BPE (GPT-2/Qwen2/...). Merge priority is the
//    rank (line number) in an explicit merges list; every raw byte is
//    remapped through a fixed byte<->printable-unicode table before BPE
//    ever runs, so unlike SPM there's no separate byte-fallback path -- the
//    remapped single bytes are themselves ordinary vocab entries.
// `hash` (both kinds) and `merge_ranks` (BPE only) are opaque hand-rolled
// string->id tables (see tokenizer.c) -- with a real tens-of-thousands
// entry vocab, a linear scan per lookup is too slow for interactive use,
// so this is one piece of "stdlib has no hashmap" infrastructure this
// project does need.
typedef struct {
    char** tokens; // n entries, owned
    f32* scores;    // n entries, SPM merge priority (higher merges first); unused for BPE
    i32* types;      // n entries, llama.cpp token_type enum (3 == control)
    u64 n;
    u32 bos_id, eos_id, unk_id;
    void* hash;       // StrMap*, piece string -> token id (both kinds)
    int is_bpe;         // 0 = SentencePiece, 1 = byte-level BPE (gpt2-style)
    void* merge_ranks;   // StrMap*, "left right" -> rank (BPE only, else NULL)
    void* bytemap;         // ByteMap*, byte<->unicode table (BPE only, else NULL)
} Vocab;

// Loads tokenizer.ggml.tokens/scores/token_type (or, for BPE, .merges) and
// the bos/eos/unk ids. Returns -1 if the file has no (or a malformed)
// tokenizer vocab.
int vocab_load_from_gguf(GGUFFile* gf, Vocab* vocab);
void vocab_free(Vocab* vocab);

// Encodes text into token ids (BOS-prefixed). Dispatches on vocab->is_bpe:
// SentencePiece-style (leading space is a word boundary, spaces become the
// SPACE marker U+2581, greedy score-ranked merge, "<0xXX>" byte-fallback)
// or byte-level BPE (every input byte remapped through the byte<->unicode
// table first, greedy rank-ranked merge, no separate fallback needed).
u32 text_to_tokens(const Vocab* vocab, const char* text, u32* tokens, u32 max_tokens);

// Decodes one token id into human-readable UTF-8 bytes in out_str. SPM:
// SPACE marker -> ' ', "<0xXX>" byte-fallback pieces -> the raw byte. BPE:
// each of the piece's remapped codepoints -> its original raw byte via the
// byte<->unicode table. Control tokens (<s>, </s>, ...) are suppressed
// (write nothing) since a chat transcript shouldn't show them. Returns the
// number of bytes written (0 for a suppressed or out-of-range token).
u32 token_to_text(const Vocab* vocab, u32 token, char* out_str, u32 str_len);

#endif
