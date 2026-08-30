# STDLIB.md

Every place this project would normally reach for a package, and the libc/POSIX
stand-in used instead. See [README.md](README.md) "Limits" for what each stand-in
still can't do.

## Package Killer: what this replaces

Running a GGUF language model from scratch normally means installing at
least one of these:

| Package people actually install | Why | Replaced here by |
| --- | --- | --- |
| `ggml` / `llama.cpp` linked as a library | GGUF parsing, block dequantization, transformer kernels | `gguf.c` (header/metadata/tensor-info parser + `F32`/`F16`/`Q4_0`/`Q4_1`/`Q8_0`/`Q4_K`/`Q6_K` decoders), `model.c` (RMSNorm, RoPE, GQA attention, SwiGLU, `linear`) |
| `llama-cpp-python`, `ctransformers` (pip) | Python bindings over the above | Not needed — the CLI *is* the binary |
| `sentencepiece` (pip / libsentencepiece) | SPM-BPE encode/decode | `tokenizer.c`, driven entirely by the GGUF file's own `tokenizer.ggml.*` metadata |
| `numpy` / a BLAS | tensor storage + matmul | `tensor.c` and `linear()` in `model.c` (pthread-parallelized, gcc-vectorized — see the flags row below) |

### What this is NOT (scope, stated up front)

- **Not a ggml reimplementation.** It implements the subset needed to load
  and run one LLaMA-architecture `Q4_K_M` file, not ggml's graph engine,
  backends, or full op set.
- **Quantization coverage is partial:** `F32`, `F16`, `Q4_0`, `Q4_1`,
  `Q8_0`, `Q4_K`, `Q6_K` only. `Q2_K`, `Q3_K`, `Q5_K`, `Q8_K`, and the `IQ*`
  family are explicitly rejected (`-1`), never silently misread.
- **Architecture coverage is LLaMA-only.** No Mistral/Phi/Qwen quirks, no
  LoRA, no GPU, no batching.
- **Not numerically verified against ggml.** No reference implementation
  was run to cross-check outputs bit-for-bit. Correctness evidence is
  `--bench`'s teacher-forced perplexity against a random-baseline ceiling
  plus known-fact completions (measured **13.28** perplexity vs. a ~32000
  ceiling, and "The capital of France is" -> "Paris." — see README's
  "Correctness evidence"), not bit-exact comparison.

## Substitution log

| Instead of | Used | Rationale |
| --- | --- | --- |
| `ggml` / llama.cpp's GGUF loader | Hand-written binary parser in `gguf.c` (`open` + `mmap` + a bounds-checked cursor over the mapped bytes) | Parses the real GGUF layout: header, metadata key/value table (all 13 value types, including nested arrays), tensor-info table, alignment-padded tensor data section. Every multi-byte read is bounds-checked against the file size so a truncated/malformed file fails the parse instead of reading past the mapping. |
| `ggml`'s quantized-tensor dequantizers | Hand-rolled block decoders in `gguf.c` for `F32`, `F16`, `Q4_0`, `Q4_1`, `Q8_0`, and the two "K-quant" super-block formats `Q4_K` and `Q6_K` (the ones an actual `Q4_K_M` file uses) | Implemented from the public block layouts (block sizes, scale/min bit-packing), not copied from ggml's source. Verified against the real `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`: dequantized embedding and output-projection rows come back as small, varied, non-NaN floats consistent with real trained weights, and feeding them through the full forward pass produces recognizable English word fragments rather than noise. Anything outside this supported set (`Q2_K`, `Q3_K`, `Q5_K`, `Q8_K`, ...) is reported as unsupported (`-1`), never silently misread. |
| `sentencepiece` | Hand-written SentencePiece-style BPE encoder/decoder in `tokenizer.c`, driven entirely by the GGUF file's own `tokenizer.ggml.tokens` / `.scores` / `.token_type` metadata | Real vocab (32000 pieces for TinyLlama), real greedy score-ranked merge algorithm, real byte-fallback (`<0xXX>` pieces) for anything outside the vocab, real SPACE-marker (▁) handling on both encode and decode. Simplification: merge selection is an O(n^2) rescan per merge rather than a priority-queue heap — fine for chat-length input, would need a heap for document-length text (noted in `tokenizer.c`). |
| A hashmap/dictionary library | Hand-rolled open-addressing string->id table (`StrMap` in `tokenizer.c`), FNV-1a hashing with linear probing | With a real ~32000-entry vocab and BPE doing many lookups per input word, a linear scan is genuinely too slow for interactive use — unlike a tiny fixed vocab, this is infrastructure the project actually needs, not a gratuitous abstraction. |
| BLAS / `ggml`'s matmul kernels | `linear()` in `model.c`: a hand-written row-dot-product loop matching GGUF's native `(out_dim, in_dim)` row-major tensor layout | Deliberately not `tensor.c`'s `matmul()` (which expects a `[k,n]`-shaped operand) — reusing that would require transposing every loaded weight matrix first. Naive O(out*in) loop, no SIMD/blocking; see README "Slow". |
| A tensor/ndarray library (numpy, Eigen) | The `Tensor` struct (`tensor.h`) plus `tensor_create`/`vec_add`/`vec_scale`, and raw `f32*` buffers for the hot forward-pass path | Just a shape array, a size, and a `malloc`'d `f32*`; no broadcasting or strides beyond what the code explicitly loops over. |
| An IEEE-754 half-float library | Hand-written `f16_to_f32` bit-manipulation conversion in `gguf.c` | GGUF's `F16` tensors and every K-quant block's per-superblock scale are stored as raw 16-bit halfs; this is the standard public bit layout (sign/exponent/mantissa), not vendored from anywhere. |
| A quantization library (e.g. `ggml`'s block quantizers) | Hand-rolled scale/zero-point INT4 pack/unpack in `quant.c` | A separate, simpler teaching implementation (generic linear INT4, exercised by `--test`), unrelated to the real `Q4_K`/`Q6_K` GGUF decoders above. |
| `getopt_long` / a CLI-parsing package (clap-style) | `argc`/`argv` + `strcmp` in `main.c` | Only two shapes are needed (`--test`, or a single `.gguf` path), so a full flag parser would be pure ceremony. |
| A unit test framework (CTest, Unity, Check) | `--test` flag running in-process `assert()`/`printf` checks (`run_all_unit_tests` in `main.c`) | C ships no stdlib test framework; each `test_*` function is a self-contained smoke test, including a synthetic-weights run of the full per-layer GQA forward pass, with no fixtures or discovery magic. |
| A logging framework | Plain `printf` / `fprintf(stderr, ...)` | No log levels, no formatting pipeline — stdout for normal output, stderr for warnings/errors, per Track A CLI conventions. |
| A "safe string" / bounds-checked I/O library | Manual range checks throughout `gguf.c`'s cursor and `gguf_dequantize_tensor` before any `memcpy` out of the mmap'd region | A user-supplied `.gguf` is untrusted input; every read fails closed (`-1` / a sticky cursor error) instead of reading past the mapping on a short or malformed file. |
| A dependency-injection/config framework for model hyperparameters | `llama_config_from_gguf` in `model.c`, reading `llama.embedding_length` / `llama.block_count` / `llama.attention.head_count(_kv)` / `llama.feed_forward_length` / `llama.context_length` / RoPE and RMSNorm constants straight out of the file's own metadata | The model's shape is whatever the GGUF file says it is; hardcoded TinyLlama-1.1B numbers are only the fallback defaults for a key the file doesn't define. |
| A sampling/inference library's temperature+top-p logic | Hand-written `sample_token` in `generate.c`: numerically-stable softmax, `qsort` (libc) by probability, nucleus cutoff, then `rand()`/`srand()` (libc) to draw | `temp <= 0` dispatches straight to the pre-existing deterministic `greedy_sample`, so enabling this feature couldn't silently change the already-verified default behavior. |
| A chat-template/tokenizer library's special-token handling | `build_chat_prompt_tokens` in `main.c` inserts the real `eos_id` token programmatically between prompt segments, instead of embedding `"</s>"` as literal text | Found by testing, not assuming: `</s>` names a real EOS control token in this vocab, but typing it as text does **not** BPE-merge back into that single token (verified by tokenizing the literal string and finding zero occurrences of `eos_id` in the result) — this tokenizer has no special-token pre-split pass, unlike a real `transformers`/`sentencepiece` tokenizer's chat-template handling. Building the prompt at the token level, not the string level, sidesteps needing that pass at all. |

## Deliberately not substituted (out of scope, disclosed instead of faked)

- **No top-k, repetition penalty, or `--seed`.** `--temp` sampling (see table above) covers temperature + top-p only; no reproducible-sampled-transcript flag yet.
- **Chat template not verified against a reference tokenizer run.** `build_chat_prompt_tokens` (`main.c`) is derived from TinyLlama-Chat's published `tokenizer_config.json`, but this project has no `transformers` install to confirm the exact resulting token sequence bit-for-bit. See README Limits.
