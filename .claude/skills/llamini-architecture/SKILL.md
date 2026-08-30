---
name: llamini-architecture
description: Maps llamini.cpp modules, stubs vs target inference, and how to extend GGUF, tokenizer, quant, and the chat loop. Use when editing inference, GGUF, tokenizer, quant, KV cache, tensors, or main.c.
---

# llamini.cpp architecture

C99 mini llama.cpp. Extend existing files. Do not add a `src/` tree or a third-party tokenizer.

## Module map

| File | Role | Current state |
| --- | --- | --- |
| `common.h` | typedefs, limits, BOS/EOS | shared floor |
| `gguf.c` / `gguf.h` | GGUF mmap, real metadata + tensor-info table parser, dequantizer | real: header, all 13 metadata value types (incl. arrays), tensor-info table, alignment-padded data section; dequantizes `F32`/`F16`/`Q4_0`/`Q4_1`/`Q8_0`/`Q4_K`/`Q6_K`, fails loudly (`-1`) on anything else; parsing is hardened against malformed/hostile files (sane bounds on counts and lengths, no multiply-overflow on array sizes — see `test_gguf_reject` in `main.c`) |
| `tensor.c` / `tensor.h` | `malloc` tensors, matmul, vec ops | implemented; `matmul()` only used by `--test`, not the real forward pass (see `linear()` below) |
| `model.c` / `model.h` | real config from GGUF, RMSNorm, SwiGLU, RoPE, `linear()`, GQA causal attention, weight loading | real: `llama_config_from_gguf` reads `llama.*` hyperparams from the file; `llama_load_weights` loads every tensor (embeddings, all 22 layers' attn/FFN, final norm, output projection) by name, fully dequantized to persistent `f32` buffers at load time (~4.4GB resident for TinyLlama-1.1B — see "Still open" #1); `causal_mha` is GQA-aware (32 query heads / 4 KV heads for TinyLlama) with real softmax; `linear()` is pthread-parallelized across output rows above a size threshold (small k/v-proj calls stay serial) and gcc-vectorized (conservative fast-math flags in the Makefile) |
| `kv_cache.c` / `kv_cache.h` | attention cache | init/reset; one instance **per layer** (array in `main.c`), sized by `kv_dim = n_heads_kv * head_dim`, not the full embedding dim |
| `quant.c` / `quant.h` | generic INT4 scale/zp | teaching-only, exercised by `--test`; unrelated to the real `Q4_K`/`Q6_K` decoders in `gguf.c` used to actually load the model |
| `tokenizer.c` / `tokenizer.h` | real SentencePiece-BPE encode/decode + hash table | real: vocab/scores/token_type loaded from GGUF metadata, greedy score-ranked merge (O(n^2) rescan, see file comment), byte-fallback, SPACE-marker (▁) handling; in `SRC`. Has no special-token pre-split pass — a literal `"</s>"` in input text does NOT BPE-merge into the real EOS token, verified empirically (see `main.c`'s `build_chat_prompt_tokens` comment) |
| `generate.c` / `generate.h` | full per-layer autoregressive loop, teacher-forced perplexity | real: `forward_step` (embed -> 22x layer stack -> final norm -> lm_head) is shared by `generate_autoregressive` (prefills the whole prompt through the KV cache before sampling/appending anything — critical fix, see git history) and `compute_perplexity` (teacher-forced, used by `--bench`); `sample_token` supports temperature + top-p nucleus sampling, `temp<=0` dispatches to the original deterministic `greedy_sample`; in `SRC` |
| `main.c` | CLI, `--test`, `--bench`, chat | opens the GGUF file first, derives config from it, then inits/loads the model, loads the vocab, allocates per-layer KV caches, and either chats (`start_chat`, with `build_chat_prompt_tokens` wrapping input in TinyLlama-Chat's own template unless `--raw`) or runs the correctness benchmark (`run_bench`, `--bench`) |

Makefile `SRC`: `main.c tensor.c model.c gguf.c kv_cache.c quant.c tokenizer.c generate.c` → `mini_llama`. `CFLAGS` includes `-pthread` (used by `model.c`'s `linear()`) and conservative fast-math flags (see Makefile comment).

Config is no longer hardcoded — `llama_config_from_gguf` reads it from the file, falling back to TinyLlama-1.1B-shaped defaults (dim 2048, 22 layers, 32 heads, vocab 32000) only for keys a file doesn't define.

CLI flags: `--test` (synthetic-weight unit tests), `--bench` (real-model perplexity + fact-completion evidence), `--temp X` (opt-in sampling), `--raw` (skip the chat template).

## Still open (see README limits, do not pretend these are done)

1. **Fused/on-demand dequantization** — the clear next architectural step, deliberately not attempted. Weights are fully dequantized to `f32` at load time (~4.4GB resident + the ~668MB mmap'd file, both alive at once). The architecturally "correct" fix is to keep weights quantized and dequantize per-block inside `linear()`'s inner loop instead (the dequant kernels in `gguf.c` are already factored out and reusable as-is for this — only *when*/*into what buffer* they're called would change). Two research passes independently rated this **high risk** (3-6 hours; touches `gguf.c`'s dequant switch, `model.h`'s `DecoderLayer` shape, `model.c`'s `linear()`, and breaks `main.c`'s `test_generate` synthetic-weights harness, which writes floats directly into tensors that would no longer exist in `f32` form) for a ~6x reduction in per-token memory traffic and resident footprint. If ever attempted: do it on a single revertable commit on top of a working baseline, give `linear()`'s quantized variant a **per-thread** scratch buffer (a shared static one would repeat the `causal_mha` race-trap one layer deeper), and use "identical generated text on a fixed prompt vs. the pre-refactor binary" as the acceptance bar — a subtle block-offset bug produces plausible-looking *wrong* text, not an obvious failure.
2. Single-turn: every layer's KV cache resets each chat line; no multi-turn conversational memory.
3. Dequantization doesn't cover `Q2_K`/`Q3_K`/`Q5_K`/`Q8_K` (not used by a real `Q4_K_M` file, so untested and deliberately left as "unsupported" rather than shipped unverified).
4. `causal_mha`'s attention is still a naive scalar loop (not parallelized — its compute is negligible next to `linear()`'s, and its `static f32 scores[MAX_SEQ_LEN]` buffer would need to become per-thread first).
5. Sampling has no top-k, repetition penalty, or `--seed` for a reproducible sampled transcript.
6. Chat template (`build_chat_prompt_tokens`, `main.c`) is a best-effort reconstruction from TinyLlama-Chat's published `tokenizer_config.json`, not verified against a real `transformers` install (there isn't one — zero deps).

## How to change it

- Match `common.h` typedefs (`f32`, `u32`, …) and existing names (`gguf_open`, `tensor_create`, `llama_model_init`).
- One concern per `.c` / `.h` pair.
- GGUF weight tensors are row-major `(out_dim, in_dim)` — use `linear()` (`model.c`), not `tensor.c`'s `matmul()`, for anything reading a loaded weight tensor.
- Load files with POSIX `open` / `mmap` as in `gguf.c`.
- A user-supplied `.gguf` is untrusted input — validate sizes/lengths before allocating or reading (see `gguf.c`'s bounds checks and `test_gguf_reject`), don't just trust file-controlled counts.
- Name remaining stubs in README limits; do not paper over the memory footprint, missing sampling knobs, or an unverified chat template.
