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
| `gguf.c` / `gguf.h` | GGUF mmap, real metadata + tensor-info table parser, dequantizer | real: header, all 13 metadata value types (incl. arrays), tensor-info table, alignment-padded data section; dequantizes `F32`/`F16`/`Q4_0`/`Q4_1`/`Q8_0`/`Q4_K`/`Q6_K`, fails loudly (`-1`) on anything else |
| `tensor.c` / `tensor.h` | `malloc` tensors, matmul, vec ops | implemented; `matmul()` only used by `--test`, not the real forward pass (see `linear()` below) |
| `model.c` / `model.h` | real config from GGUF, RMSNorm, SwiGLU, RoPE, `linear()`, GQA causal attention, weight loading | real: `llama_config_from_gguf` reads `llama.*` hyperparams from the file; `llama_load_weights` loads every tensor (embeddings, all 22 layers' attn/FFN, final norm, output projection) by name; `causal_mha` is GQA-aware (32 query heads / 4 KV heads for TinyLlama) with real softmax |
| `kv_cache.c` / `kv_cache.h` | attention cache | init/reset; one instance **per layer** now (array in `main.c`), sized by `kv_dim = n_heads_kv * head_dim`, not the full embedding dim |
| `quant.c` / `quant.h` | generic INT4 scale/zp | teaching-only, exercised by `--test`; unrelated to the real `Q4_K`/`Q6_K` decoders in `gguf.c` used to actually load the model |
| `tokenizer.c` / `tokenizer.h` | real SentencePiece-BPE encode/decode + hash table | real: vocab/scores/token_type loaded from GGUF metadata, greedy score-ranked merge (O(n^2) rescan, see file comment), byte-fallback, SPACE-marker (▁) handling; in `SRC` |
| `generate.c` / `generate.h` | full per-layer autoregressive loop | real: embed -> 22x (RMSNorm -> q/k/v proj -> RoPE -> GQA attention -> o_proj -> residual -> RMSNorm -> gate/up proj -> SwiGLU -> down_proj -> residual) -> final norm -> lm_head; in `SRC` |
| `main.c` | CLI, `--test`, chat | opens the GGUF file first, derives config from it, then inits/loads the model, loads the vocab, allocates per-layer KV caches, and chats using the real tokenizer + generate loop |

Makefile `SRC`: `main.c tensor.c model.c gguf.c kv_cache.c quant.c tokenizer.c generate.c` → `mini_llama`.

Config is no longer hardcoded — `llama_config_from_gguf` reads it from the file, falling back to TinyLlama-1.1B-shaped defaults (dim 2048, 22 layers, 32 heads, vocab 32000) only for keys a file doesn't define.

## Still open (see README limits, do not pretend these are done)

1. No chat template — raw text completion, not `<|system|>/<|user|>/<|assistant|>` formatted chat.
2. Greedy decoding only, no sampling.
3. Single-turn: every layer's KV cache resets each chat line.
4. Naive scalar `linear()`/`causal_mha` — no SIMD/BLAS/threading; loading + generation are genuinely slow (see README).
5. Dequantization doesn't cover `Q2_K`/`Q3_K`/`Q5_K`/`Q8_K` (not used by a real `Q4_K_M` file, so untested and deliberately left as "unsupported" rather than shipped unverified).

## How to change it

- Match `common.h` typedefs (`f32`, `u32`, …) and existing names (`gguf_open`, `tensor_create`, `llama_model_init`).
- One concern per `.c` / `.h` pair.
- GGUF weight tensors are row-major `(out_dim, in_dim)` — use `linear()` (`model.c`), not `tensor.c`'s `matmul()`, for anything reading a loaded weight tensor.
- Load files with POSIX `open` / `mmap` as in `gguf.c`.
- Name remaining stubs in README limits; do not paper over the missing chat template, sampling, or performance characteristics.
