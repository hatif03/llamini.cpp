---
name: llamini-architecture
description: Maps llamini.cpp modules, stubs vs target inference, and how to extend GGUF, tokenizer, quant, and the chat loop. Use when editing inference, GGUF, tokenizer, quant, KV cache, tensors, or main.c.
---

# llamini.cpp architecture

C99 mini llama.cpp. Extend existing files. Do not add a `src/` tree or a third-party tokenizer. When `tokenizer.c` / `generate.c` become real, add them to `Makefile` `SRC`.

## Module map

| File | Role | Current state |
| --- | --- | --- |
| `common.h` | typedefs, limits, BOS/EOS | shared floor |
| `gguf.c` / `gguf.h` | GGUF mmap | magic + header; metadata/tensor table skipped; `tensor_offset` is `sizeof(GGUFHeader)` |
| `tensor.c` / `tensor.h` | `malloc` tensors, matmul, vec ops | implemented |
| `model.c` / `model.h` | TinyLlama-shaped structs, RMSNorm, SwiGLU, RoPE, causal MHA | structs + ops; weights not wired from GGUF |
| `kv_cache.c` / `kv_cache.h` | attention cache | init/reset |
| `quant.c` / `quant.h` | INT4 scale/zp | generic INT4, not GGUF `Q4_K_M` blocks |
| `tokenizer.c` / `tokenizer.h` | text ↔ ids | space-split demo vocab; **not in `SRC`** |
| `generate.c` / `generate.h` | greedy + autoregressive loop | demo loop; **not in `SRC`** |
| `main.c` | CLI, `--test`, chat | keyword `text_to_token` / `token_to_text`; fake logits |

Makefile `SRC` today: `main.c tensor.c model.c gguf.c kv_cache.c quant.c` → `mini_llama`.

Hardcoded TinyLlama-shaped config in `main.c`: dim 2048, 22 layers, 32 heads, vocab 32000, `MAX_SEQ_LEN`.

## Target (do not pretend it is done)

1. Parse GGUF metadata and the tensor info table; map named tensors into `LLaMAModel`.
2. Decode real quant types used by the demo weight (e.g. `Q4_K_M`) or document that only F32/F16 is supported.
3. SentencePiece / BPE-style tokenizer from GGUF vocab — written here, not imported.
4. Forward pass: embed → layers (attn + FFN) → norm → lm_head → greedy (or documented sampler).
5. Wire `tokenizer.c` and `generate.c` into `SRC`; delete the keyword tables in `main.c` once they are unused.

## How to change it

- Match `common.h` typedefs (`f32`, `u32`, …) and existing names (`gguf_open`, `tensor_create`, `llama_model_init`).
- One concern per `.c` / `.h` pair.
- Load files with POSIX `open` / `mmap` as in `gguf.c`.
- Name remaining stubs in README limits; do not paper over keyword chat or skipped GGUF metadata.
