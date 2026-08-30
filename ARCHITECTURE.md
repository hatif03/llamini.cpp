# ARCHITECTURE.md — a guide to this codebase, for humans

This file is for a reader who knows basic C but has never seen this project
before — a judge skimming the repo, a contributor, or future-you in six
months. It explains what each file does and how they fit together, in plain
language. (There's also a `.claude/skills/llamini-architecture/SKILL.md` in
this repo — that one is written for an AI coding assistant working *on* the
code and is much denser; this file is the human-readable version.)

If you just want to know what the project does or its limits, read
[README.md](README.md) first. If you want to know what stood in for every
package this project would normally reach for, read [STDLIB.md](STDLIB.md).
If you want the raw, chronological history of every change (including the
mistakes), read [BUILDLOG.md](BUILDLOG.md). This file is none of those — it's
a map of the *code*, not the project's story.

## Start here: what this program actually does

In one sentence per step: **open a `.gguf` model file → figure out which
architecture and shape it describes → load its weights → turn your text into
numbers → run those numbers through the model, one layer at a time → turn
the model's answer back into text → repeat.**

Everything in this repo exists to do one of those six steps. There's no
framework, no plugin system, no configuration language — just under 3,000
lines of C across a dozen files, each responsible for one step (or one part
of one step).

## The whole pipeline, told as a walkthrough

Follow one real command, `./llamini model.gguf`, through the code:

1. **`main.c`** parses your command-line arguments (`--test`, `--bench`,
   `--raw`, `--temp X`) and calls `gguf_open()` on the file path you gave it.
2. **`gguf.c`** memory-maps the file (`mmap`, not a `read()` into a buffer —
   this matters later) and parses the real GGUF binary layout: a header, a
   metadata key/value table, a tensor-info table, then the actual tensor
   bytes. It doesn't touch the weight data yet — just figures out where
   everything is and what every tensor's shape and quantization format is.
3. **`main.c`** asks the parsed file one question via `gguf_get_str`: what
   does `general.architecture` say? If it says `"gpt2"`, everything from here
   on happens in **`gpt2.c`** instead — GPT-2 is different enough (see below)
   that it gets its own entirely separate code path, not a config flag.
   Otherwise, you're on the "LLaMA-family" path, which actually covers three
   architectures (LLaMA, Qwen2, Gemma) because they're similar enough to
   share code.
4. **`model.c`**'s `llama_config_from_gguf()` reads the file's own
   hyperparameters (hidden size, layer count, head counts, vocabulary size...)
   out of the metadata table gguf.c already parsed, and sets a few
   per-architecture behavior switches (does this model use RoPE's "NORM" or
   "NEOX" pairing convention? SwiGLU or GeGLU gating? does attention have a
   bias?) based on what `general.architecture` said.
5. Still in **`model.c`**, `llama_load_weights()` walks through every named
   tensor the model needs (`token_embd.weight`, `blk.0.attn_q.weight`, ...)
   and either dequantizes it immediately into an ordinary `f32` array (for
   small tensors — norms, biases) or marks it "lazy" (for the big weight
   matrices — see "CPU optimization" below) so it gets dequantized in small
   pieces later, on demand, straight out of the memory-mapped file.
6. **`tokenizer.c`** loads the vocabulary embedded in the same GGUF file (no
   separate vocab file) and turns your typed text into a sequence of integer
   token ids.
7. **`generate.c`**'s `generate_autoregressive()` is the actual loop: for
   each token position, run `forward_step()` (embed the token → normalize →
   attention → normalize → feed-forward → repeat for every layer → produce a
   probability over every possible next token), pick one (greedily, or by
   sampling if you passed `--temp`), and repeat until you hit an end token or
   run out of room.
8. **`main.c`** decodes the resulting token ids back into text via
   `tokenizer.c`'s `token_to_text()` and prints it.

```
 ./llamini model.gguf
        │
        ▼
   main.c ──opens file──▶ gguf.c (mmap + parse header/metadata/tensor-info)
        │
        ├─ general.architecture == "gpt2"? ──yes──▶ gpt2.c (its own model,
        │                                            own forward pass)
        │
        no (LLaMA-family: llama / qwen2 / gemma)
        │
        ▼
   model.c ── reads hyperparameters, loads weights (tensor.c holds them)
        │
        ▼
   tokenizer.c ── your typed text ──▶ token ids
        │
        ▼
   generate.c ── forward_step() through every layer, per token,
        │         using kv_cache.c to remember earlier positions
        ▼
   tokenizer.c ── token ids ──▶ text
        │
        ▼
   main.c ── prints it, loops for your next line
```

## File-by-file reference

| File | What it's responsible for | One thing worth knowing |
| --- | --- | --- |
| `common.h` | Shared type aliases (`f32`, `u32`, ...) and a few size limits (`MAX_SEQ_LEN`, etc.) used everywhere. | No `.c` file — it's pure declarations, included by almost every other header. |
| `gguf.c` / `gguf.h` | Parses the GGUF binary file format and turns quantized tensor bytes into real floating-point numbers. | Dequantization isn't all-or-nothing: `gguf_dequantize_tensor` converts a whole tensor at once, but `gguf_dequantize_rows` (added for the CPU-optimization pass) converts just a row-range at a time, straight out of the memory-mapped file, so a big weight matrix never has to exist fully in memory as floats. |
| `tensor.c` / `tensor.h` | Defines the `Tensor` struct (a shape + a float buffer) and basic tensor operations. | A `Tensor` can be "lazy" (its data lives in the still-mapped GGUF file, not in memory, until something asks for a specific row) or "eager" (a normal, fully-populated float array) — see `linear()` in `model.c` for where that distinction actually matters. |
| `model.c` / `model.h` | The shared transformer building blocks (attention, normalization, the gated feed-forward layer, RoPE) and the model-loading logic for the LLaMA/Qwen2/Gemma family. | `linear()` — the function that multiplies a vector by a weight matrix, called dozens of times per generated token — is the single busiest piece of code in the whole project, and where both the threading and the memory-optimization work live. |
| `gpt2.c` / `gpt2.h` | A second, complete forward pass just for GPT-2, which differs enough from LLaMA's shape (one fused Q+K+V tensor instead of three separate ones, learned position embeddings instead of RoPE, LayerNorm instead of RMSNorm, no gating in its feed-forward layer) that folding it into `model.c` would have made that file harder to read, not easier. | Still reuses `model.c`'s `linear()` and `causal_mha()` — those two functions turn out to be general enough that GPT-2 doesn't need its own versions. |
| `kv_cache.c` / `kv_cache.h` | Stores the attention "memory" (keys and values from every previous token) so each new token doesn't have to recompute attention over the whole conversation from scratch. | There's one `KVCache` **per transformer layer**, not one shared cache for the whole model — `main.c` allocates an array of them, sized to the model's layer count. |
| `quant.c` / `quant.h` | A small, self-contained INT4 quantize/dequantize pair. | This is a **teaching-only demonstration**, not part of the real model-loading path — it exists so a reader can see the *idea* of quantization (pack two 4-bit values per byte, one shared scale) without wading through the real GGUF block formats' bit-twiddling. The real dequantizers real models actually use live in `gguf.c`. |
| `tokenizer.c` / `tokenizer.h` | Turns text into token ids and back, for both tokenizer conventions real GGUF files use. | One `Vocab` struct handles two genuinely different schemes (SentencePiece-style, used by LLaMA; byte-level BPE, used by GPT-2/Qwen2), picked at load time by reading the file's own `tokenizer.ggml.model` metadata — not a compile-time choice. |
| `generate.c` / `generate.h` | The autoregressive generation loop, token sampling, and the correctness benchmark (teacher-forced perplexity). | `forward_step()` — one token's full pass through every layer — is called by both the interactive chat loop and the benchmark, so there's exactly one place the "real" forward pass is written; nothing about correctness testing uses a separate, possibly-diverging implementation. |
| `main.c` | The command-line entry point: argument parsing, opening the file, architecture dispatch, the chat loop, and every `--test` unit test. | The GPT-2 vs. LLaMA-family branch happens *before* any model-specific config is read — `main.c` peeks `general.architecture` first and then never lets the two paths cross. |

## Where do I look if I want to understand X?

- **How is the GGUF binary format actually parsed?** `gguf_open()` in `gguf.c`.
- **How does quantized data become real numbers?** `dequantize_range()` (the
  shared internal implementation), and its two public entry points
  `gguf_dequantize_tensor()`/`gguf_dequantize_rows()`, all in `gguf.c`.
- **How does attention work?** `causal_mha()` in `model.c` — it's shared by
  every architecture (GPT-2 included), since plain multi-head attention turns
  out to be grouped-query attention with the group size set to 1.
- **How does the model figure out its own shape from the file?**
  `llama_config_from_gguf()` in `model.c` (or `gpt2_config_from_gguf()` in
  `gpt2.c` for GPT-2).
- **How does text become tokens, and back?** `text_to_tokens()` /
  `token_to_text()` in `tokenizer.c` — both dispatch internally on which
  tokenizer convention the loaded vocab uses.
- **Where's the actual generation loop?** `generate_autoregressive()` in
  `generate.c` (or `gpt2_generate()` in `gpt2.c`).
- **Where's the CPU-performance work (the thread pool, the on-demand
  dequantization)?** `linear()`'s dispatch logic in `model.c` — see README's
  "CPU optimization" section for *why* it's built the way it is, and
  STDLIB.md's substitution log for exactly what it's simpler than ggml's own
  real approach.
- **How do I add support for a new architecture?** Read `model.c`'s
  `llama_config_from_gguf()` for how an existing one (say, `qwen2`) is
  detected and configured — most new LLaMA-shaped architectures are a matter
  of adding a new per-architecture branch there, not writing a new forward
  pass. A genuinely different shape (fused QKV, a different normalization,
  no RoPE at all) is what earns a file like `gpt2.c` instead.

---

*See [README.md](README.md) for what this project does and its honest
limits, [STDLIB.md](STDLIB.md) for every package this replaced and what
stood in for it, [BUILDLOG.md](BUILDLOG.md) for the raw engineering history,
and [WRITEUP.md](WRITEUP.md) for the narrative version of that history.*
