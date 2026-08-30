# STDLIB.md

Every place this project would normally reach for a package, and the libc/POSIX
stand-in used instead. See [README.md](README.md) "Limits" for what each stand-in
still can't do.

## Package Killer: what this replaces

Running a GGUF language model from scratch normally means installing at
least one of these:

| Package people actually install | Why | Replaced here by |
| --- | --- | --- |
| `ggml` / `llama.cpp` linked as a library | GGUF parsing, block dequantization, transformer kernels | `gguf.c` (header/metadata/tensor-info parser + `F32`/`F16`/`Q4_0`/`Q4_1`/`Q5_0`/`Q8_0`/`Q4_K`/`Q5_K`/`Q6_K` decoders), `model.c` (the LLaMA/Qwen2/Gemma forward pass — RMSNorm, RoPE in both its NORM and NEOX pairings, GQA/MQA attention, gated SwiGLU/GeGLU FFN, `linear`), `gpt2.c` (a second, wholly separate forward pass for GPT-2's architecture) |
| `llama-cpp-python`, `ctransformers` (pip) | Python bindings over the above | Not needed — the CLI *is* the binary |
| `sentencepiece` (pip / libsentencepiece) | SPM-BPE encode/decode | `tokenizer.c`'s SentencePiece path, driven entirely by the GGUF file's own `tokenizer.ggml.*` metadata |
| A byte-level BPE tokenizer (as ships inside `transformers`/`tiktoken`-adjacent tooling) | GPT-2/Qwen2-style tokenization | `tokenizer.c`'s second, independent path (same file, different vocab convention — see the substitution log), driven by the same GGUF file's `tokenizer.ggml.merges` |
| `numpy` / a BLAS | tensor storage + matmul | `tensor.c` and `linear()` in `model.c` (a persistent pthread pool, gcc-vectorized dot products, and on-demand row dequantization straight from the mmap'd file instead of a fully-materialized `f32` copy — see the substitution log below) |

### What this is NOT (scope, stated up front)

- **Not a ggml reimplementation.** It implements the subset needed to load
  and run real files from four architecture families (see README's "Other
  model families"), not ggml's graph engine, backends, or full op set.
- **Quantization coverage is partial:** `F32`, `F16`, `Q4_0`, `Q4_1`,
  `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` — every format the four verified
  files actually use (found by inspecting real tensor types, not assumed
  from a file's "Q4_K_M"-style name, which turned out not to guarantee
  every tensor uses only that scheme). `Q2_K`, `Q3_K`, `Q8_K`, and the
  `IQ*` family are explicitly rejected (`-1`), never silently misread.
- **Architecture coverage is LLaMA, Qwen2, GPT-2, and Gemma — all four now
  verified end-to-end.** Gemma-2b was initially untestable on this dev
  machine's memory budget; the CPU optimization pass's on-demand
  dequantization (see below) fixed that. Mistral (its GGUF architecture is
  literally `"llama"`, so it would load as-is) and Gemma2 (real added
  complexity for the same memory cost as Gemma1) were not attempted. No
  LoRA, no GPU, no batching.
- **Not numerically verified against ggml.** No reference implementation
  was run to cross-check outputs bit-for-bit, for any of the four
  architectures. Correctness evidence is `--bench`'s teacher-forced
  perplexity against each file's own random-baseline ceiling plus
  known-fact completions (see README's "Other model families" table for
  all four), not bit-exact comparison.

## Substitution log

| Instead of | Used | Rationale |
| --- | --- | --- |
| `ggml` / llama.cpp's GGUF loader | Hand-written binary parser in `gguf.c` (`open` + `mmap` + a bounds-checked cursor over the mapped bytes) | Parses the real GGUF layout: header, metadata key/value table (all 13 value types, including nested arrays), tensor-info table, alignment-padded tensor data section. Every multi-byte read is bounds-checked against the file size so a truncated/malformed file fails the parse instead of reading past the mapping. |
| `ggml`'s quantized-tensor dequantizers | Hand-rolled block decoders in `gguf.c` for `F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q8_0`, and the "K-quant" super-block formats `Q4_K`, `Q5_K`, `Q6_K` (every format the four verified real files actually use) | Implemented from the public block layouts (block sizes, scale/min bit-packing), not copied from ggml's source. Verified against four real downloaded files: dequantized embedding/output rows come back as small, varied, non-NaN floats consistent with real trained weights, and feeding them through the full forward pass produces recognizable, architecture-appropriate English (and, for Qwen2.5, correctly-round-tripped Chinese) rather than noise. `Q5_0` and `Q5_K` were added *after* TinyLlama's file (which doesn't use them) worked, when Qwen2.5's and GPT-2's real files turned out to use them despite being labeled "Q4_K_M" — found by probing real tensor types, not assumed. Anything still outside this set (`Q2_K`, `Q3_K`, `Q8_K`, ...) is reported as unsupported (`-1`), never silently misread. |
| `sentencepiece` | Hand-written SentencePiece-style BPE encoder/decoder in `tokenizer.c`, driven entirely by the GGUF file's own `tokenizer.ggml.tokens` / `.scores` / `.token_type` metadata | Real vocab (32000-256128 pieces across the four verified files), real greedy score-ranked merge algorithm, real byte-fallback (`<0xXX>` pieces) for anything outside the vocab, real SPACE-marker (▁) handling on both encode and decode. Simplification: merge selection is an O(n^2) rescan per merge rather than a priority-queue heap — fine for chat-length input, would need a heap for document-length text (noted in `tokenizer.c`). |
| A byte-level BPE tokenizer (GPT-2/Qwen2's real convention, a genuinely different vocab format from SentencePiece above, not a config tweak) | A second encoder/decoder path in `tokenizer.c`, dispatched via a `Vocab.is_bpe` flag detected from `tokenizer.ggml.model == "gpt2"`: every raw byte is remapped through a fixed byte<->printable-unicode table (GPT-2's own public convention, reimplemented from the described algorithm) before BPE runs, and merge priority is rank (position in an explicit `tokenizer.ggml.merges` list) instead of a per-token score | Verified against two real files (Qwen2.5-0.5B, GPT-2-124M) including correct multi-byte UTF-8 round-tripping (Qwen2.5 continuing "hello" into fluent Chinese). ponytail: pre-tokenization is whitespace-based, not GPT-2's real Unicode-category-aware regex splitter (no `\p{L}`/`\p{N}` tables in this project) — disclosed, not silently assumed equivalent. |
| A hashmap/dictionary library | Hand-rolled open-addressing string->id table (`StrMap` in `tokenizer.c`), FNV-1a hashing with linear probing | With a real ~32000-entry vocab and BPE doing many lookups per input word, a linear scan is genuinely too slow for interactive use — unlike a tiny fixed vocab, this is infrastructure the project actually needs, not a gratuitous abstraction. |
| BLAS / `ggml`'s matmul kernels | `linear()` in `model.c`: a hand-written row-dot-product loop matching GGUF's native `(out_dim, in_dim)` row-major tensor layout, dispatched across a persistent thread pool for eager (small/norm/bias) tensors or on-demand row-chunk dequantization for large weight matrices (see the threading/quantized-dot-product rows below) | Deliberately not `tensor.c`'s `matmul()` (which expects a `[k,n]`-shaped operand) — reusing that would require transposing every loaded weight matrix first. No hand SIMD intrinsics (would need non-POSIX/non-stdlib headers like `immintrin.h`); relies on `-O3`/`-march=native` auto-vectorizing the inner dot-product loop instead — see README "Slow". |
| A tensor/ndarray library (numpy, Eigen) | The `Tensor` struct (`tensor.h`) plus `tensor_create`/`vec_add`/`vec_scale`, and raw `f32*` buffers for the hot forward-pass path | Just a shape array, a size, and a `malloc`'d `f32*`; no broadcasting or strides beyond what the code explicitly loops over. |
| An IEEE-754 half-float library | Hand-written `f16_to_f32` bit-manipulation conversion in `gguf.c` | GGUF's `F16` tensors and every K-quant block's per-superblock scale are stored as raw 16-bit halfs; this is the standard public bit layout (sign/exponent/mantissa), not vendored from anywhere. |
| A quantization library (e.g. `ggml`'s block quantizers) | Hand-rolled scale/zero-point INT4 pack/unpack in `quant.c` | A separate, simpler teaching implementation (generic linear INT4, exercised by `--test`), unrelated to the real `Q4_K`/`Q6_K` GGUF decoders above. |
| `getopt_long` / a CLI-parsing package (clap-style) | `argc`/`argv` + `strcmp` in `main.c` | Only two shapes are needed (`--test`, or a single `.gguf` path), so a full flag parser would be pure ceremony. |
| A unit test framework (CTest, Unity, Check) | `--test` flag running in-process `assert()`/`printf` checks (`run_all_unit_tests` in `main.c`) | C ships no stdlib test framework; each `test_*` function is a self-contained smoke test, including a synthetic-weights run of the full per-layer GQA forward pass, with no fixtures or discovery magic. |
| A logging framework | Plain `printf` / `fprintf(stderr, ...)` | No log levels, no formatting pipeline — stdout for normal output, stderr for warnings/errors, per Track A CLI conventions. |
| A "safe string" / bounds-checked I/O library | Manual range checks throughout `gguf.c`'s cursor and `gguf_dequantize_tensor` before any `memcpy` out of the mmap'd region | A user-supplied `.gguf` is untrusted input; every read fails closed (`-1` / a sticky cursor error) instead of reading past the mapping on a short or malformed file. |
| A dependency-injection/config framework for model hyperparameters | `llama_config_from_gguf` in `model.c` reads `general.architecture` first, then builds each key as `"<arch>.<suffix>"` (`qwen2.embedding_length`, `gemma.attention.key_length`, ...) instead of a hardcoded `"llama."` prefix -- every architecture's hyperparameters live under its own namespace, not just LLaMA's | The model's shape (and, via a small per-architecture table for quirks the metadata itself doesn't encode -- RoPE pairing convention, gate activation, QKV bias, embedding scale -- the model's *behavior*) is whatever the GGUF file's own architecture says it is; hardcoded TinyLlama-1.1B numbers are only the fallback defaults for a key no file defines. |
| A sampling/inference library's temperature+top-p logic | Hand-written `sample_token` in `generate.c`: numerically-stable softmax, `qsort` (libc) by probability, nucleus cutoff, then `rand()`/`srand()` (libc) to draw | `temp <= 0` dispatches straight to the pre-existing deterministic `greedy_sample`, so enabling this feature couldn't silently change the already-verified default behavior. |
| A chat-template/tokenizer library's special-token handling | `build_chat_prompt_tokens` in `main.c` inserts the real `eos_id` token programmatically between prompt segments, instead of embedding `"</s>"` as literal text | Found by testing, not assuming: `</s>` names a real EOS control token in this vocab, but typing it as text does **not** BPE-merge back into that single token (verified by tokenizing the literal string and finding zero occurrences of `eos_id` in the result) — this tokenizer has no special-token pre-split pass, unlike a real `transformers`/`sentencepiece` tokenizer's chat-template handling. Building the prompt at the token level, not the string level, sidesteps needing that pass at all. |
| A model-manager package (`huggingface_hub`, or shelling out to `curl`/`wget`) for `--setup`'s "fetch a model" step | `main.c` prints the exact `curl` command and waits for you to run it yourself, then detects the file via `stat()` (POSIX) and launches straight into it | HuggingFace serves only over HTTPS; implementing TLS from scratch is out of scope for a weekend, and the only two ways around that — a crypto/TLS library, or shelling out to `curl`/`wget` — are both explicitly forbidden by this hackathon's own rules (a vendored dependency, or a "hidden dep" on a separately-installed tool). Printing the command instead of hiding it behind a library call is the compliant version of the same convenience: the network request happens in the open, in a line you can read before running it. |
| A benchmarking/profiling tool (`/usr/bin/time -v`, `psutil`) for reporting memory/throughput | `getrusage(RUSAGE_SELF, ...)` (POSIX, `<sys/resource.h>`) plus `clock_gettime`, both called directly from `main.c`, printed as a session summary after every run | Real numbers this process actually used, not modeled or estimated — verified against `/usr/bin/time -v`'s independently-measured RSS for the same file (811MB self-reported vs. ~831MB externally measured, same order, ordinary run-to-run variance) rather than trusted on its own. |
| `ggml`'s persistent worker-thread pool | A hand-rolled `ThreadPool` in `model.c` (mutex + two condvars + a generation counter, workers created once and blocked between dispatches) | Researched from `ggml-cpu.c`'s actual design (a persistent pool with work-stealing via an atomic chunk counter), reimplemented independently and much simpler for this project's single hot call site (`linear()`) — no work-stealing, no NUMA fallback. Replaced the original per-call `pthread_create`/`pthread_join` (~9 spawns/layer/token); verified bit-identical output before/after (BUILDLOG.md's CPU optimization phase). |
| `ggml`'s quantized dot-product kernels (matmul without ever dequantizing) | On-demand row-chunk dequantization in `linear_lazy()`/`gguf_dequantize_rows` (`model.c`/`gguf.c`), reading straight out of the mmap'd file instead of a persistent `f32` copy of every weight | Researched from `ggml-cpu/quants.c`'s real approach: CPU batch-1 inference is memory-bandwidth bound, so never materializing a dequantized tensor is the actual lever, not raw FLOPs. This project's version is deliberately simpler than ggml's true int8×int8 quantized dot product with a runtime-quantized activation vector — it dequantizes each row-chunk to `f32` into a small scratch buffer and reuses the existing `f32` dot product (see "Deliberately not substituted" below for exactly what that skips). Verified bit-identical output for TinyLlama, Qwen2.5, and GPT-2 against the pre-change binary, and a measured ~6-7x resident-memory reduction (from ~4.4-4.9GB down to ~700MB-1.5GB) that let Gemma-2b run end-to-end for the first time — see README's "CPU optimization". |

## Deliberately not substituted (out of scope, disclosed instead of faked)

- **No top-k, repetition penalty, or `--seed`.** `--temp` sampling (see table above) covers temperature + top-p only; no reproducible-sampled-transcript flag yet.
- **Chat template not verified against a reference tokenizer run.** `build_chat_prompt_tokens` (`main.c`) is derived from TinyLlama-Chat's published `tokenizer_config.json`, but this project has no `transformers` install to confirm the exact resulting token sequence bit-for-bit. See README Limits.
- **Mistral and Gemma2 were not attempted at all.** Mistral's GGUF architecture is literally `"llama"` (it would load with the existing code, demonstrating no new capability); Gemma2 has real added complexity (logit/attention softcapping, sliding-window attention, a `head_dim` that isn't `dim/n_heads`) for the same memory cost as Gemma1. Neither is a stdlib gap — both are scoping decisions, not missing infrastructure.
- **No Qwen-family (ChatML) chat template.** Qwen2 uses `<|im_start|>`/`<|im_end|>`, a different convention from TinyLlama-Chat's `<|system|>/<|user|>/<|assistant|>` that `build_chat_prompt_tokens` implements. Not built; Qwen2.5 is exercised here in raw-completion mode only.
- **No true quantized dot product, and the lazy path isn't thread-pooled.** `linear_lazy()` (`model.c`) still converts each row-chunk to `f32` before the dot product, unlike ggml's real int8×int8 kernels that never produce a float until the final scaled sum — this project's version buys the memory-footprint and bandwidth win without the integer-SIMD complexity, at the cost of some avoidable float conversion work. It's also deliberately serial, not dispatched to the thread pool: chunking by a fixed row count (not thread count) is what keeps peak transient memory bounded regardless of a tensor's width (critical for Gemma's 256128-row embedding table — see BUILDLOG.md's CPU optimization phase for the reasoning), and CPU batch-1 inference being memory-bandwidth-bound means N threads racing for the same bytes isn't an obvious win anyway. Revisit with per-worker scratch buffers if a real profile ever shows otherwise. Each chunk's scratch buffer is also a plain `malloc`/`free` per call rather than a reused per-tensor buffer — simpler, correct, but real allocator churn on the hot path.
- **`gpt2.c`'s weights were not brought into the on-demand dequantization pass.** Only the LLaMA-family path (`model.c`, shared by LLaMA/Qwen2/Gemma) got lazy tensors; GPT-2 still fully dequantizes to `f32` at load time. Measured effect: GPT-2's resident memory (~831MB) is actually *higher* than real llama.cpp's for the same file (~169MB) — see README's "Benchmarks". A scoping decision (GPT-2's weights are tiny, ~124M params, so the absolute cost is small), not an oversight, but disclosed rather than left for a reader to discover by comparing the numbers themselves.

## Benchmark methodology (not a project dependency)

README's "Benchmarks" section cites real numbers from an unmodified,
upstream `llama.cpp` (`ggml-org/llama.cpp`, commit `0b5be7e`), built from
source with `cmake` purely to generate comparison data on this same
machine. This is **not vendored, not shipped, not referenced by any code
path in this project, and never invoked by the submitted binary** — it was
cloned and built once, by hand, in a scratch directory entirely outside
this repo (`~/llama.cpp-bench` in WSL, never under `d:\llamini.cpp`), run
directly (`llama-bench`), and its output copied into README/this file as
data. The zero-dependency rule (`AGENTS.md`'s "No hidden deps": don't
`system()`/`popen` a separately-installed tool) governs what the *shipped
program* does at runtime — it says nothing about a developer building a
reference implementation once, by hand, to get honest comparison numbers,
any more than the research agents that read real ggml source this session
made ggml a dependency of this project. `cmake` was installed in WSL
(`apt-get install cmake`) specifically to build that external reference; it
is not required to build or run llamini.cpp itself (see the Makefile —
`gcc`/`make` only).

**Landscape comparison** (researched, not independently rebuilt/benchmarked
on this machine for the non-llama.cpp rows — see BUILDLOG.md's CPU
optimization phase for sourcing):

| Project | Vendored ML source | Architectures (from-scratch, no vendoring) | Build tools |
| --- | --- | --- | --- |
| **llamini.cpp** | 0 bytes | 4, auto-detected from GGUF metadata | `gcc`/`make` only |
| `llama2.c` (karpathy) | 0 bytes | 1, hardcoded (Llama-2 shape, custom `.bin` format, no GGUF) | `gcc`/`make` only |
| `llama.cpp` | n/a (it's the reference) | many, via `ggml`'s own graph engine | `cmake` + a C++ compiler |
| `llamafile` | ~26MB (`llama.cpp`+`ggml` vendored as a git submodule) | inherits llama.cpp's | its own cosmocc toolchain |

The honest framing: `llama2.c` is the only other genuinely from-scratch,
zero-vendored comparator, and it's smaller in raw line count than this
project — the defensible claim isn't "fewer lines," it's **architecture
generality per line**: 4 GGUF-metadata-auto-detected architectures here
vs. `llama2.c`'s 1 architecture hardcoded regardless of size. Against
`llamafile`, the clean claim is genuinely zero vendored ML source (0 bytes)
vs. a real, git-submodule-pinned dependency.
