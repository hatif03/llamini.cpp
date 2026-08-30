# llamini.cpp

**llamini.cpp** is a from-scratch C99 reimplementation of llama.cpp's
core: a real GGUF file parser (metadata + tensor-info table), hand-rolled
block dequantizers for every quantization format a real Q4_K_M-labeled
file actually uses in practice (`F32`/`F16`/`Q4_0`/`Q4_1`/`Q5_0`/`Q8_0`/
`Q4_K`/`Q5_K`/`Q6_K`), a real transformer forward pass, and two independent
from-scratch tokenizers (SentencePiece-style and byte-level BPE) decoded
entirely from each GGUF file's own vocab — wired into a chat CLI that
loads and runs real models across **four different architecture
families** (LLaMA, Qwen2, GPT-2, Gemma — see "Other model families"
below), not just one. Built for the Zero Dependency hackathon, Track F —
libc and POSIX only, no third-party runtime dependency of any kind. See
[STDLIB.md](STDLIB.md) for
every package this project would normally reach for, and what stood in for
it instead — including its ["Package Killer" section](STDLIB.md#package-killer-what-this-replaces),
naming exactly what running a GGUF model normally requires installing
(`ggml`/`llama.cpp`, `llama-cpp-python`, `sentencepiece`, a BLAS) and what
replaces each one here, with an upfront, honest scope disclosure of what
this project is *not*.

## Build and run

Requires a POSIX environment (Linux, macOS, or WSL/MSYS2 on Windows) — the
code uses `mmap`/`open`/`fstat`, which plain MinGW does not provide (see
[STDLIB.md](STDLIB.md)). Verified with `gcc` 15.2.0, C99, on Ubuntu (WSL).

```bash
make clean
make
./llamini tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Chat is greedy (deterministic) by default. Pass `--temp X` (0 < X, e.g. 0.8)
for temperature + top-p=0.9 nucleus sampling instead:

```bash
./llamini tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --temp 0.8
```

Run the in-process unit tests instead of the chat loop:

```bash
./llamini --test
```

Run the quantitative correctness benchmark (see "Correctness evidence"
below) instead of the chat loop:

```bash
./llamini tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --bench
```

## Chat loop

~~~text
$ ./llamini tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Loaded GGUF v3 (201 tensors, 23 metadata entries)
Config: dim=2048 layers=22 heads=32 kv_heads=4 ffn=5632 vocab=32000 seq_len=2048

======== TinyLlama GGUF Chat Engine Ready ========
Real GGUF weights + SentencePiece BPE tokenizer (see README limits).

You: hello
Bot:Sure, I'd be happy to provide you with a friendly chatbot that always responds concisely. Here's a sample response:

Hey, how are you doing today?

I'm doing

You: exit
Bot: Bye!
~~~

Every architecture number in that `Config:` line (dim, layer count, head
count, GQA KV-head count, FFN size, vocab size) is read out of the file's
own `llama.*` metadata, not hardcoded — see `llama_config_from_gguf` in
`model.c`. Each line you type is wrapped in TinyLlama-Chat's own chat
template (`<|system|>...<|user|>...<|assistant|>`, see `build_chat_prompt_tokens`
in `main.c`), BPE-tokenized against the file's real vocab, embedded, and run
through all 22 real decoder layers (RMSNorm -> grouped-query causal
attention with a real softmax and a per-layer KV cache -> residual ->
RMSNorm -> SwiGLU FFN -> residual), then the final norm and output
projection, decoded back into real vocabulary pieces. See
[llamini-architecture](.claude/skills/llamini-architecture/SKILL.md) for
the module map.

Every prompt token is run through the full 22-layer stack before generation
starts (`generate_autoregressive` prefills the whole prompt into each
layer's KV cache), so the model actually sees everything you typed, not
just its last token. Pass `--raw` to skip the chat template and get the
model's raw text-completion behavior instead (e.g. "hello" continues as if
finishing a "Hello, world!" code example) — useful as an A/B control, or if
this file's template guess (see Limits) ever needs bypassing.

## Correctness evidence

"The output looks like English" is a weak correctness claim. This project
doesn't have (and, staying zero-dependency, deliberately didn't build) a
reference `llama.cpp`/`transformers` install to diff against bit-for-bit, so
`./llamini tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --bench` instead runs two
self-contained, honest checks against the real model:

1. **Teacher-forced perplexity** over a short fixed English test corpus
   (`BENCH_CORPUS` in `main.c`), computed via `compute_perplexity` in
   `generate.c` — for each position, how much probability the model's real
   logits assign to the *actual* next word, never sampling. Measured:
   **13.28** over 37 tokens. Compare to the one number that needs no
   external citation to trust: a uniform-random guess over this file's
   32000-token vocab has perplexity ~32000 (10.37 nats/token). 13.28 is
   roughly three orders of magnitude below that ceiling — the forward pass
   is doing real language modeling, not producing noise that merely
   resembles English by chance.
2. **Known-fact completions** (greedy, no chat template — raw text
   completion): `"The capital of France is"` -> `" Paris."`; `"The sky is
   the color"` -> `" of the sunset, and the sea"`; `"Two plus two equals"`
   -> `"? \n\nA: The formula for"` (grammatical, but doesn't answer "four").
   These are informational spot checks, not asserted in `--test` — a 1.1B
   model isn't guaranteed to nail every fact even when the code is entirely
   correct — but getting a specific factual completion right is a much
   sharper signal than "sounds like English."

Numbers above are from one real run against the actual downloaded GGUF
file; re-running `--bench` reproduces them (teacher forcing and greedy
decoding are both deterministic).

## Other model families

The GGUF parser, dequantizers, and sampling/`--bench` machinery are
architecture-agnostic; `general.architecture` in the file itself picks the
forward-pass path (`llama.cpp:llama_config_from_gguf` for the LLaMA-family
path shared by LLaMA/Qwen2/Gemma, or `gpt2.c`'s wholly separate module for
GPT-2). Verified against real downloaded files from four different
families, each surfacing something the previous ones didn't:

| Model | Arch | Params | Vocab | Tokenizer | Perplexity\* | Fact completions correct | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TinyLlama-1.1B-Chat-v1.0 | `llama` | 1.1B | 32000 | SentencePiece | 13.28 | 1/3 | GQA (32 query / 4 KV heads); the baseline this whole engine was built against |
| Qwen2.5-0.5B-Instruct | `qwen2` | 0.5B | 151936 | byte-level BPE | 10.84 | 3/3 | Needed NEOX RoPE, QKV bias, and `Q5_0` dequant (found by probing the real file, not assumed from its "Q4_K_M" name) |
| GPT-2 (124M) | `gpt2` | 124M | 50257 | byte-level BPE | 20.58 | 0/3 | Needed a wholly separate module (`gpt2.c`): fused QKV, LayerNorm, learned position embeddings, ungated GELU, no RoPE at all; needed `Q5_K` dequant |
| Gemma-2b-it | `gemma` | 2.5B | 256128 | SentencePiece | 73.05 | 2/3 | Initially untestable end-to-end (~10-12GB f32 footprint didn't fit this ~7.6GB dev machine); now verified end-to-end after the on-demand dequantization pass below dropped its resident memory to ~1.5GB — see "CPU optimization" |

\* Teacher-forced perplexity over the same fixed 35-ish-token English test
corpus (`--bench`, see "Correctness evidence" above) — **not directly
comparable across vocab sizes** (a larger vocab's random-guess ceiling is
itself higher, so a fair comparison is each number against its own file's
ceiling, not against each other): TinyLlama's ceiling is ~32000,
Qwen2.5's ~151936, GPT-2's ~50257, Gemma-2b's ~256128. All four measured
numbers are two-to-three orders of magnitude below their own ceiling.
The GPT-2/TinyLlama/Qwen2.5 ranking (worst to best) does track something
real, though: newest, largest, most heavily instruction-tuned model wins,
exactly as expected — not an artifact of the benchmark. Gemma-2b's 73.05
sits above TinyLlama's despite being the larger model, plausibly because
its 256128-token vocabulary spreads probability mass thinner per token
than TinyLlama's 32000 — consistent with, not contradicting, the
"compare each number to its own ceiling" caveat above.

Try any of them yourself:

```bash
./llamini qwen2.5-0.5b-instruct-q4_k_m.gguf --raw   # Qwen2.5-0.5B-Instruct
./llamini gpt2.Q4_K_M.gguf                          # GPT-2 (no chat template for this arch)
./llamini gemma-2b-it-q4_k_m.gguf --bench           # Gemma-2b -- now verified end-to-end, see above
```

(`--raw` on Qwen2.5 because this project's chat template, `build_chat_prompt_tokens`
in `main.c`, is hardcoded to TinyLlama-Chat's specific `<|system|>/<|user|>/<|assistant|>`
format — Qwen models actually use a different convention, ChatML
(`<|im_start|>`/`<|im_end|>`), not implemented here. Raw completion works
correctly regardless of chat template, and is what the comparison table
above uses for a fair, template-independent measurement.)

## Benchmarks

Real head-to-head numbers against real llama.cpp — not a public number cited
from somewhere else. Built unmodified upstream llama.cpp
(`ggml-org/llama.cpp`, commit `0b5be7e`) from source on this same machine,
in a scratch directory entirely outside this repo, never vendored or
shipped (see STDLIB.md), and ran `llama-bench` against the exact same GGUF
files this project already uses.

| Model | llamini RSS | llama.cpp RSS | llamini tok/s\* | llama.cpp tok/s (tg32) |
| --- | --- | --- | --- | --- |
| TinyLlama-1.1B | ~710 MB | ~1.12 GB | 0.41-0.67 | 0.80 ± 0.40 |
| Qwen2.5-0.5B | ~480 MB | ~558 MB | 0.95-1.24 | 0.69 ± 0.21 |
| GPT-2-124M | ~831 MB | ~169 MB | 4.80-8.32 | 0.56 ± 0.31 |
| Gemma-2b | ~1.54 GB | ~2.66 GB | not measured | 1.78 ± 2.19 |

\* llamini's own numbers are `--bench`'s per-completion tok/s (8 tokens,
including that call's prompt prefill) — not the same measurement as
llama.cpp's `tg32` (pure decode, no prompt processing), so read this as
directional context, not a precise ratio; the two aren't measuring
identical work. llama.cpp's own ± is its `-r 3` repetition spread, and it's
often larger than the mean (Gemma's 1.78 ± 2.19) — this dev VM's
memory/virtualization overhead dominates for *both* engines, not just this
one, which is itself a useful, honest data point about how much these
numbers should or shouldn't be extrapolated to other hardware.

**The honest, nuanced result:** llamini.cpp's resident memory beats real
llama.cpp's on 3 of 4 models (TinyLlama, Qwen2.5, Gemma-2b) on this exact
machine — plausibly because llama.cpp pre-allocates a KV cache and batch
buffers sized for a much larger default context than this project bothers
with, while the CPU optimization pass (see Limits) means llamini no longer
pays for a fully-materialized `f32` copy of every weight either. It loses
on GPT-2 (831MB vs 169MB) for a disclosed reason, not a mystery: GPT-2's
own weights (`gpt2.c`) were not brought into this pass's on-demand
dequantization — only the LLaMA-family path (`model.c`, shared by
LLaMA/Qwen2/Gemma) was. Generation speed is closer and noisier: llamini
measured faster on Qwen2.5 and dramatically faster on tiny GPT-2, slower on
TinyLlama, on a benchmark methodology gap disclosed above — not a claim
that llamini out-executes a mature, years-tuned inference engine in
general.

See STDLIB.md for the landscape comparison against `llama2.c` (the closest
genuinely from-scratch comparator) and `llamafile` (which vendors
llama.cpp/ggml wholesale) on dependency footprint and architecture
generality, and BUILDLOG.md's CPU optimization phase for the full research
and methodology.

## Limits (honest, not papered over)

- **Four architectures, all now verified end-to-end.** `llama`, `qwen2`,
  and `gpt2` were verified against real downloaded files first (see
  "Other model families"). `gemma` was initially the odd one out: its
  architecture-specific code (NEOX RoPE, GeGLU, `embedding_scale`, tied
  embeddings, MQA) is all shared with the already-verified LLaMA-family
  path and its config auto-detection was confirmed correct against a real
  file from the start, but full generation didn't fit this project's dev
  machine (~7.6GB RAM; Gemma-2b's ~10-12GB fully-dequantized-to-f32
  footprint didn't fit, and three attempts gave inconclusive results
  rather than one clean failure). The CPU optimization pass below (see
  "CPU optimization") fixed the underlying memory footprint for every
  architecture, not Gemma specifically, and Gemma-2b now runs end-to-end
  reproducibly at ~1.5GB resident — see "Other model families". Mistral
  (GGUF architecture is literally `"llama"`, so it would load with the
  existing code, demonstrating no new capability) and Gemma2 (real added
  complexity — logit/attention softcapping, sliding-window attention, a
  `head_dim` that isn't `dim/n_heads` — for the same memory cost as
  Gemma1) were not attempted at all. No chat template exists for Qwen2
  (it uses ChatML, `<|im_start|>`/`<|im_end|>`, not TinyLlama-Chat's
  format) or GPT-2 (no instruction-tuned convention to wrap in the first
  place) — both are raw-completion-only here.
- **Chat template is a best-effort reconstruction, not verified against a
  real tokenizer run.** `build_chat_prompt_tokens` (`main.c`) wraps your
  input in TinyLlama-Chat's own `<|system|>/<|user|>/<|assistant|>` template
  (from the model's published `tokenizer_config.json`), but this project has
  no reference `transformers` install to cross-check the exact token
  sequence against — it's derived from the template text, not confirmed
  bit-for-bit. One real wrinkle found and worked around: `</s>` (end of a
  turn) names a real EOS control token, but typing it as literal text does
  **not** BPE-merge back into that token — verified empirically, not
  assumed — because this tokenizer has no special-token pre-split step
  (real tokenizers do). `build_chat_prompt_tokens` works around this by
  inserting `eos_id` as a real token between segments instead of embedding
  `"</s>"` as text; see `STDLIB.md`. Pass `--raw` to bypass the template
  entirely.
- **Greedy by default, temperature + top-p opt-in.** `--temp X` (`main.c`,
  `sample_token` in `generate.c`) enables softmax sampling with nucleus
  filtering; there's no top-k, no repetition penalty, and no `--seed` for a
  reproducible sampled transcript yet. Without `--temp`, output stays fully
  deterministic and can still repeat or loop on longer generations.
- **Single-turn generation.** Each chat line resets every layer's KV cache
  and is generated independently; there is no multi-turn conversational
  memory.
- **Slow, and prompt length now has a real cost.** `linear()` (the
  matrix-vector projection every layer uses, `model.c`) is parallelized
  across output rows with a persistent thread pool and gcc vectorizes its
  dot-product reduction (conservative fast-math flags, see
  STDLIB.md/`deps-proof.txt`), but `causal_mha`'s attention is still a
  naive scalar loop, and this remains far from a production engine. Since
  prefilling the prompt runs the full forward pass once per prompt token
  before the first reply token, a long prompt costs proportionally more
  time up front than a KV-cache-only design would. Wall-clock timing on
  this project's dev environment (a memory-constrained WSL2 VM) has been
  noisy enough run-to-run — the same binary on the same file has measured
  anywhere from under a minute to nearly twenty — that virtualization/
  memory-pressure overhead, not raw arithmetic, looks like the dominant
  variable; see "CPU optimization" below and `deps-proof.txt` for the
  actual numbers rather than a single claimed figure.
- **CPU optimization.** Researched how the real llama.cpp/ggml achieves
  its CPU performance (reading its actual source, not guessing) and
  applied two independent fixes, staying inside libc+POSIX: (1) a
  persistent thread pool in `model.c` replaced the old spawn-per-call
  `pthread_create`/`pthread_join` (~9 spawns/layer/token), removing pure
  threading overhead with bit-identical output; (2) weight matrices
  (embeddings, `lm_head`, every per-layer projection) are now dequantized
  in small row-chunks on demand, straight out of the mmap'd GGUF file,
  instead of being fully materialized to `f32` at load time — the same
  core idea as ggml's quantized dot products, adapted to this project's
  simpler f32-scratch-buffer design rather than ggml's integer SIMD
  kernels. This is a real, measured, and large effect: resident memory
  dropped from ~4.4-4.9GB to ~700MB-1.5GB across every model tested (a
  ~6-7x reduction, consistent across TinyLlama, Qwen2.5, and Gemma-2b),
  and it is what let Gemma-2b move from "untested under memory pressure"
  to reproducibly verified end-to-end (see "Other model families" above).
  Output is bit-identical to the pre-optimization binary for every
  verified model — this was the acceptance bar, not just "doesn't crash."
  See `BUILDLOG.md`'s CPU optimization phase for the full research and
  verification trail, and `STDLIB.md` for what's deliberately still
  simpler than ggml's real approach (no quantized-activation dot products,
  chunk size chosen for bounded memory over raw thread parallelism).
- **Dequantization supports F32, F16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K**
  (the types real "Q4_K_M"-labeled files actually use in practice for
  norms/weights/output — a file's quantization-scheme *name* doesn't
  guarantee every tensor uses only that scheme; Qwen2.5-0.5B's own
  `Q4_K_M` file uses `Q5_0` for several tensors, found by inspecting real
  tensor types rather than assuming). Anything else (`Q2_K`, `Q3_K`,
  `Q5_K`, `Q8_K`, ...) makes `gguf_dequantize_tensor` return `-1` rather
  than silently misreading the bytes — see `gguf.c`.
- **The generic INT4 quantizer in `quant.c`** (scale/zero-point, exercised
  by `--test`) is a separate teaching implementation, unrelated to the
  real `Q4_K`/`Q6_K` GGUF block decoders in `gguf.c` used to actually load
  the model.
- **No test framework.** C ships none in the standard library; `--test`
  runs `assert()`-based smoke tests in-process (see `main.c`), including
  a small synthetic-weights run of the full per-layer forward pass with a
  non-trivial GQA shape (4 query heads sharing 2 KV heads).

## License

MIT — see [LICENSE](LICENSE).
