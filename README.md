# llamini.cpp

**llamini.cpp** is a from-scratch C99 reimplementation of llama.cpp's
core: a real GGUF file parser (metadata + tensor-info table), a
hand-rolled `Q4_K`/`Q6_K` block dequantizer, a 22-layer
grouped-query-attention transformer forward pass, and a real SentencePiece-BPE
tokenizer decoded from the GGUF file's own vocab — wired into a chat CLI that
loads and runs an actual `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` file. Built
for the Zero Dependency hackathon, Track F — libc and POSIX only, no
third-party runtime dependency of any kind. See [STDLIB.md](STDLIB.md) for
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

## Limits (honest, not papered over)

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
  across output rows with pthreads and gcc now vectorizes its dot-product
  reduction (conservative fast-math flags, see STDLIB.md/`deps-proof.txt`),
  but `causal_mha`'s attention is still a naive scalar loop, and this
  remains far from a production engine. Loading the ~638MB Q4_K_M file and
  dequantizing all ~1.1B parameters to f32 (~4.4GB resident) took anywhere
  from about 25 seconds to several minutes across runs in this project's
  dev environment (a memory-constrained WSL2 VM) — real measurements
  varied enough run-to-run that memory/virtualization overhead, not raw
  arithmetic, looks like the dominant cost there (see `deps-proof.txt`).
  Generation throughput measured 0.15-0.21 tok/s before threading and
  1.0-2.1 tok/s after, on that same VM's 12 logical cores. Since prefilling
  the prompt runs the full forward pass once per prompt token before the
  first reply token, a long prompt now costs proportionally more time up
  front than the old (incorrect) O(1)-in-prompt-length behavior did. The
  architecturally "correct" fix for the memory footprint — keep weights
  quantized and dequantize per-block on the fly inside `linear()`, instead
  of fully dequantizing to `f32` at load time — was deliberately not
  attempted: it's rated high risk (could silently produce plausible-but-
  wrong output rather than an obvious failure) for a benefit not yet
  confirmed to be worth it here. See the `llamini-architecture` skill's
  "Still open" section for the concrete plan if this is picked up later.
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
