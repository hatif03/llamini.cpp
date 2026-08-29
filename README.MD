# llamini.cpp

A C99 mini `llama.cpp`: a real GGUF file parser (metadata + tensor-info
table), a hand-rolled `Q4_K`/`Q6_K` block dequantizer, a 22-layer
grouped-query-attention transformer forward pass, and a real SentencePiece-BPE
tokenizer decoded from the GGUF file's own vocab — wired into a chat CLI that
loads and runs an actual `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` file. Built
for the Zero Dependency hackathon, Track F — libc and POSIX only, no
third-party runtime dependency of any kind. See [STDLIB.md](STDLIB.md) for
every package this project would normally reach for, and what stood in for
it instead.

## Build and run

Requires a POSIX environment (Linux, macOS, or WSL/MSYS2 on Windows) — the
code uses `mmap`/`open`/`fstat`, which plain MinGW does not provide (see
[STDLIB.md](STDLIB.md)). Verified with `gcc` 15.2.0, C99, on Ubuntu (WSL).

```bash
make clean
make
./mini_llama tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Run the in-process unit tests instead of the chat loop:

```bash
./mini_llama --test
```

## Chat loop

```text
$ ./mini_llama tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Loaded GGUF v3 (201 tensors, 23 metadata entries)
Config: dim=2048 layers=22 heads=32 kv_heads=4 ffn=5632 vocab=32000 seq_len=2048

======== TinyLlama GGUF Chat Engine Ready ========
Real GGUF weights + SentencePiece BPE tokenizer (see README limits).

You: hello
Bot:. The event, DP, and

You: exit
Bot: Bye!
```

Every architecture number in that `Config:` line (dim, layer count, head
count, GQA KV-head count, FFN size, vocab size) is read out of the file's
own `llama.*` metadata, not hardcoded — see `llama_config_from_gguf` in
`model.c`. Each line you type is BPE-tokenized against the file's real
vocab, embedded, and run through all 22 real decoder layers (RMSNorm ->
grouped-query causal attention with a real softmax and a per-layer KV
cache -> residual -> RMSNorm -> SwiGLU FFN -> residual), then the final
norm and output projection, greedily decoded back into real vocabulary
pieces. See [llamini-architecture](.claude/skills/llamini-architecture/SKILL.md)
for the module map.

Replies are **not** coherent chat responses yet — see Limits below for why.

## Limits (honest, not papered over)

- **No chat template.** TinyLlama-Chat expects a specific prompt format
  (`<|system|>...<|user|>...<|assistant|>`); this project feeds your raw
  line straight to the tokenizer instead, so the model is doing raw text
  completion, not answering a chat turn. That's why "hello" continues as
  if it were the start of some other sentence, rather than replying to
  you. Wiring up the real chat template is the natural next step here.
- **Greedy decoding only.** No temperature/top-p/top-k sampling, so output
  is deterministic and can repeat or loop on longer generations.
- **Single-turn generation.** Each chat line resets every layer's KV cache
  and is generated independently; there is no multi-turn conversational
  memory.
- **Slow.** `linear()` (the matrix-vector projection every layer uses,
  `model.c`) and `causal_mha`'s attention are naive scalar loops — no
  SIMD, no BLAS, no multithreading. Loading the ~638MB Q4_K_M file and
  dequantizing all ~1.1B parameters to f32 (~4.4GB resident) took anywhere
  from about 1 to 5 minutes in this project's dev environment (a
  memory-constrained WSL2 VM); each generated token costs roughly a
  billion multiply-adds across 22 layers, so expect real per-token
  latency on modest hardware. A production engine would keep weights
  quantized and dequantize per-block on the fly, and vectorize the inner
  loops; this one keeps the code readable instead.
- **Dequantization supports F32, F16, Q4_0, Q4_1, Q8_0, Q4_K, Q6_K** (the
  types an actual Q4_K_M file uses for norms/weights/output). Anything
  else (`Q2_K`, `Q3_K`, `Q5_K`, `Q8_K`, ...) makes `gguf_dequantize_tensor`
  return `-1` rather than silently misreading the bytes — see `gguf.c`.
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
