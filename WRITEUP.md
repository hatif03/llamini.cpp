# Building llamini.cpp: a from-scratch llama.cpp core in C99, zero dependencies

*Written for the Zero Dependency hackathon's Write-Up side quest. Full,
unedited engineering log in [BUILDLOG.md](BUILDLOG.md) — this post is the
narrative version of that log, not a replacement for it.*

## The pitch

llamini.cpp loads and runs a real `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf`
file — a 668MB, 1.1-billion-parameter, 4-bit-quantized language model — using
nothing but libc and POSIX. No `ggml`, no `llama.cpp` linked as a library, no
Python bindings, no BLAS, no `sentencepiece`. The GGUF binary parser, the
`Q4_K`/`Q6_K` block dequantizers, the 22-layer grouped-query-attention
transformer forward pass, and the SentencePiece-style BPE tokenizer are all
written from scratch, this weekend, in C.

I built this with Claude Code as a pair programmer. This is the honest
account of how it actually went — including the parts that didn't work the
first time.

## What I reimplemented

Everything a real `llama.cpp`-class inference stack needs, minus the parts
that are genuinely out of scope for one weekend:

- **A real GGUF parser** — not just the 24-byte header, but the actual
  metadata key/value table (all 13 value types, including nested arrays) and
  the tensor-info table, with alignment-padded tensor data after that. Most
  toy GGUF readers stop at "read the header and print the version"; the
  metadata table is where the model's real shape (hidden size, layer count,
  head count, vocab) actually lives, and it's non-trivial to parse correctly
  because entries are variable-length and self-describing.
- **`Q4_K` and `Q6_K` block dequantization** — the two "K-quant" super-block
  formats a real `Q4_K_M` file actually uses (plus the simpler `F32`, `F16`,
  `Q4_0`, `Q4_1`, `Q8_0` for completeness). These are bit-packed formats:
  scales and minimums for eight 32-element sub-blocks packed into 12 bytes
  using overlapping 6-bit fields. Getting the bit shifts right with zero
  reference implementation to diff against was the part of this project
  closest to "hope this compiles into correct math."
- **A 22-layer transformer forward pass** with grouped-query attention (32
  query heads sharing 4 key/value heads — TinyLlama doesn't use plain
  multi-head attention), RoPE positional encoding, SwiGLU feed-forward
  layers, and RMSNorm — the actual architecture, not a simplified stand-in.
- **A SentencePiece-style BPE tokenizer**, driven entirely by the vocab,
  merge-scores, and token-type arrays embedded in the GGUF file itself — no
  external vocab file, no `sentencepiece` library.

## What the standard library made painful

C's standard library has no hashmap, no JSON, no HTTP, and (this is the one
that actually mattered) no obvious way to verify you got any of this right.
Every other language in this hackathon's toolbox has some path to "install
the real library once, diff your output against it." C's answer is "write
it carefully and reason about it," which is a fundamentally different — and
slower — way to build confidence.

The place this bit hardest: BPE tokenization needs a fast string→id lookup
over a real ~32,000-entry vocabulary, and a linear scan is genuinely too
slow for interactive use once you're not dealing with a 15-word toy vocab
anymore. That's a legitimate reason to hand-roll an open-addressing hash
table (FNV-1a, linear probing) — not because "C has no hashmap" is an excuse
to build one everywhere, but because this specific spot actually needed it.

## The package I made look unnecessary

`ggml`/`llama.cpp` linked as a library, `llama-cpp-python`, `sentencepiece`,
and a BLAS/numpy-equivalent for the tensor math. All four are things you'd
normally `pip install` or link against to do exactly what this project does
from a single C binary with an empty dependency manifest. I'm not claiming
to have reimplemented `ggml` in general — this loads one architecture family
(LLaMA-shaped: RMSNorm + RoPE + SwiGLU + GQA) and a specific quantization
mix. But for that one job, the library isn't required.

## The edge case that ate an afternoon (well — several)

**The one that would have eaten an afternoon, if I'd let it:** after
rebuilding everything for real, the model would type back something like
`". The event, DP, and"` in response to "hello" — real English fragments,
not noise, but clearly disconnected from the input. The obvious suspects
are tokenizer bugs, wrong weights, or broken attention math, and chasing any
of those from the symptom alone could easily have burned hours. The actual
bug, found by reading the generation loop line by line instead of
guessing: it only ever embedded the *last* token of whatever you typed. Every
earlier position's slot in the KV cache stayed zeroed, and the attention
math still summed over those zeros regardless. The model was never blind —
it just never got shown the input.

**The one where I tested an assumption before building on it, and the
assumption was wrong:** I planned to build a chat template by literally
inserting the text `"</s>"` between turns, reasoning that since `</s>` is a
real end-of-turn token with a merge-priority score of `0.0` while every
ordinary vocabulary piece has a negative score, the tokenizer's own
score-based merging would naturally reassemble it into that single token.
Before writing the feature, I wrote a five-minute throwaway probe that just
tokenized the literal string and printed the result. It split into three
ordinary sub-word pieces and the real end-of-turn token never appeared once.
Real tokenizers special-case strings like this *before* running BPE; mine
doesn't, so it never had a chance. The fix ended up better than the original
plan — build the prompt at the token level and insert the real token id
directly, which sidesteps the whole problem instead of fighting it.

**The one I shipped wrong on the first try, on purpose left in this
write-up:** fixing the "only sees the last token" bug above, I rewrote the
generation loop's stopping condition as a single precomputed iteration
count — prefill steps plus generation steps. It was off by one: the last
prefill step and the first generated token turn out to be the *same* loop
iteration, so my count generated one extra token every time. The project's
own test suite caught it immediately (`Assertion 'total >= 4 && total <= 8'
failed`) before it ever reached a commit. I didn't get this right on the
first attempt, the tests did their job, and the actual fix (three explicit,
separately-readable stopping conditions instead of one clever formula) is
better code than what I would have shipped if I'd been more confident and
skipped writing a test for it.

**The one that wasn't a code bug at all:** for a while, the same command
took anywhere from 25 seconds to over five minutes, with no pattern I could
find in the code. Turned out to be a previous test run that a `timeout`
wrapper hadn't actually killed — 19 minutes deep, still holding 4.4GB of RAM
in a VM that only had 7.6GB total. I'd been benchmarking my own leftover
mess. Lesson that's now permanent habit: check `ps aux` before trusting a
timer, especially inside a VM.

## How I actually verified this was correct, not just "not crashing"

"The output looks like English" is a weak claim — a broken forward pass
producing token-shaped garbage can still coincidentally resemble language
in short bursts. I looked for a published TinyLlama perplexity number to
compare against, found one via a web search, and when I went to actually
verify it against the paper the search cited, **the number wasn't in that
paper.** The search had synthesized it. That's a real, embarrassing-if-I'd-
missed-it lesson: an AI-summarized search result is not a citation until
you've opened the source and checked.

So instead of leaning on an external number I couldn't verify, I built a
self-contained benchmark: teacher-forced perplexity over a small fixed
English test corpus, compared against the one number that needs no citation
at all — a uniform random guess over a 32,000-token vocabulary has a
perplexity of exactly the vocabulary size (~32000, or 10.37 nats/token).
Measured: **13.28**. Three orders of magnitude below random. Alongside that,
a handful of specific factual completions — `"The capital of France is"` →
`" Paris."` — which is a sharper, more falsifiable signal than "sounds like
English," because there's exactly one right answer and the model gave it.

## What made things faster, and what didn't move the needle as much as expected

Compiling with `-march=native` plus a few reassociation flags (short of full
`-ffast-math`, which also assumes away NaN/Inf checks I wanted to keep) let
gcc vectorize the core matrix-vector dot product — confirmed by reading the
generated assembly, not assumed. Parallelizing that same operation across
CPU cores with a dozen lines of pthreads on top of that got another 5-7x.
Both were verified to produce byte-identical output to the unoptimized
version first, because a "faster" change that quietly changes the answer
isn't actually a win.

What I *didn't* do: keep weights quantized in memory and dequantize them
on the fly during matrix multiplication instead of expanding everything to
full 32-bit floats up front (currently ~4.4GB resident for a 1.1B-parameter
model). That's the architecturally correct fix for memory footprint, and I
scoped it out in detail before deciding not to attempt it — it touches the
most correctness-critical code in the project, and a subtle bug there
produces *plausible-looking wrong text*, not a crash you'd notice. Given the
speedup from the safer changes above, I chose not to gamble a working
project on a rewrite whose benefit I hadn't confirmed was still necessary.
Documented as the clear next step instead of quietly dropped.

## What's still genuinely unfinished

No multi-turn conversation memory (every line is a fresh single-turn
generation). No sampling beyond temperature + top-p (no top-k, no repetition
penalty). The chat template is my best reconstruction of TinyLlama-Chat's
own format, not verified against a real `transformers` install, because
having one would defeat the point of the project. Quantization support
covers the formats an actual `Q4_K_M` file uses and nothing more —
anything else is explicitly rejected rather than silently misread. All of
this is written down in the repo's own limits section, not discovered by a
judge.

---

*Full engineering log, commit-by-commit, including every dead end and every
number measured: [BUILDLOG.md](BUILDLOG.md). Source: [repo link].*
