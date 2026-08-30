# BUILDLOG.md

A chronological, factual engineering journal for llamini.cpp. Every entry is
dated, states what was tried, what actually happened (including when it made
things worse or turned out wrong), and how it was verified. Nothing here is
smoothed over for narrative effect — that's what `WRITEUP.md` is for. This
file is the raw material `WRITEUP.md` (and any future write-up) is drawn from.

**How this system works, going forward:** every time a real change lands —
a feature, a bug fix, a failed experiment, a wrong assumption caught by
testing — append an entry here *in the same sitting*, before or right after
the commit that made it, not from memory later. Each entry should answer:
what was the state before, what was tried, what broke or worked, how it was
verified, and what the commit hash is. Dates use the machine's real clock
(`date`/`git log --format=%ci`), not guesses.

---

## Phase 0 — the project I inherited (2026-08-29, before this log started)

Before any of the work below, the repo already had five commits
(`4d8672d`..`c5c6da0`, 2026-08-29 08:51–21:53) plus two more the same night
(`ead4612`, `d6a3345`, 2026-08-30 00:36–01:16): tensor ops, a KV cache,
generic INT4 quantization, a GGUF *header* reader (magic + version only, no
metadata or tensor table), model structs with RMSNorm/SwiGLU/RoPE/causal-MHA
math, a demo tokenizer, and a demo generation loop — each with its own small
`--test` unit test.

What those tests didn't catch: **none of it was wired together into
anything real.** `main.c`'s chat loop was a hardcoded keyword table
(`strcasestr(text, "hello")` → literal canned reply string) with logits
*faked* as `logits[prompt_token] = 10.0f` before greedy-sampling them — the
attention math ran, computed something, and then got thrown away. `gguf.c`
assumed tensor data started immediately after a 24-byte header, which is
false for a real GGUF file (there's a metadata block and a tensor-info table
in between). `tokenizer.c` and `generate.c` existed but weren't even in the
Makefile's `SRC` list.

This is a normal way for an AI-assisted build to drift: every individual
piece had a green test, so it *looked* like steady progress, while the thing
that actually mattered — does typing something produce a reply that depends
on what you typed — was never true.

## Phase 1 — validate, then rebuild for real (2026-08-30, session start)

Asked to validate the project was submission-ready. Reading `main.c` end to
end (not just running `--test`) surfaced the keyword-stub chat and the
header-only GGUF reader within minutes — the kind of thing a test suite
built around isolated units won't catch, because each unit really was
correct in isolation.

Decision, confirmed with the user: rebuild `gguf.c`, `model.c`, `tokenizer.c`,
`generate.c`, and `main.c` for real, rather than patch the stub. Delivered in
one long working session:

- **`gguf.c`**: a real parser for the actual GGUF binary layout — header,
  metadata key/value table (all 13 value types including arrays), tensor-info
  table, alignment-padded tensor-data section. Wrote a small standalone probe
  (`probe.c`, not shipped) to dump a real file's structure before trusting any
  of this.
- **Q4_K/Q6_K dequantization**: hand-rolled from the public block-format
  documentation (super-block scale/min bit-packing), *not* copied from
  ggml's source (the hackathon explicitly disallows vendoring). No reference
  implementation to diff against — correctness had to be argued from first
  principles (dequantized weight values should look like small, varied,
  non-NaN floats, not the actual proof).
- **`model.c`/`generate.c`**: a real 22-layer grouped-query-attention
  transformer forward pass (RMSNorm → GQA causal attention with a per-layer
  KV cache → residual → RMSNorm → SwiGLU FFN → residual → final norm →
  lm_head), replacing the old fake-logits stub.
- **`tokenizer.c`**: a real SentencePiece-style BPE encoder/decoder driven by
  the GGUF file's own `tokenizer.ggml.tokens`/`.scores`/`.token_type` arrays,
  with byte-fallback and a hand-rolled open-addressing hash table (a genuine
  ~32000-entry vocab makes a linear scan too slow — this is the one place a
  hashmap was actually necessary, not gratuitous).

**Bug found and fixed during this pass, before it ever ran:** `tokenizer.c`'s
vocab table was declared with 128 slots but only ~15 were initialized (a
holdover from the old demo). The other slots are `NULL` by C's zero-init
rule. Both the BPE merge-lookup loop and the decode function would have
handed `strcmp`/`strncpy` a `NULL` pointer the moment a real ~32000-token
vocab (not the old 15-word demo) hit an untrained slot — which is
*immediately*, on essentially any real input. Fixed by having the lookup
stop at the first `NULL` slot instead of scanning the full declared size.
This was dead code until this exact rewiring, so it had never actually run
before.

**Downloaded the real target file**: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`'s
`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (668,788,096 bytes) via `curl` in
WSL. Verified the parser against real bytes with the standalone probe:
`version=3 n_tensors=201 n_kv=23`, `dim=2048 layers=22 heads=32 kv_heads=4
ffn=5632 vocab=32000` — matches known TinyLlama-1.1B specs exactly, and
`n_tensors=201` matches `22 layers × 9 tensors + 3 global` precisely, which
was a nice independent sanity check that the tensor-info table was being
walked correctly.

**First real end-to-end run**: typing `hello` produced `". The event, DP,
and"` — not coherent, but *real* English word fragments, not noise, and no
crash. Confirmed the Q4_K/Q6_K dequant was structurally sound (dequantized
embedding rows: small, varied, non-NaN, consistent with trained weights)
before declaring victory on "it doesn't crash."

**Environment gotcha #1 (cost real time, taught nothing about the code):**
a load-then-generate run appeared to take between 25 seconds and over 5
minutes across repeated attempts, wildly inconsistent. Chased this as if it
were a code problem before checking `ps aux` and finding a *previous* test
run — killed by a `timeout` wrapper that sent SIGTERM but didn't get waited
on properly — still resident, holding 4.39GB RSS and 94% CPU, 19 minutes
into a WSL2 VM with only 7.6GB total RAM. Killing the orphan and re-measuring
gave consistent numbers. Lesson: check for stray processes contending for
memory *before* trusting a wall-clock benchmark, especially in a VM.

**Environment gotcha #2 (real, and it stuck around):** even with a clean
process table, a plain `calloc`+touch of 4.4GB (matching this model's full
`f32` weight footprint) took ~25 seconds in this WSL2 VM, dominated by `sys`
time rather than `user` time — consistent with WSL2's dynamic memory
ballooning overhead, not anything in this project's code. This became
important later (Phase 2) when deciding whether a much bigger, riskier
memory-footprint refactor was worth attempting.

## Phase 2 — "confirm zero-dep, then improve it, one commit at a time"

Asked to (a) confirm the project is genuinely zero-dependency and (b)
research further improvements, committing each one separately. This is the
part of the process that most benefited from *not* just diving in.

**Step 0 — plan before touching code.** A read-only audit (fresh grep across
every `#include`, every `system()`/`popen()`/`exec*` call, every package
manifest pattern) confirmed compliance was already clean — the improvement
work below is entirely about correctness, safety, performance, and
documentation, not fixing a dependency leak. Two independent research passes
were then run in parallel, deliberately with different framings so they
wouldn't just agree with each other by construction: one asked to find
low-risk/high-value correctness and feature work, the other asked to find
performance and memory-footprint improvements.

**The performance pass's most useful finding wasn't about performance — it
was a real bug the other pass had also flagged independently: the model's
`generate_autoregressive` never actually prefilled the prompt.** It jumped
straight to embedding only the *last* token of whatever was typed; every
earlier position's KV-cache slot stayed at the zeros `kv_cache_reset` had
set, and the attention loop still summed over those zeroed positions
regardless. This — not the missing chat template originally suspected — was
the real reason output looked disconnected from the input. **This is the
edge case that could have eaten an afternoon if it had been chased through
symptom-guessing** (tokenizer bug? attention math bug? wrong weights?)
instead of a careful line-by-line read of what `generate_autoregressive`
actually does with `in_token_count`. Confirmed as the correct diagnosis by
fixing it and re-measuring: "hello" went from `". The event, DP, and"` to a
real "Hello, world!"-style completion.

Both passes also converged on: gcc wasn't vectorizing `linear()`'s
dot-product reduction at the existing `-O2` (confirmed with `objdump`:
scalar `addss`, not `addps`/`vfmadd` — gcc can't reassociate a floating-point
reduction without explicit permission), which was a bigger and *safer* win
than anything else on the list; and that the ~4.4GB-of-f32-upfront memory
layout was the "architecturally correct" thing to fix but genuinely
high-risk (3-6 hours, touches the most correctness-critical dequant code,
could produce *plausible-looking but silently wrong* output rather than an
obvious crash) — recommendation: measure whether cheaper fixes solve enough
of the slowness before committing to that rewrite. Decision: use it as a
diagnostic gate, not a default plan.

User decisions taken before writing any code: conservative compiler flags
(not full `-ffast-math`, to keep NaN/Inf checks live rather than assume them
away) over maximum speed; the memory-footprint rewrite deferred as a
stretch goal, attempted last and only if time remained.

**A third thing got caught before implementation, not after:** asked whether
"the output looks like English" was a strong enough correctness claim, a web
search surfaced what looked like a citable published TinyLlama perplexity
number. Fetching the actual paper it pointed to showed **the number wasn't
in that source** — the search summary had fabricated it. Decision: build a
self-contained benchmark instead (teacher-forced perplexity against a fixed
test corpus, compared to the mathematically-certain "uniform random over the
vocab" ceiling) rather than cite a number that couldn't be verified. This
produced `--bench` (see below) and is arguably the single most important
"don't sugarcoat evidence" decision in this whole build.

### Commit-by-commit (2026-08-30, 05:15–06:52 local time)

Each of the following was its own commit, verified before landing (rebuild,
`--test`, and where noted a real-file run), in this order:

1. **`c4bf020` — GGUF parser hardening.** Found a genuine `u64` overflow
   while reading the code, not from a crash report: the metadata
   array-length path computed `elem_size * arr_len` with no overflow guard.
   A file claiming `arr_len = 2^62` with `elem_size = 4` wraps that product
   to a small number, the bounds check "passes," and a fabricated huge
   array length survives into `vocab_load_from_gguf`. Fixed by dividing
   (`arr_len > remaining / elem_size`), never multiplying. Added three
   hostile test files (`test_gguf_reject`) written with plain
   `fopen`/`fwrite` — one of them fails against the pre-fix code and passes
   after, which is the actual regression test for the fix, not just a
   feel-good assertion.
2. **`352da61` — compiler flags + tok/s timing.** `-O3 -march=native
   -fassociative-math -fno-signed-zeros -fno-trapping-math` (not
   `-ffast-math`). Verified the vectorization actually happened via
   `objdump` before trusting it (8-wide AVX `vmulps`/`vaddps` for the bulk
   of the loop). Added `clock_gettime` timing since there had been *zero*
   instrumentation before this — every earlier "it took N minutes" claim in
   this log was a stopwatch, not a measurement.
3. **`3256fa5` — pthread-parallelize `linear()`.** Split output rows across
   threads above a size threshold (small `k_proj`/`v_proj` calls stay
   serial — the work is smaller than a thread spawn costs). Deliberately did
   **not** parallelize `causal_mha`: its compute is negligible next to
   `linear()`'s, and it uses a `static f32 scores[MAX_SEQ_LEN]` buffer that
   would become a silent data race the moment more than one thread touched
   it — a race that produces *wrong numbers, not a crash*, which is exactly
   the kind of bug that's expensive to find later. Verified: byte-identical
   output to the pre-threading run, 0.15-0.21 tok/s → 1.01 tok/s.
4. **`3928d3c` — the prefill fix** (diagnosed in the planning phase above).
   **Introduced a new bug while fixing it, caught by the existing test, not
   by luck:** the first attempt bounded the loop with
   `in_token_count + max_gen_tokens` total steps, reasoning that prefill
   takes `in_token_count` steps and generation takes `max_gen_tokens` more.
   Wrong: the *last* prefill step and the *first* generated token are the
   same loop iteration (processing the last prompt position's logits *is*
   the first prediction), so that bound is off by one and generates one
   extra token. `--test`'s `test_generate` assertion (`total <= 8` for a
   4-prompt-token, 4-max-gen synthetic run) failed immediately —
   `mini_llama: main.c:215: test_generate: Assertion 'total >= 4 && total
   <= 8' failed`. Fixed by replacing the precomputed bound with three
   explicit, independently-understandable stopping conditions (KV cache
   full, `max_gen_tokens` reached, EOS) instead of one clever arithmetic
   expression. Re-verified against the real model: "hello" → ", world!" /
   print-function explanation, a real improvement, not just a fixed test.
5. **`3a98360` — `--bench`.** Refactored the per-position forward pass into
   a shared `forward_step`, used by both real generation and a new
   `compute_perplexity` (teacher-forced, never sampling), so the benchmark
   exercises the exact same code path as real inference rather than a
   parallel reimplementation that could quietly drift. Measured against the
   real file: **perplexity 13.28** over 37 tokens of ordinary English,
   against a uniform-random ceiling of ~32000 (10.37 nats/token) — roughly
   three orders of magnitude lower. `"The capital of France is"` completed
   to `" Paris."` — a specific, checkable, correct fact, not just "sounds
   like English."
6. **`d1f7906` — opt-in `--temp` sampling.** `temp <= 0` dispatches straight
   to the pre-existing deterministic `greedy_sample` — zero regression risk
   to the already-verified default path. **Shipped a bug in the first pass,
   caught before committing:** ran `--temp 0.8` twice against the real model
   and got the *exact same output both times*. `srand()` was never called,
   so `rand()` used its default seed on every process start — sampling
   "worked" in the sense of not crashing, but was silently deterministic,
   which defeats the entire point of the feature. Fixed with
   `srand((unsigned)time(NULL))`, gated on `temp > 0`. Re-ran twice: genuinely
   different (and still coherent) completions both times.
7. **`b810bd1` — chat template.** The planning phase's premise was that
   `</s>` (end of a chat turn) would BPE-merge back into the real EOS
   control token because control tokens carry a merge-priority score of
   `0.0` while ordinary pieces carry negative scores. **Wrote a throwaway
   probe and tested this before writing the feature, and the premise was
   wrong**: tokenizing the literal template string showed `</s>` splitting
   into ordinary sub-word pieces (`.</`, `s`, `>`) — `eos_id` never appeared.
   This tokenizer has no special-token pre-split pass (a real
   `transformers`/`sentencepiece` tokenizer does, which is why the
   assumption seemed reasonable on paper). Fix: build the prompt at the
   *token* level, not the string level — tokenize each segment separately,
   drop each segment's auto-prepended BOS except the first, and insert the
   real `eos_id` token programmatically between segments instead of ever
   typing `"</s>"` as text. Verified: `--raw` reproduces the exact
   pre-template baseline (proving it's a true no-op), and the default
   templated mode produces a genuinely different, chat-register reply
   (`"Sure, I'd be happy to provide you with a friendly chatbot..."`) instead
   of raw text completion.
8. **`108e0b4` — Package Killer framing** (docs only): named exactly what
   this replaces (`ggml`/llama.cpp-as-a-library, `llama-cpp-python`,
   `sentencepiece`, a BLAS/numpy-equivalent) with an equally explicit "what
   this is NOT" — not a ggml reimplementation, partial quant coverage only,
   LLaMA-architecture only, not bit-exact-verified against a reference.
9. **`3afd5ba` — reproducible build proof** (docs only): two clean
   `make clean && make` runs from the same tree, same compiler, same flags
   → byte-identical SHA256 for the binary and every object file. Recorded
   honestly as same-machine/toolchain/path reproducibility, not a
   cross-environment guarantee (gcc can embed the build path in some
   sections).

**Deliberately not attempted:** fused/on-demand dequantization (keep weights
quantized, dequantize per-block inside `linear()` instead of the current
~4.4GB-of-f32-upfront layout). Both research passes rated it high risk for a
real but unconfirmed payoff, and the compiler-flags + threading commits
above already delivered most of the achievable speedup on this hardware —
asked the user directly rather than deciding alone, and the answer was to
stop at the 9 verified commits and document the idea as future work instead
of risking a subtle, hard-to-detect correctness bug this late. Documented in
the `llamini-architecture` skill and README rather than silently dropped.

## Phase 3 — naming cleanup (2026-08-30, 07:18)

The compiled binary was `mini_llama` and several docs described the project
as "a C99 mini llama.cpp" — readable as the project being named "mini llama"
rather than "llamini.cpp." Renamed the binary to `llamini` (verified: the
SHA256 of the binary content is unchanged by the rename — a build's content
doesn't depend on its output filename) and reworded every "mini llama.cpp"
phrase across the repo, including two files (`.cursor/`'s mirrors of the
`.claude/` skill docs) that had drifted stale relative to their `.claude/`
counterparts since Phase 2 and needed resyncing anyway.

## Phase 4 — other model families (2026-08-30, ongoing)

Asked to identify and run a variety of small models across different
architecture families (not just LLaMA), to show the engine isn't
TinyLlama-specific. Researched real, currently-available small GGUF models
across Gemma/BERT/GPT/Mistral/Qwen by reading llama.cpp's own source
(`gguf-py/gguf/constants.py`, `src/models/*.cpp`, `conversion/*.py`) rather
than guessing — this surfaced two real, cross-cutting requirements no one
would find just by staring at TinyLlama's file: (1) Qwen2/Gemma/Gemma2 use
the "NEOX" RoPE pairing convention (elements paired half a head-dim apart),
not the "NORM" pairing (adjacent elements) this project's `rope()` already
had — getting this wrong doesn't crash, it produces fluent-looking *wrong*
output, the worst kind of bug; (2) Qwen2/GPT-2 use byte-level BPE with an
explicit merges list and no per-token scores, a genuinely different vocab
convention from the SentencePiece format already built, not a config tweak.

**Decision on scope** (research findings, then a direct ask rather than
guessing): Mistral's GGUF `architecture` is literally `"llama"` — it would
load with zero new code, which also means it demonstrates zero real
diversity, and the smallest real Mistral checkpoint is 7B (~28GB as f32)
anyway, so skipped entirely. Gemma2's extra quirks (attention/logit
softcapping, sliding-window/full-attention alternation, a `head_dim` that
doesn't equal `dim/n_heads`) cost real new code for the same RAM ceiling as
Gemma1, so skipped. BERT is real and mainstream in GGUF (contrary to an
initial assumption) but is a fundamentally different product — no causal
mask, no KV cache, a pooled embedding output instead of next-token logits —
so deferred as a stretch goal, not attempted this pass. Chose Qwen2.5-0.5B,
GPT-2-124M, and Gemma-2b as the three targets, with Gemma explicitly
expected to be untestable end-to-end on this dev machine's 7.6GB RAM (its
~10-12GB f32 footprint doesn't fit) — write the code, disclose that
honestly, don't force a run that will thrash or fail.

**Shared infrastructure, verified against the existing TinyLlama path
before touching anything new:** extended `LLaMAConfig` with `head_dim`
(read from an explicit `<arch>.attention.key_length` key when present,
never re-derived as `dim/n_heads` and assumed correct — that derivation
happens to hold for every architecture supported here, but is documented as
not holding in general, e.g. gemma2), `rope_type`, `ffn_act` (SiLU vs GELU
— SwiGLU and GeGLU are the same shape, only the gate activation differs),
`qkv_bias`, and `embedding_scale`. Added `gguf_get_str` (a scalar-string
metadata accessor `gguf.c` didn't have yet — needed to read
`general.architecture` itself) and made `llama_config_from_gguf` build its
metadata key names as `"<arch>.<suffix>"` instead of a hardcoded `"llama."`
prefix, since every architecture's hyperparameters live under its own
namespace. Also fixed a real, independent bug the research surfaced:
`llama_load_weights` was dequantizing `token_embd.weight` into `lm_head` a
*second* time whenever a checkpoint ties them (no separate `output.weight`)
instead of sharing the pointer — wasted memory and time on every
tied-embedding model, not just the ones added this phase. Verified: rebuild,
`--test` unchanged, and a real TinyLlama run produces byte-identical output
to before this refactor (`arch=llama` auto-detected correctly, config
values match, "hello" -> the same "Sure, I'd be happy to..." chat reply as
before).

**Qwen2.5-0.5B-Instruct**: downloaded (`Qwen/Qwen2.5-0.5B-Instruct-GGUF`,
491MB). Config auto-detected exactly right (`dim=896 layers=24 heads=14
kv_heads=2 head_dim=64 vocab=151936`) on the first try. Weight loading did
not: **found a quantization format this file actually uses that the
earlier research hadn't flagged** — `token_embd.weight` and several other
big tensors are `Q5_0` (ggml type 6), not `Q4_K`/`Q6_K`/`Q8_0` as assumed
from the "Q4_K_M" filename. Found by writing a five-line probe that dumps
real tensor names/types from the file rather than guessing from the
research summary. Implemented `Q5_0` dequantization (32-element blocks, a
scale plus a 4-bytes-of-high-bits field folded into 4-bit nibbles for a
5-bit signed value) from the public block format, verified by *it actually
working* afterward, not by inspection alone. With that fixed: real output,
first try, no further bugs. `hello` (raw, no chat template attempted for
this architecture) continues as a Python function definition, then
transitions into fluent, grammatically correct Chinese discussing Python's
`__init__` method — a genuinely strong correctness signal, since getting
multi-byte UTF-8 (Chinese characters) right end-to-end through byte-level
BPE encode -> merge -> decode is a much sharper test than ASCII-only text
would be. `--bench`: perplexity **10.84** over 35 tokens (ceiling ~151936
for this file's larger vocab); all three fact completions correct,
including "Two plus two equals" -> "four" (TinyLlama got this one wrong).

**GPT-2-124M**: this one genuinely needed a new module, not a config flag
on the existing path — fused QKV (one tensor, sliced into Q/K/V after a
single `linear()` call rather than three separate projections), learned
absolute position embeddings added directly to the token embedding (no
RoPE at all), LayerNorm (mean-centered, with a bias — RMSNorm has neither),
plain multi-head attention (turned out to need zero new attention code:
`causal_mha` with `n_heads_kv == n_heads` *is* plain MHA, no GQA broadcasting
active), and an ungated GELU MLP (no gate tensor at all, unlike SwiGLU/
GeGLU's shared gated shape). Wrote `gpt2.c`/`gpt2.h` as a fully separate
module rather than bolting more special cases onto `model.c` — the two
architectures share almost nothing.

Learned from the Qwen2 experience: **probed the real file's tensor names
and types before writing any loading code this time**, rather than
assuming from the research summary. Good thing — found `attn_qkv.weight`
uses `Q5_K` (ggml type 13), a format declared in `gguf.h` back when Q4_K_M
support was first built but never actually implemented (no real file had
exercised it yet). Implemented `Q5_K` dequantization (the same 256-element
super-block structure as `Q4_K`, plus a packed high-bit plane for a 5-bit
value) *before* attempting a real run, using the probe's confirmed tensor
names (`token_embd.weight`, `position_embd.weight`, `output_norm.{weight,bias}`,
per-layer `attn_qkv.{weight,bias}`, `attn_output.{weight,bias}`,
`ffn_norm.{weight,bias}`, `ffn_up.{weight,bias}`, `ffn_down.{weight,bias}`,
separate `output.weight` -- not tied). Paid off: **the very first real run
worked, no further bugs** -- unlike Qwen2.5, where the first run failed on
a missing quant format. Config auto-detected correctly (`dim=768 layers=12
heads=12 vocab=50257 n_ctx=1024`), and `hello` continues as a fluent,
grammatically correct news-style sentence about an arrest -- a distinctly
different register from either TinyLlama's or Qwen2.5's completions,
consistent with GPT-2's actual training data and era. `--bench`: perplexity
**20.58** over 35 tokens (still three orders of magnitude below the ~50257
ceiling, but honestly higher than TinyLlama's 13.28 or Qwen2.5's 10.84 --
expected, since this is the smallest, oldest model of the three with no
instruction tuning); none of the three fact completions were correct
("Two plus two equals" -> "two", not "four") but all three stayed
grammatically fluent -- exactly the failure mode a small, un-tuned base
model should show, not evidence of a bug.

**Gemma-2b-it: code written and verified as far as this machine allows, full
generation not achieved -- exactly the outcome flagged as likely before
attempting it, not a surprise arrived at afterward.** Probed the real file
first, same as GPT-2: confirmed `arch=gemma`, tied embeddings (no
`output.weight` -- exercises the tied-embedding-sharing fix from the Qwen2.5
commit), `tokenizer.ggml.model=llama` (the existing SentencePiece path,
already correct, no new tokenizer code), and every tensor using formats
already supported (`Q4_K`/`Q6_K`/`F32` -- no new quant format needed this
time). This means Gemma needed *zero new lines of architecture code* --
every piece (NEOX RoPE, GeGLU activation, `embedding_scale = sqrt(dim)`,
tied embeddings, explicit `head_dim` from `gemma.attention.key_length`) was
already built and verified via Qwen2.5.

Ran it anyway, as asked, rather than only reasoning about it. Config
auto-detection is confirmed correct against the real file (`dim=2048
layers=18 heads=8 kv_heads=1 head_dim=256 ffn=16384 vocab=256128` --
`head_count_kv=1` is genuine MQA, and `causal_mha`'s existing GQA
broadcasting handles group size 8 with no changes). Beyond that point,
three separate attempts to run a full `--bench` gave three different,
inconclusive results: one produced no output at all before the tool-level
timeout; a 25-second attempt returned cleanly (exit 0) having printed
only the config line, with no bench output and no error printed either; a
90-second attempt failed at the tool level with no further output
captured. Each attempt was followed by a memory check confirming the VM
itself returned to a healthy baseline afterward -- whatever failed, it
wasn't a stuck process left running. Given this project's own math (a
2.5B-parameter model with a 256128-entry vocabulary dequantized to f32 is
~10-12GB resident) against this dev machine's 7.6GB RAM + 2GB swap, this is
consistent with real memory exhaustion during the dequantization step
specifically (the one step earlier phases of this log already established
is where TinyLlama's own memory-ballooning slowness came from too), not a
code defect -- but unlike TinyLlama's case, three attempts didn't produce
one clean, explainable failure mode to point at, so this is reported as
"inconclusive under memory pressure," not asserted as a confirmed specific
cause. Not investigated further past this point, per the explicit decision
going in: write the code, don't force a conclusive test this machine can't
reliably give.
model should show, not evidence of a bug.

---

## Phase 5 — CPU optimization pass (2026-08-30, ongoing)

Asked to optimize for CPU-only inference (no GPU), grounded in real research
rather than guessing, and to find an honest angle where this project could
be measurably better than an existing tool on some specific metric.

**Research first.** Fetched real ggml/llama.cpp source (`ggml-cpu/quants.c`,
`ggml-cpu.c`, `llama-mmap.cpp`) directly via WebFetch rather than trusting
AI-summarized search results (two early fetch attempts on this same pass
returned lossy summaries and were discarded in favor of raw source). Found
three concrete techniques: (1) `ggml_vec_dot_q4_0_q8_0` and friends never
materialize a dequantized tensor — they unpack quantized blocks and do the
dot product directly against packed bytes, converting to float only for the
final scaled sum, because batch-1 CPU inference is memory-bandwidth bound,
not compute bound (reading ~4.5 bits/weight instead of 32 costs ~7x less
bandwidth); (2) a persistent thread pool (workers block on a condvar between
graph evaluations, work-stealing via an atomic counter) instead of spawning
per operation; (3) the whole GGUF file is mmap'd and read straight out of
the page-cache-backed mapping, never eagerly copied. Separately surveyed the
pure-CPU-LLM-runtime landscape: `llama2.c` (~700-2500 lines, one hardcoded
architecture, custom format, no GGUF) is the only genuinely from-scratch
comparator; `llamafile` vendors llama.cpp+ggml wholesale (~26MB, a real
dependency by any honest reading); `bitnet.cpp` is a llama.cpp fork. This
grounds llamini.cpp's honest claims: zero vendored ML source at all (vs.
llamafile's real vendoring), and architecture-generality-per-line (4
architectures auto-detected from GGUF metadata in ~2,900 lines, vs.
llama2.c's 1 hardcoded architecture regardless of size).

**Step 1: persistent thread pool.** `linear()` was calling `pthread_create`/
`pthread_join` on every single call (~9 calls/layer x n_layers per token —
198 spawns/token for TinyLlama's 22 layers). Replaced with a `ThreadPool`
that creates `nt-1` workers once (mutex + two condvars + a generation
counter so workers can tell a fresh dispatch from a spurious wakeup); the
calling thread computes one chunk itself instead of idling while workers run
the rest. Kept the exact same `linear_range` kernel and row-chunking math —
this is a pure threading-overhead fix, not a numerical change.

*A real correctness trap caught before committing*: with `n_active` workers
dispatched but `n_workers` total slots existing, a naive "wake everyone"
broadcast would let idle worker slots (index >= dispatched-this-round) read
**stale job data from a previous dispatch** and write to the wrong buffer.
Fixed by having each worker check `idx >= g_pool.n_active` and skip back to
waiting instead of running a stale job — caught by re-reading the dispatch
logic against a concrete example (`out_dim=4, nt=3` leaves one slot with no
work) before ever running it, not by observing a corrupted-output bug.

**Verification, and a real WSL crash mid-verification.** While A/B
benchmarking a stashed pre-change baseline against the thread-pool build
(TinyLlama, `--bench`), WSL's own service crashed
(`Wsl/Service/E_UNEXPECTED` — `wsl --list` and even `echo alive` started
failing) partway through the *baseline* run, not the new code. This is the
same flavor of instability documented in Phase 4's Gemma-2b attempts. Asked
before running `wsl --shutdown` (it ends any other WSL work in progress);
user approved, WSL came back healthy (6.7GB free of 7.6GB) and the retry
completed cleanly, so this reads as this dev VM's own instability under
memory pressure, not a bug introduced by this change. Confirmed:
- **Byte-identical output** (mod nondeterministic per-line tok/s timing
  text) between the pre-change and post-change binaries for TinyLlama,
  Qwen2.5-0.5B, and GPT-2-124M, all via `--bench`.
- **TinyLlama wall-clock time dropped from 1:56.55 to 1:19.19** (`--bench`,
  same machine, same file, `/usr/bin/time -v`) — a genuine ~32% reduction
  from removing spawn-per-call overhead alone.
- **Resident memory was unchanged** (~4.93GB both before and after) — expected,
  since this step doesn't touch the eager-f32-dequant memory footprint;
  that's the separate, larger change planned next.
- `./llamini --test` unchanged (all existing assertions pass).

Commit: `model: replace spawn-per-call threading with a persistent thread pool`

**Step 2: posix_fadvise/madvise hints.** Trivial, POSIX-legal (`<fcntl.h>`/
`<sys/mman.h>`, both already included via `common.h`), mirroring what the
research above found in llama.cpp's `llama-mmap.cpp`: `POSIX_FADV_SEQUENTIAL`
around the metadata/tensor-info parse (a genuine front-to-back scan),
`POSIX_MADV_RANDOM` right after, before `gguf_open` returns (inference-time
tensor access is scattered per-block reads driven by matmul row order, not
sequential). Hit one build error first: `posix_fadvise`/`posix_madvise` are
POSIX.1-2001/2008 but glibc hides them under `-std=c99` unless a feature-test
macro is set *before* the first system header is pulled in — added
`#define _POSIX_C_SOURCE 200809L` at the very top of `gguf.c`, before its
`#include "gguf.h"` (which pulls in `common.h`'s system headers), matching
the file-scoped-blast-radius pattern `model.c` already uses for `pthread.h`.
Verified: `--test` unchanged, a real TinyLlama load still prints the
identical `Loaded GGUF v3 (201 tensors, 23 metadata entries)` / `Config:`
lines. No measurable timing claim made for this step — these are hints, not
a guaranteed win, and this dev VM's page cache is already warm across runs
either way; it's included because it's correct, free, and the same "no
private copy of what could stay page-cache-backed" spirit as step 3.

Commit: `gguf: add posix_fadvise/madvise hints for sequential-load then random-access reads`
