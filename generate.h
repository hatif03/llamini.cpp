#ifndef GENERATE_H
#define GENERATE_H

#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "model.h"

// Pick the index with the largest logit.
u32 greedy_sample(const f32* logits, u32 vocab_size);

// Runs one token through the full forward pass at sequence position `pos`
// (embedding lookup -> every decoder layer, RMSNorm -> GQA causal attention
// with a per-layer KV cache -> residual -> RMSNorm -> SwiGLU FFN ->
// residual -> final norm -> lm_head), writing model->cfg.vocab_size logits
// into `logits_out`. Does not sample or track position bookkeeping --
// shared by generate_autoregressive and compute_perplexity below, so there
// is exactly one place this forward pass is written.
void forward_step(LLaMAModel* model, KVCache* caches, u32 pos, u32 token_id, f32* logits_out);

// Full autoregressive single-turn generation. First prefills every prompt
// position through the full layer stack (RMSNorm -> GQA causal attention
// with a per-layer KV cache -> residual -> RMSNorm -> SwiGLU FFN ->
// residual -> final norm -> lm_head), populating each position's real KV
// cache entry, before sampling or appending anything; only once the whole
// prompt has been seen does it start greedily decoding new tokens the same
// way. `caches` must have model->cfg.n_layers entries, one KV cache per
// layer (see kv_cache_init, called with n_heads_kv * head_dim as its "dim").
// Cost scales with prompt length (a P-token prompt costs P forward passes
// before the first generated token), not O(1) -- see README limits.
u32 generate_autoregressive(LLaMAModel* model, KVCache* caches,
    u32* input_tokens, u32 in_token_count,
    u32* out_tokens, u32 max_gen_tokens, u32 eos_id);

// Teacher-forced perplexity over tokens[0..n_tokens-1]: resets every
// layer's KV cache, then for each position scores the probability
// forward_step's logits assign to the *actual* next token in `tokens`
// (never sampling), and returns exp(mean negative log-likelihood). A
// broken/randomly-wired forward pass lands near a uniform-random ceiling
// of ~vocab_size; a working language model should land dramatically lower
// -- see main.c's --bench and README's "Correctness evidence". Returns
// -1.0f if n_tokens is too short (< 2) to evaluate.
f32 compute_perplexity(LLaMAModel* model, KVCache* caches, const u32* tokens, u32 n_tokens);

#endif
