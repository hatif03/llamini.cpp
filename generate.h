#ifndef GENERATE_H
#define GENERATE_H

#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "model.h"

// Pick the index with the largest logit.
u32 greedy_sample(const f32* logits, u32 vocab_size);

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

#endif
