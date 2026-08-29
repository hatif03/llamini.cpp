#ifndef GENERATE_H
#define GENERATE_H

#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "model.h"

// Pick the index with the largest logit.
u32 greedy_sample(const f32* logits, u32 vocab_size);

// Full autoregressive single-turn generation: for each step, embeds the
// most recent token and runs it through every decoder layer (RMSNorm ->
// GQA causal attention with a per-layer KV cache -> residual -> RMSNorm
// -> SwiGLU FFN -> residual), then the final norm and lm_head projection,
// greedily decoding the next token. `caches` must have model->cfg.n_layers
// entries, one KV cache per layer (see kv_cache_init, called with
// n_heads_kv * head_dim as its "dim").
u32 generate_autoregressive(LLaMAModel* model, KVCache* caches,
    u32* input_tokens, u32 in_token_count,
    u32* out_tokens, u32 max_gen_tokens, u32 eos_id);

#endif
