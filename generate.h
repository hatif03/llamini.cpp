#ifndef GENERATE_H
#define GENERATE_H

#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "model.h"

/**
 * Greedy sampling: pick index with maximum logit value
 * @param logits Input 1D logit tensor (vocab_size elements)
 * @return Token ID with highest probability
 */
u32 greedy_sample(Tensor* logits);

/**
 * Full autoregressive single-turn generation loop
 * Iteratively predict next token until EOS or max length
 * @param model Loaded LLaMA model structure
 * @param cache Preallocated KV Cache
 * @param input_tokens Initial prompt token array
 * @param in_token_count Number of input prompt tokens
 * @param out_tokens Output generated token buffer
 * @param max_gen_tokens Maximum allowed generated token count
 * @return Total tokens (input + generated)
 */
u32 generate_autoregressive(LLaMAModel* model, KVCache* cache,
    u32* input_tokens, u32 in_token_count,
    u32* out_tokens, u32 max_gen_tokens);

#endif