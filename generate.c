#include "generate.h"

/**
 * Greedy search: select token with maximum log probability
 * Greedy sampling algorithm, pick the token index with the largest logit value
 * @param logits Tensor storing raw logit scores of all vocabulary tokens
 * @return Integer ID of the most probable next token
 */
u32 greedy_sample(Tensor* logits)
{
    // Initialize index and value with the first token's logit
    u32 max_idx = 0;
    f32 max_val = logits->data[0];

    // Traverse all vocabulary logit values to find the maximum
    for (u32 i = 1; i < logits->size; i++)
    {
        // Update maximum value and corresponding token index if larger logit found
        if (logits->data[i] > max_val)
        {
            max_val = logits->data[i];
            max_idx = i;
        }
    }
    // Return token ID with highest prediction score
    return max_idx;
}

/**
 * Minimal demo autoregressive generation loop
 * All weights are zero initialized for demo, no real GGUF loaded
 * Core autoregressive inference loop to generate token sequence step by step
 * @param model Pointer to initialized LLaMA model struct
 * @param cache Pointer to KV Cache for attention state reuse
 * @param input_tokens Token ID array of user input prompt
 * @param in_token_count Valid token count of input prompt
 * @param out_tokens Output buffer storing full prompt + generated reply tokens
 * @param max_gen_tokens Upper limit of tokens allowed to generate
 * @return Total number of tokens (prompt + generated reply)
 */
u32 generate_autoregressive(LLaMAModel* model, KVCache* cache,
    u32* input_tokens, u32 in_token_count,
    u32* out_tokens, u32 max_gen_tokens)
{
    // Copy all input prompt tokens into output buffer as sequence prefix
    u32 total_tokens = in_token_count;
    memcpy(out_tokens, input_tokens, in_token_count * sizeof(u32));

    // Define tensor dimension: single token, hidden dimension same as model config
    u32 vec_shape[] = {1, model->cfg.dim};
    // Hidden state tensor storing embedding vector of current token
    Tensor* hidden = tensor_create(2, vec_shape);
    // Query tensor for multi-head attention calculation
    Tensor* q = tensor_create(2, vec_shape);
    // Key tensor for multi-head attention calculation
    Tensor* k = tensor_create(2, vec_shape);
    // Value tensor for multi-head attention calculation
    Tensor* v = tensor_create(2, vec_shape);
    // Output tensor to store attention layer computation result
    Tensor* attn_out = tensor_create(2, vec_shape);

    // Logit tensor dimension: single token, full vocabulary size
    u32 logit_shape[] = {1, model->cfg.vocab_size};
    // Tensor storing raw vocabulary prediction scores
    Tensor* logits = tensor_create(2, logit_shape);

    // Autoregressive loop: generate new tokens up to max limit
    for (u32 step = 0; step < max_gen_tokens; step++)
    {
        // Get sequence position index of the last token in current full sequence
        u32 current_pos = total_tokens - 1;
        // Demo placeholder: fill embedding vector with all zeros (dummy weight mode)
        memset(hidden->data, 0, hidden->size * sizeof(f32));

        // Apply RMS normalization to hidden state before attention projection
        rms_norm(q->data, hidden->data, model->layers[0].attn_norm->data, model->cfg.dim);
        // Copy normalized Q vector to K and V for simplified demo logic
        memcpy(k->data, q->data, model->cfg.dim * sizeof(f32));
        memcpy(v->data, q->data, model->cfg.dim * sizeof(f32));

        // Apply RoPE rotary positional encoding to Q and K vectors
        rope(q->data, k->data, current_pos, model->cfg.dim, model->cfg.dim / model->cfg.n_heads);

        // Run causal masked multi-head attention, reuse history states via KV Cache
        causal_mha(q, k, v, cache, attn_out, current_pos, model->cfg.n_heads);

        // Demo placeholder: reset all logit values to zero for testing
        memset(logits->data, 0, logits->size * sizeof(f32));
        // Pick next token ID via greedy sampling
        u32 next_tok = greedy_sample(logits);
        // Append newly generated token to full output token sequence
        out_tokens[total_tokens++] = next_tok;

        // Terminate generation early if end-of-sequence token is predicted
        if (next_tok == TOKEN_EOS)
            break;
    }

    // Release all dynamically allocated temporary tensors to prevent memory leaks
    tensor_free(hidden);
    tensor_free(q);
    tensor_free(k);
    tensor_free(v);
    tensor_free(attn_out);
    tensor_free(logits);
    // Return total length of combined prompt + generated token sequence
    return total_tokens;
}