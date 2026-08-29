#include "generate.h"

u32 greedy_sample(const f32* logits, u32 vocab_size) {
    u32 max_idx = 0;
    f32 max_val = logits[0];
    for (u32 i = 1; i < vocab_size; i++)
        if (logits[i] > max_val) { max_val = logits[i]; max_idx = i; }
    return max_idx;
}

u32 generate_autoregressive(LLaMAModel* model, KVCache* caches,
    u32* input_tokens, u32 in_token_count,
    u32* out_tokens, u32 max_gen_tokens, u32 eos_id)
{
    u32 total_tokens = in_token_count;
    memcpy(out_tokens, input_tokens, in_token_count * sizeof(u32));

    u32 dim = model->cfg.dim;
    u32 ffn = model->cfg.ffn_dim;
    u32 n_heads = model->cfg.n_heads;
    u32 n_heads_kv = model->cfg.n_heads_kv;
    u32 head_dim = dim / n_heads;
    u32 kv_dim = n_heads_kv * head_dim;

    f32* hidden    = (f32*)malloc(dim * sizeof(f32));
    f32* xnorm     = (f32*)malloc(dim * sizeof(f32));
    f32* q         = (f32*)malloc(dim * sizeof(f32));
    f32* k         = (f32*)malloc(kv_dim * sizeof(f32));
    f32* v         = (f32*)malloc(kv_dim * sizeof(f32));
    f32* attn_out  = (f32*)malloc(dim * sizeof(f32));
    f32* attn_proj = (f32*)malloc(dim * sizeof(f32));
    f32* gate      = (f32*)malloc(ffn * sizeof(f32));
    f32* up        = (f32*)malloc(ffn * sizeof(f32));
    f32* ffn_hid   = (f32*)malloc(ffn * sizeof(f32));
    f32* ffn_out   = (f32*)malloc(dim * sizeof(f32));
    f32* logits    = (f32*)malloc((u64)model->cfg.vocab_size * sizeof(f32));

    for (u32 step = 0; step < max_gen_tokens; step++) {
        u32 current_pos = total_tokens - 1;
        u32 cur_tok = out_tokens[current_pos];
        memcpy(hidden, model->embeddings->data + (u64)cur_tok * dim, dim * sizeof(f32));

        for (u32 l = 0; l < model->cfg.n_layers; l++) {
            DecoderLayer* layer = &model->layers[l];

            rms_norm(xnorm, hidden, layer->attn_norm->data, dim, model->cfg.rms_eps);
            linear(q, xnorm, layer->q_proj->data, dim, dim);
            linear(k, xnorm, layer->k_proj->data, dim, kv_dim);
            linear(v, xnorm, layer->v_proj->data, dim, kv_dim);

            rope(q, current_pos, dim, head_dim, model->cfg.rope_freq_base);
            rope(k, current_pos, kv_dim, head_dim, model->cfg.rope_freq_base);

            causal_mha(q, k, v, &caches[l], attn_out, current_pos, n_heads, n_heads_kv, head_dim);
            linear(attn_proj, attn_out, layer->o_proj->data, dim, dim);
            for (u32 d = 0; d < dim; d++) hidden[d] += attn_proj[d];

            rms_norm(xnorm, hidden, layer->ffn_norm->data, dim, model->cfg.rms_eps);
            linear(gate, xnorm, layer->gate_proj->data, dim, ffn);
            linear(up,   xnorm, layer->up_proj->data,   dim, ffn);
            swiglu(ffn_hid, gate, up, ffn);
            linear(ffn_out, ffn_hid, layer->down_proj->data, ffn, dim);
            for (u32 d = 0; d < dim; d++) hidden[d] += ffn_out[d];
        }

        rms_norm(xnorm, hidden, model->final_norm->data, dim, model->cfg.rms_eps);
        linear(logits, xnorm, model->lm_head->data, dim, model->cfg.vocab_size);

        u32 next_tok = greedy_sample(logits, model->cfg.vocab_size);
        out_tokens[total_tokens++] = next_tok;
        if (next_tok == eos_id) break;
    }

    free(hidden); free(xnorm); free(q); free(k); free(v);
    free(attn_out); free(attn_proj); free(gate); free(up); free(ffn_hid); free(ffn_out); free(logits);
    return total_tokens;
}
