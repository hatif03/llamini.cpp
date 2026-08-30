#include "gpt2.h"
#include "model.h"    // linear(), causal_mha() -- both architecture-agnostic, reused as-is
#include "generate.h" // sample_token(), greedy_sample() -- sampling is architecture-agnostic too

GPT2Config gpt2_config_from_gguf(GGUFFile* gf) {
    GPT2Config cfg;
    cfg.dim        = gguf_get_u32(gf, "gpt2.embedding_length", 768);
    cfg.n_layers    = gguf_get_u32(gf, "gpt2.block_count", 12);
    cfg.n_heads      = gguf_get_u32(gf, "gpt2.attention.head_count", 12);
    cfg.ffn_dim        = gguf_get_u32(gf, "gpt2.feed_forward_length", cfg.dim * 4);
    u32 ctx = gguf_get_u32(gf, "gpt2.context_length", 1024);
    cfg.n_ctx = ctx < MAX_SEQ_LEN ? ctx : MAX_SEQ_LEN;
    cfg.ln_eps = 1e-5f; // GPT-2's own default (HF GPT2Config); no dedicated GGUF key for this arch

    u64 vocab_n = gguf_meta_array_len(gf, "tokenizer.ggml.tokens", NULL);
    cfg.vocab_size = vocab_n > 0 ? (u32)vocab_n : 50257;
    return cfg;
}

int gpt2_model_init(GPT2Model* model, GPT2Config* cfg) {
    memset(model, 0, sizeof(GPT2Model));
    model->cfg = *cfg;
    model->layers = (GPT2Layer*)calloc(cfg->n_layers, sizeof(GPT2Layer));
    if (!model->layers) return -1;

    u32 dim = cfg->dim, ffn = cfg->ffn_dim;
    u32 s_emb[]  = {cfg->vocab_size, dim};
    model->token_embd = tensor_create(2, s_emb);
    u32 s_pos[]  = {cfg->n_ctx, dim};
    model->pos_embd = tensor_create(2, s_pos);
    u32 s_v[]    = {1, dim};
    model->output_norm_w = tensor_create(2, s_v);
    model->output_norm_b = tensor_create(2, s_v);
    u32 s_head[] = {cfg->vocab_size, dim};
    model->lm_head = tensor_create(2, s_head);

    for (u32 l = 0; l < cfg->n_layers; l++) {
        GPT2Layer* layer = &model->layers[l];
        u32 s_norm[]  = {1, dim};
        u32 s_qkv_w[] = {3 * dim, dim};
        u32 s_qkv_b[] = {1, 3 * dim};
        u32 s_out[]   = {dim, dim};
        u32 s_up_w[]  = {ffn, dim};
        u32 s_up_b[]  = {1, ffn};
        u32 s_down[]  = {dim, ffn};

        layer->attn_norm_w = tensor_create(2, s_norm);
        layer->attn_norm_b = tensor_create(2, s_norm);
        layer->attn_qkv_w  = tensor_create(2, s_qkv_w);
        layer->attn_qkv_b  = tensor_create(2, s_qkv_b);
        layer->attn_out_w  = tensor_create(2, s_out);
        layer->attn_out_b  = tensor_create(2, s_norm);
        layer->ffn_norm_w  = tensor_create(2, s_norm);
        layer->ffn_norm_b  = tensor_create(2, s_norm);
        layer->ffn_up_w    = tensor_create(2, s_up_w);
        layer->ffn_up_b    = tensor_create(2, s_up_b);
        layer->ffn_down_w  = tensor_create(2, s_down);
        layer->ffn_down_b  = tensor_create(2, s_norm);
    }
    return 0;
}

void gpt2_model_free(GPT2Model* model) {
    if (!model) return;
    tensor_free(model->token_embd);
    tensor_free(model->pos_embd);
    tensor_free(model->output_norm_w);
    tensor_free(model->output_norm_b);
    tensor_free(model->lm_head);
    if (model->layers) {
        for (u32 l = 0; l < model->cfg.n_layers; l++) {
            GPT2Layer* layer = &model->layers[l];
            tensor_free(layer->attn_norm_w); tensor_free(layer->attn_norm_b);
            tensor_free(layer->attn_qkv_w);  tensor_free(layer->attn_qkv_b);
            tensor_free(layer->attn_out_w);  tensor_free(layer->attn_out_b);
            tensor_free(layer->ffn_norm_w);  tensor_free(layer->ffn_norm_b);
            tensor_free(layer->ffn_up_w);    tensor_free(layer->ffn_up_b);
            tensor_free(layer->ffn_down_w);  tensor_free(layer->ffn_down_b);
        }
        free(model->layers);
    }
    memset(model, 0, sizeof(GPT2Model));
}

static int load_tensor(GGUFFile* gf, const char* name, Tensor* t) {
    const GGUFTensorInfo* info = gguf_find_tensor(gf, name);
    if (!info) return -1;
    return gguf_dequantize_tensor(gf, info, t->data, t->size);
}

int gpt2_load_weights(GPT2Model* model, GGUFFile* gf) {
    if (load_tensor(gf, "token_embd.weight", model->token_embd) != 0) return -1;
    if (load_tensor(gf, "position_embd.weight", model->pos_embd) != 0) return -1;
    if (load_tensor(gf, "output_norm.weight", model->output_norm_w) != 0) return -1;
    if (load_tensor(gf, "output_norm.bias", model->output_norm_b) != 0) return -1;
    if (load_tensor(gf, "output.weight", model->lm_head) != 0) return -1;

    char name[64];
    for (u32 l = 0; l < model->cfg.n_layers; l++) {
        GPT2Layer* layer = &model->layers[l];
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
        if (load_tensor(gf, name, layer->attn_norm_w) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.attn_norm.bias", l);
        if (load_tensor(gf, name, layer->attn_norm_b) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", l);
        if (load_tensor(gf, name, layer->attn_qkv_w) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.attn_qkv.bias", l);
        if (load_tensor(gf, name, layer->attn_qkv_b) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", l);
        if (load_tensor(gf, name, layer->attn_out_w) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.attn_output.bias", l);
        if (load_tensor(gf, name, layer->attn_out_b) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
        if (load_tensor(gf, name, layer->ffn_norm_w) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.bias", l);
        if (load_tensor(gf, name, layer->ffn_norm_b) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", l);
        if (load_tensor(gf, name, layer->ffn_up_w) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_up.bias", l);
        if (load_tensor(gf, name, layer->ffn_up_b) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", l);
        if (load_tensor(gf, name, layer->ffn_down_w) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_down.bias", l);
        if (load_tensor(gf, name, layer->ffn_down_b) != 0) return -1;
    }
    return 0;
}

// LayerNorm: mean-centered, unlike RMSNorm (model.c), and carries a bias.
static void layer_norm(f32* out, const f32* x, const f32* w, const f32* b, u32 dim, f32 eps) {
    f32 mean = 0.0f;
    for (u32 i = 0; i < dim; i++) mean += x[i];
    mean /= (f32)dim;
    f32 var = 0.0f;
    for (u32 i = 0; i < dim; i++) { f32 d = x[i] - mean; var += d * d; }
    var /= (f32)dim;
    f32 inv_std = 1.0f / sqrtf(var + eps);
    for (u32 i = 0; i < dim; i++) out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
}

// GGML's tanh GELU approximation -- same curve as model.c's static gelu(),
// duplicated rather than exported: GPT-2 and the LLaMA-family path
// (model.c) are otherwise unrelated, and this is three lines.
static f32 gelu(f32 x) {
    const f32 SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const f32 GELU_COEF_A = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

void gpt2_forward_step(GPT2Model* model, KVCache* caches, u32 pos, u32 token_id, f32* logits_out) {
    u32 dim = model->cfg.dim, ffn = model->cfg.ffn_dim, n_heads = model->cfg.n_heads;
    u32 head_dim = dim / n_heads;

    f32* hidden  = (f32*)malloc(dim * sizeof(f32));
    f32* xnorm   = (f32*)malloc(dim * sizeof(f32));
    f32* qkv     = (f32*)malloc(3 * dim * sizeof(f32));
    f32* attn_out = (f32*)malloc(dim * sizeof(f32));
    f32* attn_proj = (f32*)malloc(dim * sizeof(f32));
    f32* ffn_hid  = (f32*)malloc(ffn * sizeof(f32));
    f32* ffn_out  = (f32*)malloc(dim * sizeof(f32));

    // Embed: token + learned absolute position (no RoPE for GPT-2 at all).
    for (u32 d = 0; d < dim; d++)
        hidden[d] = model->token_embd->data[(u64)token_id * dim + d] + model->pos_embd->data[(u64)pos * dim + d];

    for (u32 l = 0; l < model->cfg.n_layers; l++) {
        GPT2Layer* layer = &model->layers[l];

        layer_norm(xnorm, hidden, layer->attn_norm_w->data, layer->attn_norm_b->data, dim, model->cfg.ln_eps);
        linear(qkv, xnorm, layer->attn_qkv_w->data, dim, 3 * dim);
        add_bias(qkv, layer->attn_qkv_b->data, 3 * dim);
        f32* q = qkv; f32* k = qkv + dim; f32* v = qkv + 2 * dim; // fused QKV: contiguous slices, no RoPE to apply

        causal_mha(q, k, v, &caches[l], attn_out, pos, n_heads, n_heads, head_dim); // plain MHA == GQA with n_heads_kv == n_heads
        linear(attn_proj, attn_out, layer->attn_out_w->data, dim, dim);
        add_bias(attn_proj, layer->attn_out_b->data, dim);
        for (u32 d = 0; d < dim; d++) hidden[d] += attn_proj[d];

        layer_norm(xnorm, hidden, layer->ffn_norm_w->data, layer->ffn_norm_b->data, dim, model->cfg.ln_eps);
        linear(ffn_hid, xnorm, layer->ffn_up_w->data, dim, ffn);
        add_bias(ffn_hid, layer->ffn_up_b->data, ffn);
        for (u32 d = 0; d < ffn; d++) ffn_hid[d] = gelu(ffn_hid[d]); // ungated: no gate tensor, unlike SwiGLU/GeGLU
        linear(ffn_out, ffn_hid, layer->ffn_down_w->data, ffn, dim);
        add_bias(ffn_out, layer->ffn_down_b->data, dim);
        for (u32 d = 0; d < dim; d++) hidden[d] += ffn_out[d];
    }

    layer_norm(xnorm, hidden, model->output_norm_w->data, model->output_norm_b->data, dim, model->cfg.ln_eps);
    linear(logits_out, xnorm, model->lm_head->data, dim, model->cfg.vocab_size);

    free(hidden); free(xnorm); free(qkv); free(attn_out); free(attn_proj); free(ffn_hid); free(ffn_out);
}

u32 gpt2_generate(GPT2Model* model, KVCache* caches, u32* input_tokens, u32 in_token_count,
    u32* out_tokens, u32 max_gen_tokens, u32 eos_id, f32 temp, f32 top_p)
{
    u32 total_tokens = in_token_count;
    memcpy(out_tokens, input_tokens, in_token_count * sizeof(u32));
    f32* logits = (f32*)malloc((u64)model->cfg.vocab_size * sizeof(f32));

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    u32 pos = 0;
    u32 max_pos = caches[0].max_seq < model->cfg.n_ctx ? caches[0].max_seq : model->cfg.n_ctx;
    for (;;) {
        if (pos >= max_pos) break;
        u32 current_pos = pos;
        u32 cur_tok = out_tokens[current_pos];
        gpt2_forward_step(model, caches, current_pos, cur_tok, logits);

        pos++;
        if (pos < in_token_count) continue;

        u32 next_tok = sample_token(logits, model->cfg.vocab_size, temp, top_p);
        out_tokens[total_tokens++] = next_tok;
        if (next_tok == eos_id) break;
        if (total_tokens - in_token_count >= max_gen_tokens) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    u32 n_generated = total_tokens - in_token_count;
    double elapsed = (double)(t_end.tv_sec - t_start.tv_sec) + (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    if (n_generated > 0 && elapsed > 0.0)
        fprintf(stderr, "[%u tokens in %.2fs, %.2f tok/s]\n", n_generated, elapsed, n_generated / elapsed);

    free(logits);
    return total_tokens;
}

f32 gpt2_compute_perplexity(GPT2Model* model, KVCache* caches, const u32* tokens, u32 n_tokens) {
    for (u32 l = 0; l < model->cfg.n_layers; l++) kv_cache_reset(&caches[l]);

    u32 vocab = model->cfg.vocab_size;
    f32* logits = (f32*)malloc((u64)vocab * sizeof(f32));
    u32 max_pos = caches[0].max_seq < model->cfg.n_ctx ? caches[0].max_seq : model->cfg.n_ctx;

    double nll_sum = 0.0;
    u32 count = 0;
    for (u32 pos = 0; pos + 1 < n_tokens && pos < max_pos; pos++) {
        gpt2_forward_step(model, caches, pos, tokens[pos], logits);

        f32 mx = logits[0];
        for (u32 i = 1; i < vocab; i++) if (logits[i] > mx) mx = logits[i];
        f32 sum_exp = 0.0f;
        for (u32 i = 0; i < vocab; i++) sum_exp += expf(logits[i] - mx);
        f32 log_prob = (logits[tokens[pos + 1]] - mx) - logf(sum_exp);

        nll_sum += -(double)log_prob;
        count++;
    }

    free(logits);
    if (count == 0) return -1.0f;
    return expf((f32)(nll_sum / count));
}
