#include "model.h"
#include "gguf.h" // 放在这里才正确

void rms_norm(f32* out, const f32* x, const f32* w, u32 dim) {
    f32 sum_sq = 0.0f;
    for (u32 i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    f32 rms = sqrtf(sum_sq / dim + EPS);
    f32 inv_rms = 1.0f / rms;
    for (u32 i = 0; i < dim; i++) out[i] = x[i] * inv_rms * w[i];
}

void swiglu(f32* out, const f32* gate, const f32* up, u32 dim) {
    for (u32 i = 0; i < dim; i++) {
        f32 g = gate[i] / (1.0f + expf(-gate[i]));
        out[i] = g * up[i];
    }
}

void rope(f32* q, f32* k, u32 pos, u32 dim, u32 head_dim) {
    for (u32 i = 0; i < dim; i += 2) {
        f32 theta = powf(10000.0f, -(f32)(i % head_dim) / (f32)head_dim);
        f32 cos_t = cosf(pos * theta);
        f32 sin_t = sinf(pos * theta);

        f32 q0 = q[i]; f32 q1 = q[i+1];
        q[i] = q0 * cos_t - q1 * sin_t;
        q[i+1] = q0 * sin_t + q1 * cos_t;

        f32 k0 = k[i]; f32 k1 = k[i+1];
        k[i] = k0 * cos_t - k1 * sin_t;
        k[i+1] = k0 * sin_t + k1 * cos_t;
    }
}

void causal_mha(Tensor* q, Tensor* k, Tensor* v, KVCache* cache, Tensor* out, u32 pos, u32 n_heads) {
    u32 dim = q->dims[1];
    u32 head_dim = dim / n_heads;

    memcpy(cache->key + pos * dim, q->data, dim * sizeof(f32));
    memcpy(cache->val + pos * dim, v->data, dim * sizeof(f32));
    cache->cur_seq = pos + 1;

    memset(out->data, 0, out->size * sizeof(f32));
    for (u32 t = 0; t <= pos; t++) {
        f32 score = 0.0f;
        for (u32 d = 0; d < dim; d++) score += q->data[d] * (cache->key + t * dim)[d];
        score /= sqrtf((f32)head_dim);
        f32 exp_s = expf(score);
        for (u32 d = 0; d < dim; d++) out->data[d] += exp_s * (cache->val + t * dim)[d];
    }
}

int llama_model_init(LLaMAModel* model, LLaMAConfig* cfg) {
    memset(model, 0, sizeof(LLaMAModel));
    model->cfg = *cfg;
    model->layers = (DecoderLayer*)calloc(cfg->n_layers, sizeof(DecoderLayer));
    if (!model->layers) return -1;

    u32 emb_shape[] = {cfg->vocab_size, cfg->dim};
    model->embeddings = tensor_create(2, emb_shape);

    u32 norm_shape[] = {1, cfg->dim};
    model->final_norm = tensor_create(2, norm_shape);

    u32 head_shape[] = {cfg->dim, cfg->vocab_size};
    model->lm_head = tensor_create(2, head_shape);

    return 0;
}

void llama_model_free(LLaMAModel* model) {
    if (!model) return;
    tensor_free(model->embeddings);
    tensor_free(model->final_norm);
    tensor_free(model->lm_head);
    if (model->layers) free(model->layers);
    memset(model, 0, sizeof(LLaMAModel));
}

int llama_load_weights(LLaMAModel* model, GGUFFile* gf) {
    u32 dim = model->cfg.dim;
    u64 offset = 0;
    gguf_read_f32(gf, offset, model->embeddings->data, model->embeddings->size);
    return 0;
}