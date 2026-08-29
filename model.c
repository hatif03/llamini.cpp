#include "model.h"
#include "gguf.h"

/**
 * RoPE Rotary Positional Encoding
 * In-place rotation of Q and K vector 2D coordinate pairs
 * LLaMA standard base frequency = 10000
 */
void rope(f32* q, f32* k, u32 pos, u32 dim, u32 head_dim)
{
    for (u32 i = 0; i < dim; i += 2)
    {
        f32 pair_idx = (f32)(i % head_dim);
        f32 theta = powf(10000.0f, -pair_idx / (f32)head_dim);
        f32 rad = (f32)pos * theta;
        f32 cos_t = cosf(rad);
        f32 sin_t = sinf(rad);

        // Rotate Query pair
        f32 q0 = q[i];
        f32 q1 = q[i + 1];
        q[i]     = q0 * cos_t - q1 * sin_t;
        q[i + 1] = q0 * sin_t + q1 * cos_t;

        // Rotate Key pair
        f32 k0 = k[i];
        f32 k1 = k[i + 1];
        k[i]     = k0 * cos_t - k1 * sin_t;
        k[i + 1] = k0 * sin_t + k1 * cos_t;
    }
}

/**
 * Causal Multi-Head Attention with KV Cache
 * Simplified teaching implementation, single aggregated output for demo chat engine
 * Automatically saves current K/V to cache and computes attention over all past tokens
 */
void causal_mha(Tensor* q, Tensor* k, Tensor* v, KVCache* cache, Tensor* out, u32 pos, u32 n_heads)
{
    u32 dim = q->dims[1];
    u32 head_dim = dim / n_heads;

    // Write current token's Key and Value into KV Cache buffer
    memcpy(cache->key + pos * dim, k->data, dim * sizeof(f32));
    memcpy(cache->val + pos * dim, v->data, dim * sizeof(f32));
    cache->cur_seq = pos + 1;

    memset(out->data, 0, out->size * sizeof(f32));

    // Iterate all historical cached tokens (causal mask: only past + current)
    for (u32 t = 0; t <= pos; t++)
    {
        f32 raw_score = 0.0f;
        for (u32 d = 0; d < dim; d++)
        {
            raw_score += q->data[d] * (cache->key + t * dim)[d];
        }

        f32 scaled_score = raw_score / sqrtf((f32)head_dim);
        f32 attn_weight = expf(scaled_score);

        for (u32 d = 0; d < dim; d++)
        {
            out->data[d] += attn_weight * (cache->val + t * dim)[d];
        }
    }
}

/**
 * RMSNorm Vector Normalization
 * No mean subtraction; only root-mean-square scaling + learnable weight
 */
void rms_norm(f32* out, const f32* x, const f32* w, u32 dim)
{
    f32 sum_sq = 0.0f;
    for (u32 i = 0; i < dim; i++)
    {
        sum_sq += x[i] * x[i];
    }
    f32 rms = sqrtf(sum_sq / dim + EPS);
    f32 inv_rms = 1.0f / rms;
    for (u32 i = 0; i < dim; i++)
    {
        out[i] = x[i] * inv_rms * w[i];
    }
}

/**
 * SwiGLU Activation Function
 * SiLU(gate) element-wise multiplied with up projection vector
 */
void swiglu(f32* out, const f32* gate, const f32* up, u32 dim)
{
    for (u32 i = 0; i < dim; i++)
    {
        f32 silu = gate[i] / (1.0f + expf(-gate[i]));
        out[i] = silu * up[i];
    }
}

/**
 * Initialize empty model and allocate all tensor memory buffers
 * FIX: Create all decoder layer weight tensors to avoid nullptr dereference
 */
int llama_model_init(LLaMAModel* model, LLaMAConfig* cfg)
{
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

    // Allocate every weight tensor for each decoder layer
    u32 dim1d[] = {1, cfg->dim};
    u32 dim2d_qkv[] = {cfg->dim, cfg->dim};
    u32 dim_ffn[] = {cfg->dim, cfg->dim * 2};
    for(u32 l = 0; l < cfg->n_layers; l++)
    {
        DecoderLayer* layer = &model->layers[l];
        layer->attn_norm = tensor_create(2, dim1d);
        layer->ffn_norm = tensor_create(2, dim1d);
        layer->q_proj = tensor_create(2, dim2d_qkv);
        layer->k_proj = tensor_create(2, dim2d_qkv);
        layer->v_proj = tensor_create(2, dim2d_qkv);
        layer->o_proj = tensor_create(2, dim2d_qkv);
        layer->gate_proj = tensor_create(2, dim_ffn);
        layer->up_proj = tensor_create(2, dim_ffn);
        layer->down_proj = tensor_create(2, dim_ffn);
    }

    return 0;
}

/**
 * Safely release all allocated tensor memory to prevent memory leaks
 * FIX: Free each layer's inner tensors
 */
void llama_model_free(LLaMAModel* model)
{
    if (!model) return;

    // Free per-layer weights
    for(u32 l = 0; l < model->cfg.n_layers; l++)
    {
        DecoderLayer* layer = &model->layers[l];
        tensor_free(layer->attn_norm);
        tensor_free(layer->ffn_norm);
        tensor_free(layer->q_proj);
        tensor_free(layer->k_proj);
        tensor_free(layer->v_proj);
        tensor_free(layer->o_proj);
        tensor_free(layer->gate_proj);
        tensor_free(layer->up_proj);
        tensor_free(layer->down_proj);
    }
    free(model->layers);

    tensor_free(model->embeddings);
    tensor_free(model->final_norm);
    tensor_free(model->lm_head);
    memset(model, 0, sizeof(LLaMAModel));
}