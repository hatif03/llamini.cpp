#include "model.h"
#include "gguf.h"

/**
 * RoPE Rotary Positional Encoding
 * In-place rotation of Q and K vector 2D coordinate pairs
 * LLaMA standard base frequency = 10000
 */
void rope(f32* q, f32* k, u32 pos, u32 dim, u32 head_dim)
{
    // Traverse vector two elements at a time, each pair is a 2D coordinate point
    for (u32 i = 0; i < dim; i += 2)
    {
        // Calculate relative index inside single attention head
        f32 pair_idx = (f32)(i % head_dim);
        // Compute rotation base angle theta
        f32 theta = powf(10000.0f, -pair_idx / (f32)head_dim);
        // Total rotation radian = position index * base angle
        f32 rad = (f32)pos * theta;
        // Precompute trigonometric values for rotation formula
        f32 cos_t = cosf(rad);
        f32 sin_t = sinf(rad);

        // --------------------------
        // Rotate Query vector pair
        // --------------------------
        f32 q0 = q[i];
        f32 q1 = q[i + 1];
        q[i]     = q0 * cos_t - q1 * sin_t;
        q[i + 1] = q0 * sin_t + q1 * cos_t;

        // --------------------------
        // Rotate Key vector pair
        // --------------------------
        f32 k0 = k[i];
        f32 k1 = k[i + 1];
        k[i]     = k0 * cos_t - k1 * sin_t;
        k[i + 1] = k0 * sin_t + k1 * cos_t;
    }
}

/**
 * RMSNorm Vector Normalization
 * No mean subtraction; only root-mean-square scaling + learnable weight
 */
void rms_norm(f32* out, const f32* x, const f32* w, u32 dim)
{
    f32 sum_sq = 0.0f;
    // Step 1: Calculate sum of squared elements across full vector
    for (u32 i = 0; i < dim; i++)
    {
        sum_sq += x[i] * x[i];
    }
    // Step 2: Compute RMS with epsilon offset to avoid division by zero
    f32 rms = sqrtf(sum_sq / dim + EPS);
    f32 inv_rms = 1.0f / rms;
    // Step3: Normalize input and apply learned scaling weight
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
        // Compute SiLU activation for gate branch
        f32 silu = gate[i] / (1.0f + expf(-gate[i]));
        // Element-wise product between activated gate and up vector
        out[i] = silu * up[i];
    }
}

/**
 * Initialize empty model and allocate all tensor memory buffers
 */
int llama_model_init(LLaMAModel* model, LLaMAConfig* cfg)
{
    memset(model, 0, sizeof(LLaMAModel));
    model->cfg = *cfg;

    // Allocate array storage for all decoder layers
    model->layers = (DecoderLayer*)calloc(cfg->n_layers, sizeof(DecoderLayer));
    if (!model->layers) return -1;

    // Create token embedding tensor [vocab_size, hidden_dim]
    u32 emb_shape[] = {cfg->vocab_size, cfg->dim};
    model->embeddings = tensor_create(2, emb_shape);

    // Single-dimension norm weight tensor shape
    u32 norm_shape[] = {1, cfg->dim};
    model->final_norm = tensor_create(2, norm_shape);

    // LM head projects hidden state to vocabulary logits
    u32 head_shape[] = {cfg->dim, cfg->vocab_size};
    model->lm_head = tensor_create(2, head_shape);

    return 0;
}

/**
 * Safely release all allocated tensor memory to prevent memory leaks
 */
void llama_model_free(LLaMAModel* model)
{
    if (!model) return;
    tensor_free(model->embeddings);
    tensor_free(model->final_norm);
    tensor_free(model->lm_head);
    if (model->layers)
        free(model->layers);
    memset(model, 0, sizeof(LLaMAModel));
}