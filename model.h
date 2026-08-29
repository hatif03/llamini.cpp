#ifndef MODEL_H
#define MODEL_H

#include "common.h"
#include "tensor.h"
#include "kv_cache.h"

// TinyLlama 1.1B official static hyperparameters
typedef struct {
    u32 dim;        // Hidden embedding dimension (2048)
    u32 n_layers;   // Total transformer decoder layers (22)
    u32 n_heads;    // Parallel attention heads count (32)
    u32 vocab_size; // Vocabulary token count (32000)
    u32 seq_len;    // Maximum context window length (2048)
} LLaMAConfig;

// Weight storage for one complete Transformer Decoder Layer
typedef struct {
    Tensor* attn_norm;   // RMSNorm weight before multi-head attention
    Tensor* q_proj;
    Tensor* k_proj;
    Tensor* v_proj;
    Tensor* o_proj;
    Tensor* ffn_norm;    // RMSNorm weight before feed-forward network
    Tensor* gate_proj;   // SwiGLU gate branch linear weights
    Tensor* up_proj;     // SwiGLU up branch linear weights
    Tensor* down_proj;   // SwiGLU output reduction linear weights
} DecoderLayer;

// Top-level container storing complete TinyLlama model
typedef struct {
    LLaMAConfig cfg;
    Tensor* embeddings;   // Token embedding lookup table
    DecoderLayer* layers; // Array of all decoder layers
    Tensor* final_norm;   // Final RMSNorm before logit output
    Tensor* lm_head;      // Linear projection to vocabulary logits
} LLaMAModel;

/**
 * @brief RMSNorm normalization core calculation
 * @param out Normalized output vector buffer
 * @param x Raw input feature vector
 * @param w Learnable RMSNorm scaling weights loaded from GGUF
 * @param dim Hidden vector dimension size
 */
void rms_norm(f32* out, const f32* x, const f32* w, u32 dim);

/**
 * @brief SwiGLU activation for LLaMA feed forward network
 * @param out Final FFN intermediate output vector
 * @param gate Output of gate_proj linear layer
 * @param up Output of up_proj linear layer
 * @param dim Hidden dimension length of vectors
 */
void swiglu(f32* out, const f32* gate, const f32* up, u32 dim);

/**
 * @brief RoPE Rotary Positional Encoding
 * Rotate Query and Key vector 2D coordinate pairs in-place
 * @param q Query vector buffer (modified after function call)
 * @param k Key vector buffer (modified after function call)
 * @param pos Current token sequence position index
 * @param dim Total hidden dimension of Q/K vector
 * @param head_dim Dimension of single attention head (dim / n_heads)
 */
void rope(f32* q, f32* k, u32 pos, u32 dim, u32 head_dim);

/**
 * @brief Initialize empty LLaMA model & allocate all tensor memory buffers
 * @param model Pointer to empty LLaMAModel struct
 * @param cfg Model hyperparameter configuration
 * @return 0 success, -1 memory allocation failure
 */
int llama_model_init(LLaMAModel* model, LLaMAConfig* cfg);

/**
 * @brief Recursively free all tensor memory inside LLaMAModel to avoid leaks
 * @param model Loaded model struct to destroy
 */
void llama_model_free(LLaMAModel* model);

#endif