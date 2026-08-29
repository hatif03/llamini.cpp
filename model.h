#ifndef MODEL_H
#define MODEL_H

#include "common.h"
#include "tensor.h"
#include "kv_cache.h"

// LLM Model Configuration (like TinyLlama 1.1B)
typedef struct {
    u32 dim;        // dimension of embeddings
    u32 n_layers;   // number of decoder layers
    u32 n_heads;    // number of attention heads
    u32 vocab_size; // size of vocabulary
    u32 seq_len;    // max sequence length
} LLaMAConfig;

// Single Transformer Decoder Layer
typedef struct {
    Tensor* attn_norm;   // Layer norm for attention
    Tensor* q_proj;      // Query projection
    Tensor* k_proj;      // Key projection
    Tensor* v_proj;      // Value projection
    Tensor* o_proj;      // Output projection
    Tensor* ffn_norm;    // Layer norm for feed-forward
    Tensor* gate_proj;   // SwiGLU gate
    Tensor* up_proj;     // SwiGLU up projection
    Tensor* down_proj;   // SwiGLU down projection
} DecoderLayer;

// Full LLM Model Structure
typedef struct {
    LLaMAConfig cfg;
    Tensor* embeddings;   // Token embeddings
    DecoderLayer* layers; // All transformer layers
    Tensor* final_norm;   // Final normalization
    Tensor* lm_head;      // Language model head (output)
} LLaMAModel;

// ------------------------------
// Core LLM Functions
// ------------------------------
void rms_norm(f32* out, const f32* x, const f32* w, u32 dim);
void swiglu(f32* out, const f32* gate, const f32* up, u32 dim);
void rope(f32* q, f32* k, u32 pos, u32 dim, u32 head_dim);
void causal_mha(Tensor* q, Tensor* k, Tensor* v, KVCache* cache, Tensor* out, u32 pos, u32 n_heads);

// Model initialization and cleanup
int llama_model_init(LLaMAModel* model, LLaMAConfig* cfg);
void llama_model_free(LLaMAModel* model);

#endif