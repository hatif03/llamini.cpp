#ifndef MODEL_H
#define MODEL_H

#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "gguf.h"

// LLM model configuration. Populated at runtime from the GGUF file's own
// metadata (llama.embedding_length, llama.block_count, ...) -- see
// llama_config_from_gguf -- not hardcoded, so any real llama-architecture
// GGUF file drives its own shape.
typedef struct {
    u32 dim;            // embedding / hidden dimension
    u32 n_layers;        // number of decoder layers
    u32 n_heads;          // number of query attention heads
    u32 n_heads_kv;        // number of key/value heads (GQA; == n_heads if none)
    u32 ffn_dim;           // SwiGLU feed-forward inner dimension
    u32 vocab_size;         // size of vocabulary
    u32 seq_len;             // max sequence length
    f32 rms_eps;              // RMSNorm epsilon
    f32 rope_freq_base;        // RoPE theta base
} LLaMAConfig;

// Single Transformer Decoder Layer. Shapes (out_dim, in_dim), matching
// GGUF's row-major tensor layout, consumed by linear() below.
typedef struct {
    Tensor* attn_norm;   // [dim]
    Tensor* q_proj;      // [dim, dim]
    Tensor* k_proj;      // [kv_dim, dim]
    Tensor* v_proj;      // [kv_dim, dim]
    Tensor* o_proj;      // [dim, dim]
    Tensor* ffn_norm;    // [dim]
    Tensor* gate_proj;   // [ffn_dim, dim]
    Tensor* up_proj;     // [ffn_dim, dim]
    Tensor* down_proj;   // [dim, ffn_dim]
} DecoderLayer;

// Full LLM Model Structure
typedef struct {
    LLaMAConfig cfg;
    Tensor* embeddings;   // [vocab_size, dim]
    DecoderLayer* layers; // n_layers entries
    Tensor* final_norm;   // [dim]
    Tensor* lm_head;      // [vocab_size, dim]
} LLaMAModel;

// ------------------------------
// Core LLM Functions
// ------------------------------
void rms_norm(f32* out, const f32* x, const f32* w, u32 dim, f32 eps);
void swiglu(f32* out, const f32* gate, const f32* up, u32 dim);
void rope(f32* vec, u32 pos, u32 dim, u32 head_dim, f32 freq_base);

// y[out_dim] = w[out_dim, in_dim] @ x[in_dim] -- a GGUF weight tensor's
// row-major layout is exactly (out_dim, in_dim), so this is a direct
// row-dot-product, no transpose needed (see STDLIB.md / README limits for
// why this isn't tensor.c's matmul()).
void linear(f32* out, const f32* x, const f32* w, u32 in_dim, u32 out_dim);

// Grouped-query causal attention for one token at sequence position `pos`.
// q has n_heads * head_dim elements; k/v (this step's projections, to be
// written into the cache) have n_heads_kv * head_dim elements. Real
// softmax over cached positions 0..pos, per query head, broadcasting each
// group of (n_heads / n_heads_kv) query heads onto one KV head.
void causal_mha(const f32* q, const f32* k, const f32* v, KVCache* cache,
                 f32* out, u32 pos, u32 n_heads, u32 n_heads_kv, u32 head_dim);

// Model initialization and cleanup
int llama_model_init(LLaMAModel* model, LLaMAConfig* cfg);
void llama_model_free(LLaMAModel* model);

// Reads llama.* architecture hyperparameters out of an opened GGUF file,
// falling back to the given defaults for any missing key.
LLaMAConfig llama_config_from_gguf(GGUFFile* gf, LLaMAConfig defaults);

// Loads every weight tensor (embeddings, per-layer attn/FFN, final norm,
// output projection) from the GGUF file into an initialized model. Returns
// 0 only if every tensor was found and successfully dequantized; on any
// failure returns -1 and the caller should treat the model as unusable
// (there is no honest per-tensor partial-load fallback for 22 real layers).
int llama_load_weights(LLaMAModel* model, GGUFFile* gf);

#endif
