#ifndef MODEL_H
#define MODEL_H

#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "gguf.h"

// RoPE has two incompatible pairing conventions that both call themselves
// "RoPE" -- NORM pairs adjacent elements (vec[i], vec[i+1]); NEOX pairs
// elements half a head-dim apart (vec[i], vec[i+head_dim/2]). LLaMA/Mistral
// GGUF conversion permutes weights at convert-time so NORM works; Qwen2,
// Gemma, and Gemma2 do not, so they need NEOX applied directly. Picking the
// wrong one doesn't crash -- it produces fluent-looking wrong output.
typedef enum { ROPE_NORM, ROPE_NEOX } RopeType;

// The gated FFN (out = act(gate) * up, then down-projected) is shared by
// every non-GPT2 architecture here; only the gate activation differs.
typedef enum { ACT_SILU, ACT_GELU } FfnAct;

// LLM model configuration. Populated at runtime from the GGUF file's own
// metadata (under the architecture's own key namespace -- "llama.*",
// "qwen2.*", "gemma.*", ... -- see llama_config_from_gguf), not hardcoded,
// so any real GGUF file of a supported architecture drives its own shape.
typedef struct {
    u32 dim;            // embedding / hidden dimension
    u32 n_layers;        // number of decoder layers
    u32 n_heads;          // number of query attention heads
    u32 n_heads_kv;        // number of key/value heads (GQA/MQA; == n_heads if none)
    u32 head_dim;           // read from <arch>.attention.key_length when present,
                             // else derived as dim/n_heads -- do not assume the
                             // derivation holds (false for e.g. Gemma2).
    u32 ffn_dim;           // gated-FFN inner dimension
    u32 vocab_size;         // size of vocabulary
    u32 seq_len;             // max sequence length
    f32 rms_eps;              // RMSNorm epsilon
    f32 rope_freq_base;        // RoPE theta base
    RopeType rope_type;         // NORM (llama/mistral) or NEOX (qwen2/gemma)
    FfnAct ffn_act;              // SiLU (SwiGLU) or GELU (GeGLU, gemma)
    int qkv_bias;                 // 1 if attn_q/k/v carry a bias (qwen2), else 0
    f32 embedding_scale;            // multiplies the embedding lookup (gemma: sqrt(dim); else 1.0)
} LLaMAConfig;

// Single Transformer Decoder Layer. Shapes (out_dim, in_dim), matching
// GGUF's row-major tensor layout, consumed by linear() below. Bias tensors
// are NULL unless cfg.qkv_bias is set (llama_model_init only allocates
// them then) -- always NULL-check before use.
typedef struct {
    Tensor* attn_norm;   // [dim]
    Tensor* q_proj;      // [dim, dim]
    Tensor* k_proj;      // [kv_dim, dim]
    Tensor* v_proj;      // [kv_dim, dim]
    Tensor* q_bias;      // [dim] or NULL
    Tensor* k_bias;      // [kv_dim] or NULL
    Tensor* v_bias;      // [kv_dim] or NULL
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

// Gated FFN: out = act(gate) * up, elementwise (act = SiLU for SwiGLU,
// GELU for GeGLU -- see FfnAct). GPT-2's ungated GELU MLP is a different
// shape entirely (no gate at all) and lives in gpt2.c, not here.
void ffn_glu(f32* out, const f32* gate, const f32* up, u32 dim, FfnAct act);

// vec has `dim` elements, arranged as (dim/head_dim) consecutive heads.
// See RopeType above for NORM vs NEOX -- the two are not interchangeable.
void rope(f32* vec, u32 pos, u32 dim, u32 head_dim, f32 freq_base, RopeType type);

// y[out_dim] = w[out_dim, in_dim] @ x[in_dim] -- a GGUF weight tensor's
// row-major layout is exactly (out_dim, in_dim), so this is a direct
// row-dot-product, no transpose needed (see STDLIB.md / README limits for
// why this isn't tensor.c's matmul()).
void linear(f32* out, const f32* x, const f32* w, u32 in_dim, u32 out_dim);

// out[i] += bias[i] for i in 0..dim-1 -- applied after linear() when a
// projection carries a bias (qwen2's attn_q/k/v; GPT-2's every projection,
// applied in gpt2.c instead of here).
void add_bias(f32* out, const f32* bias, u32 dim);

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

// Reads architecture hyperparameters out of an opened GGUF file under the
// namespace named by its own "general.architecture" metadata (e.g.
// "qwen2.embedding_length", not "llama.embedding_length" -- every
// architecture's keys live under its own prefix), falling back to the
// given defaults for any missing key. `arch_out`, if non-NULL, receives
// the architecture string (caller frees) so main.c can dispatch to the
// right forward-pass path (e.g. GPT-2's, in gpt2.c, is not this path at
// all). Also sets rope_type/ffn_act/qkv_bias/embedding_scale/head_dim from
// known per-architecture quirks (see model.c) -- an architecture this
// function doesn't recognize gets the LLaMA defaults, which is only
// correct for architectures that are genuinely LLaMA-compatible (verify
// before trusting an unlisted one).
LLaMAConfig llama_config_from_gguf(GGUFFile* gf, LLaMAConfig defaults, char** arch_out);

// Loads every weight tensor (embeddings, per-layer attn/FFN, final norm,
// output projection) from the GGUF file into an initialized model. Returns
// 0 only if every tensor was found and successfully dequantized; on any
// failure returns -1 and the caller should treat the model as unusable
// (there is no honest per-tensor partial-load fallback for 22 real layers).
int llama_load_weights(LLaMAModel* model, GGUFFile* gf);

#endif
