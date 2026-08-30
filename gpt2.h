#ifndef GPT2_H
#define GPT2_H

#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "gguf.h"

// GPT-2 is architecturally unrelated enough to the LLaMA family (model.c)
// to warrant its own pair rather than more config flags there: fused QKV
// (one tensor, not three), learned absolute position embeddings added to
// the token embedding (no RoPE at all), LayerNorm with mean-centering and
// a bias (RMSNorm has neither), plain multi-head attention (no GQA), and
// an ungated GELU MLP (no gate tensor, unlike SwiGLU/GeGLU). Reuses
// model.h's linear() and causal_mha() (plain MHA is exactly GQA with
// n_heads_kv == n_heads, so causal_mha needs no changes) and generate.h's
// sample_token(), since sampling logic is architecture-agnostic.
typedef struct {
    u32 dim, n_layers, n_heads, ffn_dim, vocab_size, n_ctx;
    f32 ln_eps;
} GPT2Config;

typedef struct {
    Tensor *attn_norm_w, *attn_norm_b;   // [dim]
    Tensor *attn_qkv_w, *attn_qkv_b;      // [3*dim, dim], [3*dim] -- fused Q+K+V
    Tensor *attn_out_w, *attn_out_b;       // [dim, dim], [dim]
    Tensor *ffn_norm_w, *ffn_norm_b;        // [dim]
    Tensor *ffn_up_w, *ffn_up_b;              // [ffn_dim, dim], [ffn_dim]
    Tensor *ffn_down_w, *ffn_down_b;            // [dim, ffn_dim], [dim]
} GPT2Layer;

typedef struct {
    GPT2Config cfg;
    Tensor* token_embd;   // [vocab_size, dim]
    Tensor* pos_embd;      // [n_ctx, dim]
    Tensor* output_norm_w;  // [dim]
    Tensor* output_norm_b;   // [dim]
    Tensor* lm_head;          // [vocab_size, dim] -- GPT-2 ships a separate output.weight, not tied
    GPT2Layer* layers;
} GPT2Model;

// Reads gpt2.* hyperparameters (this is the only architecture this project
// treats as a fully separate model, so there is no shared "arch" dispatch
// here the way model.c's llama_config_from_gguf has -- main.c checks
// general.architecture == "gpt2" itself and routes here directly).
GPT2Config gpt2_config_from_gguf(GGUFFile* gf);

int gpt2_model_init(GPT2Model* model, GPT2Config* cfg);
void gpt2_model_free(GPT2Model* model);
int gpt2_load_weights(GPT2Model* model, GGUFFile* gf);

// One token's full forward pass at position `pos`: embed (token + position)
// -> n_layers x (LayerNorm -> fused QKV -> causal MHA -> out-proj ->
// residual -> LayerNorm -> up-proj -> GELU -> down-proj -> residual) ->
// final LayerNorm -> lm_head, writing vocab_size logits into `logits_out`.
void gpt2_forward_step(GPT2Model* model, KVCache* caches, u32 pos, u32 token_id, f32* logits_out);

// Mirrors generate_autoregressive (generate.h): prefills the whole prompt
// through the KV cache before sampling/appending anything, same three
// explicit stopping conditions. Kept as a separate function (not a
// generic callback over forward_step) since GPT-2's KV cache is sized by
// the full `dim` (plain MHA, no GQA), not kv_dim like the LLaMA path.
u32 gpt2_generate(GPT2Model* model, KVCache* caches, u32* input_tokens, u32 in_token_count,
    u32* out_tokens, u32 max_gen_tokens, u32 eos_id, f32 temp, f32 top_p);

// Teacher-forced perplexity, mirroring compute_perplexity (generate.h) --
// see that function's doc comment for the full contract.
f32 gpt2_compute_perplexity(GPT2Model* model, KVCache* caches, const u32* tokens, u32 n_tokens);

#endif
