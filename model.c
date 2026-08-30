#include "model.h"
#include <pthread.h> // model.c only -- keep the blast radius of adding threading small

void rms_norm(f32* out, const f32* x, const f32* w, u32 dim, f32 eps) {
    f32 sum_sq = 0.0f;
    for (u32 i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    f32 rms = sqrtf(sum_sq / dim + eps);
    f32 inv_rms = 1.0f / rms;
    for (u32 i = 0; i < dim; i++) out[i] = x[i] * inv_rms * w[i];
}

// GGML's tanh approximation (matches ggml_gelu_f32) -- used for GeGLU
// (gemma). Not the same curve as SiLU; the two are not interchangeable.
static f32 gelu(f32 x) {
    const f32 SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const f32 GELU_COEF_A = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

void ffn_glu(f32* out, const f32* gate, const f32* up, u32 dim, FfnAct act) {
    for (u32 i = 0; i < dim; i++) {
        f32 g = (act == ACT_GELU) ? gelu(gate[i]) : gate[i] / (1.0f + expf(-gate[i])); // SiLU
        out[i] = g * up[i];
    }
}

void add_bias(f32* out, const f32* bias, u32 dim) {
    for (u32 i = 0; i < dim; i++) out[i] += bias[i];
}

void rope(f32* vec, u32 pos, u32 dim, u32 head_dim, f32 freq_base, RopeType type) {
    if (type == ROPE_NEOX) {
        // Pairs vec[i] with vec[i + head_dim/2] within each head, rather
        // than adjacent elements -- see RopeType's comment in model.h.
        u32 half = head_dim / 2;
        for (u32 base = 0; base < dim; base += head_dim) {
            for (u32 j = 0; j < half; j++) {
                f32 theta = powf(freq_base, -(f32)(2 * j) / (f32)head_dim);
                f32 cos_t = cosf(pos * theta);
                f32 sin_t = sinf(pos * theta);
                u32 i0 = base + j, i1 = base + j + half;
                f32 v0 = vec[i0]; f32 v1 = vec[i1];
                vec[i0] = v0 * cos_t - v1 * sin_t;
                vec[i1] = v0 * sin_t + v1 * cos_t;
            }
        }
        return;
    }
    for (u32 i = 0; i < dim; i += 2) {
        f32 theta = powf(freq_base, -(f32)(i % head_dim) / (f32)head_dim);
        f32 cos_t = cosf(pos * theta);
        f32 sin_t = sinf(pos * theta);

        f32 v0 = vec[i]; f32 v1 = vec[i + 1];
        vec[i]     = v0 * cos_t - v1 * sin_t;
        vec[i + 1] = v0 * sin_t + v1 * cos_t;
    }
}

#define MAX_THREADS    32
#define LINEAR_PAR_MIN (1u << 20) // out_dim*in_dim below this: not worth spawning threads

// Shared kernel for both the serial and threaded paths below -- exactly the
// original single-loop body, just over a caller-chosen row range, so there
// is only one place this dot product is written.
static void linear_range(f32* out, const f32* x, const f32* w, u32 in_dim, u32 o0, u32 o1) {
    for (u32 o = o0; o < o1; o++) {
        f32 sum = 0.0f;
        const f32* row = w + (u64)o * in_dim;
        for (u32 i = 0; i < in_dim; i++) sum += row[i] * x[i];
        out[o] = sum;
    }
}

typedef struct { f32* out; const f32* x; const f32* w; u32 in_dim, o0, o1; } LinearJob;

static void* linear_worker(void* arg) {
    LinearJob* j = (LinearJob*)arg;
    linear_range(j->out, j->x, j->w, j->in_dim, j->o0, j->o1);
    return NULL;
}

// ponytail: cached after the first call, read only from the single
// (non-threaded) caller thread -- LLAMINI_THREADS overrides nproc, useful
// under a cgroup CPU limit where sysconf overreports.
static int n_threads(void) {
    static int n = 0;
    if (!n) {
        const char* e = getenv("LLAMINI_THREADS");
        n = e ? atoi(e) : (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (n < 1) n = 1;
        if (n > MAX_THREADS) n = MAX_THREADS;
    }
    return n;
}

void linear(f32* out, const f32* x, const f32* w, u32 in_dim, u32 out_dim) {
    int nt = n_threads();
    // Small matmuls (k_proj/v_proj) do less work than spawning threads
    // would cost -- stay serial. Every linear() call site's out/x buffers
    // are distinct (verified in generate.c); this dispatcher assumes that.
    if (nt < 2 || (u64)out_dim * in_dim < LINEAR_PAR_MIN || out_dim < (u32)nt) {
        linear_range(out, x, w, in_dim, 0, out_dim);
        return;
    }

    pthread_t th[MAX_THREADS];
    LinearJob jobs[MAX_THREADS];
    u32 chunk = (out_dim + (u32)nt - 1) / (u32)nt;
    int spawned = 0;
    for (int t = 0; t < nt; t++) {
        u32 o0 = (u32)t * chunk;
        if (o0 >= out_dim) break;
        u32 o1 = o0 + chunk > out_dim ? out_dim : o0 + chunk;
        jobs[t] = (LinearJob){ out, x, w, in_dim, o0, o1 };
        if (pthread_create(&th[spawned], NULL, linear_worker, &jobs[t]) != 0) {
            // Couldn't spawn this or any later worker -- finish every
            // remaining row serially so no output row is ever left unwritten.
            linear_range(out, x, w, in_dim, o0, out_dim);
            break;
        }
        spawned++;
    }
    for (int t = 0; t < spawned; t++) pthread_join(th[t], NULL);
}

void causal_mha(const f32* q, const f32* k, const f32* v, KVCache* cache,
                 f32* out, u32 pos, u32 n_heads, u32 n_heads_kv, u32 head_dim) {
    u32 kv_dim = n_heads_kv * head_dim;
    u32 group = n_heads / n_heads_kv; // query heads sharing one KV head

    memcpy(cache->key + (u64)pos * kv_dim, k, kv_dim * sizeof(f32));
    memcpy(cache->val + (u64)pos * kv_dim, v, kv_dim * sizeof(f32));
    cache->cur_seq = pos + 1;

    // ponytail: scores buffer sized to MAX_SEQ_LEN so this never allocates
    // per call; pos < cfg.seq_len <= MAX_SEQ_LEN is enforced at config load.
    // NOT thread-safe: this is deliberately never called from more than one
    // thread at once (unlike linear(), just above, which is). Per-token
    // compute here is negligible next to linear()'s, so it isn't worth
    // parallelizing -- if that ever changes, this must become per-thread.
    static f32 scores[MAX_SEQ_LEN];

    for (u32 h = 0; h < n_heads; h++) {
        u32 kvh = h / group;
        const f32* qh = q + (u64)h * head_dim;

        f32 max_score = -1e30f;
        for (u32 t = 0; t <= pos; t++) {
            const f32* kt = cache->key + (u64)t * kv_dim + (u64)kvh * head_dim;
            f32 score = 0.0f;
            for (u32 d = 0; d < head_dim; d++) score += qh[d] * kt[d];
            score /= sqrtf((f32)head_dim);
            scores[t] = score;
            if (score > max_score) max_score = score;
        }
        f32 sum_exp = 0.0f;
        for (u32 t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        f32* oh = out + (u64)h * head_dim;
        memset(oh, 0, head_dim * sizeof(f32));
        for (u32 t = 0; t <= pos; t++) {
            f32 w = scores[t] / sum_exp;
            const f32* vt = cache->val + (u64)t * kv_dim + (u64)kvh * head_dim;
            for (u32 d = 0; d < head_dim; d++) oh[d] += w * vt[d];
        }
    }
}

// Plain malloc'd string copy -- avoids relying on POSIX strdup's
// declaration being visible under strict -std=c99.
static char* dup_str(const char* s) {
    size_t len = strlen(s) + 1;
    char* d = (char*)malloc(len);
    if (d) memcpy(d, s, len);
    return d;
}

// Every architecture's hyperparameter keys live under its own namespace
// ("llama.embedding_length" vs "qwen2.embedding_length" vs
// "gemma.embedding_length", ...) -- builds "<arch>.<suffix>" into `buf`.
static const char* arch_key(char* buf, size_t bufsz, const char* arch, const char* suffix) {
    snprintf(buf, bufsz, "%s.%s", arch, suffix);
    return buf;
}

LLaMAConfig llama_config_from_gguf(GGUFFile* gf, LLaMAConfig defaults, char** arch_out) {
    LLaMAConfig cfg = defaults;
    cfg.rope_type = ROPE_NORM;
    cfg.ffn_act = ACT_SILU;
    cfg.qkv_bias = 0;
    cfg.embedding_scale = 1.0f;

    char* arch = gguf_get_str(gf, "general.architecture");
    if (!arch) arch = dup_str("llama"); // no arch key at all: assume llama, like older files did
    if (arch_out) *arch_out = dup_str(arch);

    // Known per-architecture quirks that aren't in the metadata itself --
    // see model.h's RopeType/FfnAct comments and the llamini-architecture
    // skill for why each of these is real, not guessed.
    if (!strcmp(arch, "qwen2")) {
        cfg.rope_type = ROPE_NEOX;
        cfg.qkv_bias = 1;
    } else if (!strcmp(arch, "gemma")) {
        cfg.rope_type = ROPE_NEOX;
        cfg.ffn_act = ACT_GELU;
    }
    // Anything else (llama, mistral -- ships as arch "llama") keeps the
    // ROPE_NORM/ACT_SILU/no-bias/scale-1.0 defaults set above.

    char buf[64];
    cfg.dim        = gguf_get_u32(gf, arch_key(buf, sizeof(buf), arch, "embedding_length"), cfg.dim);
    cfg.n_layers    = gguf_get_u32(gf, arch_key(buf, sizeof(buf), arch, "block_count"), cfg.n_layers);
    cfg.n_heads      = gguf_get_u32(gf, arch_key(buf, sizeof(buf), arch, "attention.head_count"), cfg.n_heads);
    cfg.n_heads_kv    = gguf_get_u32(gf, arch_key(buf, sizeof(buf), arch, "attention.head_count_kv"), cfg.n_heads);
    cfg.ffn_dim        = gguf_get_u32(gf, arch_key(buf, sizeof(buf), arch, "feed_forward_length"), cfg.ffn_dim);
    cfg.rms_eps          = gguf_get_f32(gf, arch_key(buf, sizeof(buf), arch, "attention.layer_norm_rms_epsilon"), cfg.rms_eps);
    cfg.rope_freq_base    = gguf_get_f32(gf, arch_key(buf, sizeof(buf), arch, "rope.freq_base"), cfg.rope_freq_base);

    // head_dim: prefer the explicit key (required for correctness on
    // architectures where dim/n_heads doesn't hold, e.g. gemma2 -- not
    // supported here, but the derived fallback would silently be wrong for
    // it too, hence reading the real key whenever present).
    u32 derived_head_dim = cfg.n_heads > 0 ? cfg.dim / cfg.n_heads : 0;
    cfg.head_dim = gguf_get_u32(gf, arch_key(buf, sizeof(buf), arch, "attention.key_length"), derived_head_dim);

    if (!strcmp(arch, "gemma")) cfg.embedding_scale = sqrtf((f32)cfg.dim);

    u32 ctx_len = gguf_get_u32(gf, arch_key(buf, sizeof(buf), arch, "context_length"), cfg.seq_len);
    cfg.seq_len = ctx_len < MAX_SEQ_LEN ? ctx_len : MAX_SEQ_LEN;

    u64 vocab_n = gguf_meta_array_len(gf, "tokenizer.ggml.tokens", NULL);
    if (vocab_n > 0) cfg.vocab_size = (u32)vocab_n;

    free(arch);
    return cfg;
}

int llama_model_init(LLaMAModel* model, LLaMAConfig* cfg) {
    memset(model, 0, sizeof(LLaMAModel));
    model->cfg = *cfg;
    model->layers = (DecoderLayer*)calloc(cfg->n_layers, sizeof(DecoderLayer));
    if (!model->layers) return -1;

    u32 dim = cfg->dim, ffn = cfg->ffn_dim;
    u32 head_dim = cfg->head_dim; // from llama_config_from_gguf: explicit key or dim/n_heads
    u32 kv_dim = cfg->n_heads_kv * head_dim;
    u32 q_dim = cfg->n_heads * head_dim; // == dim for every arch supported here, but computed
                                          // properly rather than assumed (see model.h head_dim comment)

    u32 emb_shape[]  = {cfg->vocab_size, dim};
    model->embeddings = tensor_create(2, emb_shape);
    u32 norm_shape[] = {1, dim};
    model->final_norm = tensor_create(2, norm_shape);
    u32 head_shape[] = {cfg->vocab_size, dim};
    model->lm_head = tensor_create(2, head_shape);

    for (u32 l = 0; l < cfg->n_layers; l++) {
        DecoderLayer* layer = &model->layers[l];
        u32 s_norm[]   = {1, dim};
        u32 s_q[]      = {q_dim, dim};
        u32 s_kv[]     = {kv_dim, dim};
        u32 s_o[]      = {dim, q_dim};
        u32 s_gateup[] = {ffn, dim};
        u32 s_down[]   = {dim, ffn};

        layer->attn_norm = tensor_create(2, s_norm);
        layer->q_proj    = tensor_create(2, s_q);
        layer->k_proj    = tensor_create(2, s_kv);
        layer->v_proj    = tensor_create(2, s_kv);
        layer->o_proj    = tensor_create(2, s_o);
        layer->ffn_norm  = tensor_create(2, s_norm);
        layer->gate_proj = tensor_create(2, s_gateup);
        layer->up_proj   = tensor_create(2, s_gateup);
        layer->down_proj = tensor_create(2, s_down);

        if (cfg->qkv_bias) {
            u32 s_qb[] = {1, q_dim};
            u32 s_kvb[] = {1, kv_dim};
            layer->q_bias = tensor_create(2, s_qb);
            layer->k_bias = tensor_create(2, s_kvb);
            layer->v_bias = tensor_create(2, s_kvb);
        }
    }

    return 0;
}

void llama_model_free(LLaMAModel* model) {
    if (!model) return;
    tensor_free(model->embeddings);
    tensor_free(model->final_norm);
    // Tied-embedding models (see llama_load_weights) point lm_head at the
    // same Tensor* as embeddings instead of a separate copy -- only free it
    // once.
    if (model->lm_head != model->embeddings) tensor_free(model->lm_head);
    if (model->layers) {
        for (u32 l = 0; l < model->cfg.n_layers; l++) {
            DecoderLayer* layer = &model->layers[l];
            tensor_free(layer->attn_norm);
            tensor_free(layer->q_proj);
            tensor_free(layer->k_proj);
            tensor_free(layer->v_proj);
            tensor_free(layer->q_bias);
            tensor_free(layer->k_bias);
            tensor_free(layer->v_bias);
            tensor_free(layer->o_proj);
            tensor_free(layer->ffn_norm);
            tensor_free(layer->gate_proj);
            tensor_free(layer->up_proj);
            tensor_free(layer->down_proj);
        }
        free(model->layers);
    }
    memset(model, 0, sizeof(LLaMAModel));
}

// Dequantizes the named tensor directly into `t`'s pre-allocated buffer.
// Returns -1 (without touching `t`) if the tensor is missing, its element
// count doesn't match, or its ggml_type isn't one this reader supports.
static int load_tensor(GGUFFile* gf, const char* name, Tensor* t) {
    const GGUFTensorInfo* info = gguf_find_tensor(gf, name);
    if (!info) return -1;
    return gguf_dequantize_tensor(gf, info, t->data, t->size);
}

int llama_load_weights(LLaMAModel* model, GGUFFile* gf) {
    if (load_tensor(gf, "token_embd.weight", model->embeddings) != 0) return -1;
    if (load_tensor(gf, "output_norm.weight", model->final_norm) != 0) return -1;
    // Some checkpoints (e.g. gemma) tie the output projection to the input
    // embeddings and don't ship output.weight at all -- share the
    // embeddings pointer instead of dequantizing token_embd.weight a
    // second, wasted time into a separate buffer (llama_model_free checks
    // for this same-pointer case before freeing).
    if (gguf_find_tensor(gf, "output.weight")) {
        if (load_tensor(gf, "output.weight", model->lm_head) != 0) return -1;
    } else {
        tensor_free(model->lm_head);
        model->lm_head = model->embeddings;
    }

    char name[64];
    for (u32 l = 0; l < model->cfg.n_layers; l++) {
        DecoderLayer* layer = &model->layers[l];
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
        if (load_tensor(gf, name, layer->attn_norm) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", l);
        if (load_tensor(gf, name, layer->q_proj) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", l);
        if (load_tensor(gf, name, layer->k_proj) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l);
        if (load_tensor(gf, name, layer->v_proj) != 0) return -1;
        if (model->cfg.qkv_bias) {
            snprintf(name, sizeof(name), "blk.%u.attn_q.bias", l);
            if (load_tensor(gf, name, layer->q_bias) != 0) return -1;
            snprintf(name, sizeof(name), "blk.%u.attn_k.bias", l);
            if (load_tensor(gf, name, layer->k_bias) != 0) return -1;
            snprintf(name, sizeof(name), "blk.%u.attn_v.bias", l);
            if (load_tensor(gf, name, layer->v_bias) != 0) return -1;
        }
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", l);
        if (load_tensor(gf, name, layer->o_proj) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
        if (load_tensor(gf, name, layer->ffn_norm) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", l);
        if (load_tensor(gf, name, layer->gate_proj) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", l);
        if (load_tensor(gf, name, layer->up_proj) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", l);
        if (load_tensor(gf, name, layer->down_proj) != 0) return -1;
    }
    return 0;
}
