#include "generate.h"

u32 greedy_sample(const f32* logits, u32 vocab_size) {
    u32 max_idx = 0;
    f32 max_val = logits[0];
    for (u32 i = 1; i < vocab_size; i++)
        if (logits[i] > max_val) { max_val = logits[i]; max_idx = i; }
    return max_idx;
}

typedef struct { f32 p; u32 id; } PIdx;

// Descending by probability -- qsort wants -1/0/1, not a bool.
static int pidx_cmp(const void* a, const void* b) {
    f32 pa = ((const PIdx*)a)->p, pb = ((const PIdx*)b)->p;
    return (pa < pb) - (pa > pb);
}

// temp <= 0 dispatches straight to greedy_sample (bit-identical, so the
// already-verified deterministic path never changes just because sampling
// exists). Otherwise: numerically-stable softmax (subtract max before expf,
// same pattern causal_mha/compute_perplexity already use), sort by
// probability, keep the smallest nucleus whose mass reaches top_p, and draw
// from it.
u32 sample_token(const f32* logits, u32 vocab_size, f32 temp, f32 top_p) {
    if (temp <= 0.0f) return greedy_sample(logits, vocab_size);

    PIdx* c = (PIdx*)malloc((size_t)vocab_size * sizeof(PIdx));
    if (!c) return greedy_sample(logits, vocab_size); // degrade, never crash

    f32 mx = logits[0];
    for (u32 i = 1; i < vocab_size; i++) if (logits[i] > mx) mx = logits[i];

    f32 sum = 0.0f;
    for (u32 i = 0; i < vocab_size; i++) {
        f32 p = expf((logits[i] - mx) / temp);
        c[i].p = p; c[i].id = i;
        sum += p;
    }
    qsort(c, vocab_size, sizeof(PIdx), pidx_cmp);

    f32 cum = 0.0f;
    u32 keep = 0;
    while (keep < vocab_size) {
        cum += c[keep].p;
        keep++;
        if (cum >= top_p * sum) break;
    }

    f32 r = ((f32)rand() / ((f32)RAND_MAX + 1.0f)) * cum;
    f32 acc = 0.0f;
    u32 pick = c[0].id;
    for (u32 i = 0; i < keep; i++) {
        acc += c[i].p;
        if (r < acc) { pick = c[i].id; break; }
    }

    free(c);
    return pick;
}

void forward_step(LLaMAModel* model, KVCache* caches, u32 pos, u32 token_id, f32* logits_out) {
    u32 dim = model->cfg.dim;
    u32 ffn = model->cfg.ffn_dim;
    u32 n_heads = model->cfg.n_heads;
    u32 n_heads_kv = model->cfg.n_heads_kv;
    u32 head_dim = dim / n_heads;
    u32 kv_dim = n_heads_kv * head_dim;

    f32* hidden    = (f32*)malloc(dim * sizeof(f32));
    f32* xnorm     = (f32*)malloc(dim * sizeof(f32));
    f32* q         = (f32*)malloc(dim * sizeof(f32));
    f32* k         = (f32*)malloc(kv_dim * sizeof(f32));
    f32* v         = (f32*)malloc(kv_dim * sizeof(f32));
    f32* attn_out  = (f32*)malloc(dim * sizeof(f32));
    f32* attn_proj = (f32*)malloc(dim * sizeof(f32));
    f32* gate      = (f32*)malloc(ffn * sizeof(f32));
    f32* up        = (f32*)malloc(ffn * sizeof(f32));
    f32* ffn_hid   = (f32*)malloc(ffn * sizeof(f32));
    f32* ffn_out   = (f32*)malloc(dim * sizeof(f32));

    memcpy(hidden, model->embeddings->data + (u64)token_id * dim, dim * sizeof(f32));

    for (u32 l = 0; l < model->cfg.n_layers; l++) {
        DecoderLayer* layer = &model->layers[l];

        rms_norm(xnorm, hidden, layer->attn_norm->data, dim, model->cfg.rms_eps);
        linear(q, xnorm, layer->q_proj->data, dim, dim);
        linear(k, xnorm, layer->k_proj->data, dim, kv_dim);
        linear(v, xnorm, layer->v_proj->data, dim, kv_dim);

        rope(q, pos, dim, head_dim, model->cfg.rope_freq_base);
        rope(k, pos, kv_dim, head_dim, model->cfg.rope_freq_base);

        causal_mha(q, k, v, &caches[l], attn_out, pos, n_heads, n_heads_kv, head_dim);
        linear(attn_proj, attn_out, layer->o_proj->data, dim, dim);
        for (u32 d = 0; d < dim; d++) hidden[d] += attn_proj[d];

        rms_norm(xnorm, hidden, layer->ffn_norm->data, dim, model->cfg.rms_eps);
        linear(gate, xnorm, layer->gate_proj->data, dim, ffn);
        linear(up,   xnorm, layer->up_proj->data,   dim, ffn);
        swiglu(ffn_hid, gate, up, ffn);
        linear(ffn_out, ffn_hid, layer->down_proj->data, ffn, dim);
        for (u32 d = 0; d < dim; d++) hidden[d] += ffn_out[d];
    }

    rms_norm(xnorm, hidden, model->final_norm->data, dim, model->cfg.rms_eps);
    linear(logits_out, xnorm, model->lm_head->data, dim, model->cfg.vocab_size);

    free(hidden); free(xnorm); free(q); free(k); free(v);
    free(attn_out); free(attn_proj); free(gate); free(up); free(ffn_hid); free(ffn_out);
}

u32 generate_autoregressive(LLaMAModel* model, KVCache* caches,
    u32* input_tokens, u32 in_token_count,
    u32* out_tokens, u32 max_gen_tokens, u32 eos_id, f32 temp, f32 top_p)
{
    u32 total_tokens = in_token_count;
    memcpy(out_tokens, input_tokens, in_token_count * sizeof(u32));

    f32* logits = (f32*)malloc((u64)model->cfg.vocab_size * sizeof(f32));

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Prefill every prompt position through the full layer stack first (so
    // each position's KV cache slot is populated by its own real embedding,
    // not left zeroed) before sampling/appending anything. Processing the
    // last prompt position (pos == in_token_count - 1) also yields the
    // first generated token's logits -- that step does double duty, which
    // is exactly why the three stopping conditions below are each checked
    // explicitly rather than folded into one precomputed iteration count.
    u32 pos = 0;
    u32 max_pos = caches[0].max_seq;
    for (;;) {
        if (pos >= max_pos) break;
        u32 current_pos = pos;
        u32 cur_tok = out_tokens[current_pos];
        forward_step(model, caches, current_pos, cur_tok, logits);

        pos++;
        if (pos < in_token_count) continue; // still prefilling: no sample, no append

        u32 next_tok = sample_token(logits, model->cfg.vocab_size, temp, top_p);
        out_tokens[total_tokens++] = next_tok;
        if (next_tok == eos_id) break;
        if (total_tokens - in_token_count >= max_gen_tokens) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    u32 n_generated = total_tokens - in_token_count;
    double elapsed = (double)(t_end.tv_sec - t_start.tv_sec) +
                      (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    if (n_generated > 0 && elapsed > 0.0)
        fprintf(stderr, "[%u tokens in %.2fs, %.2f tok/s]\n", n_generated, elapsed, n_generated / elapsed);

    free(logits);
    return total_tokens;
}

f32 compute_perplexity(LLaMAModel* model, KVCache* caches, const u32* tokens, u32 n_tokens) {
    for (u32 l = 0; l < model->cfg.n_layers; l++) kv_cache_reset(&caches[l]);

    u32 vocab = model->cfg.vocab_size;
    f32* logits = (f32*)malloc((u64)vocab * sizeof(f32));
    u32 max_pos = caches[0].max_seq;

    double nll_sum = 0.0;
    u32 count = 0;
    for (u32 pos = 0; pos + 1 < n_tokens && pos < max_pos; pos++) {
        forward_step(model, caches, pos, tokens[pos], logits);

        // Numerically-stable log-softmax probability of the *actual* next
        // token (teacher forcing -- never sampling), same subtract-max
        // pattern causal_mha already uses.
        f32 mx = logits[0];
        for (u32 i = 1; i < vocab; i++) if (logits[i] > mx) mx = logits[i];
        f32 sum_exp = 0.0f;
        for (u32 i = 0; i < vocab; i++) sum_exp += expf(logits[i] - mx);
        f32 log_prob = (logits[tokens[pos + 1]] - mx) - logf(sum_exp);

        nll_sum += -(double)log_prob;
        count++;
    }

    free(logits);
    if (count == 0) return -1.0f; // corpus too short to evaluate
    return expf((f32)(nll_sum / count));
}
