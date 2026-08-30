#include "common.h"
#include "tensor.h"
#include "model.h"
#include "kv_cache.h"
#include "gguf.h"
#include "quant.h"
#include "tokenizer.h"
#include "generate.h"
#include "gpt2.h"

#define MAX_TURN_TOKENS 128
#define MAX_GEN_TOKENS  48

// Test functions (for teaching core LLM components)
void test_tensor_matmul();
void test_rms_norm();
void test_rope();
void test_kv_cache();
void test_int4_quant();
void test_generate();
void test_gguf_reject();

// Run all unit tests to verify LLM building blocks
void run_all_unit_tests() {
    printf("===== All Unit Tests =====\n");
    test_tensor_matmul();   // Test matrix multiplication
    test_rms_norm();        // Test normalization
    test_rope();            // Test positional encoding
    test_kv_cache();        // Test KV cache memory
    test_int4_quant();      // Test 4-bit quantization
    test_generate();        // Test tokenizer + full per-layer generate loop
    test_gguf_reject();     // Test malformed GGUF files are rejected, not misread
    printf("=========================\n");
}

// TinyLlama-1.1B-Chat-v1.0's own chat_template (Zephyr-style), per the
// model's tokenizer_config.json on Hugging Face:
//   <|system|>\n{system}</s>\n<|user|>\n{message}</s>\n<|assistant|>\n
// `<|system|>`/`<|user|>`/`<|assistant|>` are NOT special vocab tokens for
// this model (stock 32000-piece LLaMA vocab, no added tokens) -- confirmed
// by tokenizing them: they BPE-split into ordinary sub-word pieces just
// like any other text, so plain text_to_tokens handles them correctly.
// `</s>` is different: it names the real EOS control token (id 2, type
// CONTROL), but empirically does NOT BPE-merge back into that single token
// when typed as literal text -- verified by tokenizing the literal string
// and finding zero occurrences of eos_id in the output; every real
// tokenizer's chat-template handling relies on a special-token pre-split
// step this project's tokenizer doesn't implement (see STDLIB.md). So
// build_chat_prompt_tokens inserts the real eos_id token programmatically
// between segments instead of embedding "</s>" as text.
#define SYSTEM_PROMPT "You are a friendly chatbot who always responds concisely."

// text_to_tokens always prepends BOS (see tokenizer.c); only the very first
// segment's BOS is kept, later segments' auto-BOS is dropped so there's
// exactly one at the start of the whole prompt.
static u32 append_segment(const Vocab* vocab, const char* text, int keep_bos,
                           u32* out_tokens, u32 n, u32 max_tokens) {
    u32 tmp[MAX_TURN_TOKENS];
    u32 tn = text_to_tokens(vocab, text, tmp, MAX_TURN_TOKENS);
    for (u32 i = keep_bos ? 0 : 1; i < tn && n < max_tokens; i++) out_tokens[n++] = tmp[i];
    return n;
}

static u32 build_chat_prompt_tokens(const Vocab* vocab, const char* user_input,
                                     u32* out_tokens, u32 max_tokens) {
    char seg[MAX_PROMPT_LEN + 32];
    u32 n = 0;

    snprintf(seg, sizeof(seg), "<|system|>\n" SYSTEM_PROMPT);
    n = append_segment(vocab, seg, 1, out_tokens, n, max_tokens);
    if (n < max_tokens) out_tokens[n++] = vocab->eos_id;

    snprintf(seg, sizeof(seg), "<|user|>\n%s", user_input);
    n = append_segment(vocab, seg, 0, out_tokens, n, max_tokens);
    if (n < max_tokens) out_tokens[n++] = vocab->eos_id;

    n = append_segment(vocab, "<|assistant|>\n", 0, out_tokens, n, max_tokens); // generation prompt, no closing </s>
    return n;
}

// ==============================================
// Main chat loop: input -> chat template -> real BPE tokenizer -> full
// transformer forward pass -> decode -> output. Each turn is single-turn
// generation: every layer's KV cache is reset per turn rather than
// pretending to carry multi-turn context (see generate.h). `temp` <= 0 is
// the original deterministic greedy decode; > 0 samples (see sample_token,
// generate.h). `raw_mode` skips the chat template (see build_chat_prompt_tokens
// above), tokenizing your line directly -- useful as an A/B control, and as
// a fallback if this file's template guess ever needs to be bypassed.
// ==============================================
void start_chat(LLaMAModel* model, KVCache* caches, Vocab* vocab, f32 temp, int raw_mode) {
    char user_input[MAX_PROMPT_LEN];
    char word[MAX_TOKEN_LEN];
    u32 input_tokens[MAX_TURN_TOKENS];
    u32 out_tokens[MAX_TURN_TOKENS + MAX_GEN_TOKENS];

    printf("\n======== TinyLlama GGUF Chat Engine Ready ========\n");
    printf("Real GGUF weights + SentencePiece BPE tokenizer (see README limits).\n\n");

    while (1) {
        printf("You: ");
        if (!fgets(user_input, MAX_PROMPT_LEN, stdin)) break;

        size_t len = strlen(user_input);
        if (len > 0 && user_input[len - 1] == '\n')
            user_input[len - 1] = '\0';

        if (!strcmp(user_input, "exit") || !strcmp(user_input, "quit")) {
            printf("Bot: Bye!\n"); break;
        }

        u32 in_count = raw_mode
            ? text_to_tokens(vocab, user_input, input_tokens, MAX_TURN_TOKENS)
            : build_chat_prompt_tokens(vocab, user_input, input_tokens, MAX_TURN_TOKENS);

        for (u32 l = 0; l < model->cfg.n_layers; l++) kv_cache_reset(&caches[l]);
        u32 total = generate_autoregressive(model, caches, input_tokens, in_count,
                                             out_tokens, MAX_GEN_TOKENS, vocab->eos_id,
                                             temp, 0.9f);

        printf("Bot:");
        for (u32 i = in_count; i < total; i++) {
            if (token_to_text(vocab, out_tokens[i], word, sizeof(word)) > 0)
                printf("%s", word);
        }
        printf("\n\n");
    }
}

// A short, fixed, ordinary English test corpus -- no file I/O, no network,
// no new dependency. Used for a self-contained quantitative correctness
// check (see run_bench): teacher-forced perplexity against this text,
// compared to the mathematically-certain "uniform random over the vocab"
// ceiling, rather than an unverifiable external benchmark number.
static const char* BENCH_CORPUS =
    "The sun rises in the east and sets in the west every day. "
    "Water is essential for all living things to survive. "
    "Many people enjoy reading books during their free time.";

static const char* BENCH_PROMPTS[] = {
    "The capital of France is",
    "Two plus two equals",
    "The sky is the color",
};
#define BENCH_N_PROMPTS 3

// Quantitative correctness check against the real model (see README's
// "Correctness evidence" and STDLIB.md's Package Killer section): a
// teacher-forced perplexity number compared to the random-baseline ceiling,
// plus a few known-fact completions as an informational, human-readable
// spot check (not asserted -- a small model isn't guaranteed to nail every
// fact even when correctly implemented).
void run_bench(LLaMAModel* model, KVCache* caches, Vocab* vocab) {
    printf("\n======== Correctness Benchmark ========\n");
    printf("Test corpus: \"%s\"\n\n", BENCH_CORPUS);

    u32 tokens[256];
    u32 n = text_to_tokens(vocab, BENCH_CORPUS, tokens, 256);

    f32 ppl = compute_perplexity(model, caches, tokens, n);
    f32 ceiling = (f32)model->cfg.vocab_size;
    printf("Teacher-forced perplexity over %u tokens: %.2f\n", n, ppl);
    printf("Random-baseline ceiling (uniform over a %u-token vocab): ~%.0f (%.2f nats/token)\n",
           model->cfg.vocab_size, ceiling, logf(ceiling));
    printf("A broken/randomly-wired forward pass would land near that ceiling;\n"
           "a working language model should land dramatically lower.\n");

    printf("\n-- known-fact completions (greedy, raw prompt, informational not asserted) --\n");
    for (u32 p = 0; p < BENCH_N_PROMPTS; p++) {
        u32 in_tok[MAX_TURN_TOKENS];
        u32 in_n = text_to_tokens(vocab, BENCH_PROMPTS[p], in_tok, MAX_TURN_TOKENS);
        u32 out_tok[MAX_TURN_TOKENS + 8];

        for (u32 l = 0; l < model->cfg.n_layers; l++) kv_cache_reset(&caches[l]);
        // Deterministic greedy (temp=0), not the chat loop's --temp, so
        // --bench's evidence is reproducible run to run.
        u32 total = generate_autoregressive(model, caches, in_tok, in_n, out_tok, 8,
                                             vocab->eos_id, 0.0f, 0.0f);

        printf("\"%s\" ->", BENCH_PROMPTS[p]);
        char word[MAX_TOKEN_LEN];
        for (u32 i = in_n; i < total; i++)
            if (token_to_text(vocab, out_tok[i], word, sizeof(word)) > 0) printf("%s", word);
        printf("\n");
    }
}

// GPT-2 has no chat-tuned variant here and no chat template of its own --
// this is plain raw completion, matching --raw's behavior on the LLaMA
// path (there's no non-raw mode to compare against for this architecture).
void gpt2_chat(GPT2Model* model, KVCache* caches, Vocab* vocab, f32 temp) {
    char user_input[MAX_PROMPT_LEN];
    char word[MAX_TOKEN_LEN];
    u32 input_tokens[MAX_TURN_TOKENS];
    u32 out_tokens[MAX_TURN_TOKENS + MAX_GEN_TOKENS];

    printf("\n======== GPT-2 GGUF Completion Engine Ready ========\n");
    printf("Real GGUF weights + byte-level BPE tokenizer, raw completion (no chat template for this arch).\n\n");

    while (1) {
        printf("You: ");
        if (!fgets(user_input, MAX_PROMPT_LEN, stdin)) break;
        size_t len = strlen(user_input);
        if (len > 0 && user_input[len - 1] == '\n') user_input[len - 1] = '\0';
        if (!strcmp(user_input, "exit") || !strcmp(user_input, "quit")) { printf("Bot: Bye!\n"); break; }

        u32 in_count = text_to_tokens(vocab, user_input, input_tokens, MAX_TURN_TOKENS);
        for (u32 l = 0; l < model->cfg.n_layers; l++) kv_cache_reset(&caches[l]);
        u32 total = gpt2_generate(model, caches, input_tokens, in_count, out_tokens, MAX_GEN_TOKENS,
                                    vocab->eos_id, temp, 0.9f);

        printf("Bot:");
        for (u32 i = in_count; i < total; i++)
            if (token_to_text(vocab, out_tokens[i], word, sizeof(word)) > 0) printf("%s", word);
        printf("\n\n");
    }
}

void gpt2_bench(GPT2Model* model, KVCache* caches, Vocab* vocab) {
    printf("\n======== Correctness Benchmark ========\n");
    printf("Test corpus: \"%s\"\n\n", BENCH_CORPUS);

    u32 tokens[256];
    u32 n = text_to_tokens(vocab, BENCH_CORPUS, tokens, 256);

    f32 ppl = gpt2_compute_perplexity(model, caches, tokens, n);
    f32 ceiling = (f32)model->cfg.vocab_size;
    printf("Teacher-forced perplexity over %u tokens: %.2f\n", n, ppl);
    printf("Random-baseline ceiling (uniform over a %u-token vocab): ~%.0f (%.2f nats/token)\n",
           model->cfg.vocab_size, ceiling, logf(ceiling));
    printf("A broken/randomly-wired forward pass would land near that ceiling;\n"
           "a working language model should land dramatically lower.\n");

    printf("\n-- known-fact completions (greedy, raw prompt, informational not asserted) --\n");
    for (u32 p = 0; p < BENCH_N_PROMPTS; p++) {
        u32 in_tok[MAX_TURN_TOKENS];
        u32 in_n = text_to_tokens(vocab, BENCH_PROMPTS[p], in_tok, MAX_TURN_TOKENS);
        u32 out_tok[MAX_TURN_TOKENS + 8];
        for (u32 l = 0; l < model->cfg.n_layers; l++) kv_cache_reset(&caches[l]);
        u32 total = gpt2_generate(model, caches, in_tok, in_n, out_tok, 8, vocab->eos_id, 0.0f, 0.0f);
        printf("\"%s\" ->", BENCH_PROMPTS[p]);
        char word[MAX_TOKEN_LEN];
        for (u32 i = in_n; i < total; i++)
            if (token_to_text(vocab, out_tok[i], word, sizeof(word)) > 0) printf("%s", word);
        printf("\n");
    }
}

// ==============================================
// Main function: parse a real GGUF file end to end (metadata-driven
// config, real weights, real vocab) and start chatting.
// ==============================================
int main(int argc, char** argv) {
    if (argc >= 2 && !strcmp(argv[1], "--test")) {
        run_all_unit_tests();
        return 0;
    }

    if (argc < 2) {
        fprintf(stderr, "Usage:\n  %s model.gguf [--temp X] [--raw]\n  %s model.gguf --bench\n  %s --test\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }
    int bench_mode = 0;
    int raw_mode = 0;
    f32 temp = 0.0f;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) bench_mode = 1;
        else if (!strcmp(argv[i], "--raw")) raw_mode = 1;
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = (f32)atof(argv[++i]);
    }
    if (temp > 0.0f) srand((unsigned)time(NULL));

    GGUFFile gf;
    if (gguf_open(argv[1], &gf) != 0) {
        fprintf(stderr, "GGUF load failed\n");
        return -1;
    }
    printf("Loaded GGUF v%u (%llu tensors, %llu metadata entries)\n",
           gf.hdr.version, (unsigned long long)gf.hdr.n_tensors, (unsigned long long)gf.hdr.n_metadata);

    Vocab vocab;
    int have_vocab = (vocab_load_from_gguf(&gf, &vocab) == 0);
    if (!have_vocab)
        fprintf(stderr, "warning: no tokenizer vocab in this GGUF file; chat disabled\n");

    // GPT-2 is architecturally separate enough (fused QKV, LayerNorm,
    // learned position embeddings, ungated GELU, no RoPE) to be a wholly
    // different model type (gpt2.c), not another LLaMAConfig flag --
    // peek the architecture string before deciding which path to take.
    char* peek_arch = gguf_get_str(&gf, "general.architecture");
    int is_gpt2 = peek_arch && !strcmp(peek_arch, "gpt2");
    free(peek_arch);

    if (is_gpt2) {
        GPT2Config cfg = gpt2_config_from_gguf(&gf);
        printf("Config: arch=gpt2 dim=%u layers=%u heads=%u ffn=%u vocab=%u n_ctx=%u\n",
               cfg.dim, cfg.n_layers, cfg.n_heads, cfg.ffn_dim, cfg.vocab_size, cfg.n_ctx);

        GPT2Model model;
        if (gpt2_model_init(&model, &cfg) != 0) { gguf_close(&gf); return -1; }
        if (gpt2_load_weights(&model, &gf) != 0)
            fprintf(stderr, "warning: could not load model weights from this GGUF file "
                            "(missing tensor, or a quantization type this reader doesn't "
                            "support -- see README limits); running with zero-initialized "
                            "weights\n");

        KVCache* caches = (KVCache*)calloc(cfg.n_layers, sizeof(KVCache));
        for (u32 l = 0; l < cfg.n_layers; l++) kv_cache_init(&caches[l], cfg.dim, cfg.n_ctx);

        if (have_vocab && bench_mode) gpt2_bench(&model, caches, &vocab);
        else if (have_vocab) gpt2_chat(&model, caches, &vocab, temp);

        for (u32 l = 0; l < cfg.n_layers; l++) { free(caches[l].key); free(caches[l].val); }
        free(caches);
        if (have_vocab) vocab_free(&vocab);
        gguf_close(&gf);
        gpt2_model_free(&model);
        return 0;
    }

    // TinyLlama-1.1B-shaped defaults, used only for keys the file doesn't
    // define; every value the file does define overrides these. Designated
    // initializers (not positional) so adding a field to LLaMAConfig can't
    // silently misalign this list.
    LLaMAConfig defaults = {
        .dim = 2048, .n_layers = 22, .n_heads = 32, .n_heads_kv = 32,
        .ffn_dim = 5632, .vocab_size = 32000, .seq_len = MAX_SEQ_LEN,
        .rms_eps = 1e-5f, .rope_freq_base = 10000.0f,
    };
    char* arch = NULL;
    LLaMAConfig cfg = llama_config_from_gguf(&gf, defaults, &arch);
    printf("Config: arch=%s dim=%u layers=%u heads=%u kv_heads=%u head_dim=%u ffn=%u vocab=%u seq_len=%u\n",
           arch ? arch : "?", cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_heads_kv,
           cfg.head_dim, cfg.ffn_dim, cfg.vocab_size, cfg.seq_len);

    LLaMAModel model;
    if (llama_model_init(&model, &cfg) != 0) { gguf_close(&gf); return -1; }

    if (llama_load_weights(&model, &gf) != 0) {
        fprintf(stderr, "warning: could not load model weights from this GGUF file "
                        "(missing tensor, or a quantization type this reader doesn't "
                        "support -- see README limits); running with zero-initialized "
                        "weights\n");
    }

    u32 kv_dim = cfg.n_heads_kv * cfg.head_dim;
    KVCache* caches = (KVCache*)calloc(cfg.n_layers, sizeof(KVCache));
    for (u32 l = 0; l < cfg.n_layers; l++) kv_cache_init(&caches[l], kv_dim, cfg.seq_len);

    if (have_vocab && bench_mode) run_bench(&model, caches, &vocab);
    else if (have_vocab) start_chat(&model, caches, &vocab, temp, raw_mode);

    for (u32 l = 0; l < cfg.n_layers; l++) { free(caches[l].key); free(caches[l].val); }
    free(caches);
    free(arch);
    if (have_vocab) vocab_free(&vocab);
    gguf_close(&gf);
    llama_model_free(&model);
    return 0;
}

// ------------------------------
// Test implementations
// ------------------------------
void test_tensor_matmul() {
    printf("\n[MatMul Test]\n");
    u32 s[] = {2,2};
    Tensor *A = tensor_create(2,s), *B = tensor_create(2,s), *C = tensor_create(2,s);
    A->data[0]=1;A->data[1]=2;A->data[2]=3;A->data[3]=4;
    B->data[0]=5;B->data[1]=6;B->data[2]=7;B->data[3]=8;
    matmul(A,B,C);
    printf("%.2f %.2f\n%.2f %.2f\n", C->data[0],C->data[1],C->data[2],C->data[3]);
    tensor_free(A);tensor_free(B);tensor_free(C);
}

void test_rms_norm() {
    printf("\n[RMSNorm Test]\n");
    f32 x[]={1,2,3,4},w[]={1,1,1,1},o[4];
    rms_norm(o,x,w,4,1e-8f);
    for(int i=0;i<4;i++) printf("%.4f ",o[i]); printf("\n");
}

void test_rope() {
    printf("\n[RoPE Test]\n");
    f32 q[]={1,1,1,1,1,1,1,1};
    rope(q,5,8,4,10000.0f,ROPE_NORM);
    for(int i=0;i<8;i++) printf("%.2f ",q[i]); printf("\n");
    f32 q2[]={1,1,1,1,1,1,1,1};
    rope(q2,5,8,4,10000.0f,ROPE_NEOX);
    for(int i=0;i<8;i++) printf("%.2f ",q2[i]); printf("\n");
}

void test_kv_cache() {
    printf("\n[KV Cache Test]\n");
    KVCache c; kv_cache_init(&c,512,MAX_SEQ_LEN);
    c.cur_seq=10;
    printf("seq=%u\n",c.cur_seq);
    kv_cache_reset(&c);
    printf("reset=%u\n",c.cur_seq);
    free(c.key); free(c.val);
}

void test_int4_quant() {
    printf("\n[INT4 Quant Test]\n");
    f32 s[]={1,2,3,4,5,6,7,8},o[8],sc,zp;
    u8 d[4];
    quant_int4(s,d,8,&sc,&zp);
    dequant_int4(o,d,8,sc,zp);
    for(int i=0;i<8;i++) printf("%.2f | %.2f\n",s[i],o[i]);
}

void test_generate() {
    printf("\n[Tokenizer + Generate Test]\n");
    // Tiny synthetic model (not loaded from a file) exercising the full
    // per-layer forward pass, including a non-trivial GQA shape
    // (4 query heads sharing 2 KV heads), without needing a real GGUF.
    LLaMAConfig cfg = {
        .dim = 16, .n_layers = 1, .n_heads = 4, .n_heads_kv = 2, .head_dim = 4,
        .ffn_dim = 32, .vocab_size = 8, .seq_len = 32,
        .rms_eps = 1e-5f, .rope_freq_base = 10000.0f,
        .rope_type = ROPE_NORM, .ffn_act = ACT_SILU, .qkv_bias = 0, .embedding_scale = 1.0f,
    };
    LLaMAModel model;
    assert(llama_model_init(&model, &cfg) == 0);

    for (u32 i = 0; i < model.embeddings->size; i++) model.embeddings->data[i] = (f32)((i % 7) - 3) * 0.1f;
    for (u32 i = 0; i < model.final_norm->size; i++) model.final_norm->data[i] = 1.0f;
    for (u32 i = 0; i < model.lm_head->size; i++) model.lm_head->data[i] = (f32)((i % 5) - 2) * 0.1f;
    DecoderLayer* layer = &model.layers[0];
    for (u32 i = 0; i < layer->attn_norm->size; i++) layer->attn_norm->data[i] = 1.0f;
    for (u32 i = 0; i < layer->ffn_norm->size; i++) layer->ffn_norm->data[i] = 1.0f;
    for (u32 i = 0; i < layer->q_proj->size; i++) layer->q_proj->data[i] = (f32)((i % 5) - 2) * 0.05f;
    for (u32 i = 0; i < layer->k_proj->size; i++) layer->k_proj->data[i] = (f32)((i % 5) - 2) * 0.05f;
    for (u32 i = 0; i < layer->v_proj->size; i++) layer->v_proj->data[i] = (f32)((i % 5) - 2) * 0.05f;
    for (u32 i = 0; i < layer->o_proj->size; i++) layer->o_proj->data[i] = (f32)((i % 5) - 2) * 0.05f;
    for (u32 i = 0; i < layer->gate_proj->size; i++) layer->gate_proj->data[i] = (f32)((i % 5) - 2) * 0.05f;
    for (u32 i = 0; i < layer->up_proj->size; i++) layer->up_proj->data[i] = (f32)((i % 5) - 2) * 0.05f;
    for (u32 i = 0; i < layer->down_proj->size; i++) layer->down_proj->data[i] = (f32)((i % 5) - 2) * 0.05f;

    u32 kv_dim = cfg.n_heads_kv * cfg.head_dim;
    KVCache cache;
    kv_cache_init(&cache, kv_dim, cfg.seq_len);

    u32 in_tokens[4] = {1, 3, 4, 2};
    u32 out_tokens[4 + 4];
    u32 total = generate_autoregressive(&model, &cache, in_tokens, 4, out_tokens, 4, 999, 0.0f, 0.0f);
    assert(total >= 4 && total <= 8);
    for (u32 i = 0; i < total; i++) {
        assert(out_tokens[i] < cfg.vocab_size);
        printf("%u ", out_tokens[i]);
    }
    printf("\n");

    free(cache.key); free(cache.val);
    llama_model_free(&model);
}

static void write_test_gguf(const char* path, const u8* data, size_t len) {
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(data, 1, len, f); fclose(f); }
}

// A user-supplied .gguf is untrusted input -- these hostile files must make
// gguf_open() fail cleanly (-1), not attempt a pathological allocation or
// let an overflowed length survive into a later read.
void test_gguf_reject() {
    printf("\n[GGUF Reject Test]\n");
    GGUFFile gf;

    // 1. Bad magic.
    {
        u8 buf[24] = {0};
        buf[0] = 'X'; buf[1] = 'X'; buf[2] = 'X'; buf[3] = 'X';
        write_test_gguf("test_bad_magic.gguf", buf, sizeof(buf));
        assert(gguf_open("test_bad_magic.gguf", &gf) == -1);
        unlink("test_bad_magic.gguf");
    }

    // 2. Absurd tensor count (must exceed GGUF_MAX_TENSORS / the self-scaling bound).
    {
        u8 buf[24];
        u32 magic = GGUF_MAGIC, version = 3;
        u64 n_tensors = 0x0000FFFFFFFFFFFFULL, n_metadata = 0;
        memcpy(buf, &magic, 4);
        memcpy(buf + 4, &version, 4);
        memcpy(buf + 8, &n_tensors, 8);
        memcpy(buf + 16, &n_metadata, 8);
        write_test_gguf("test_huge_tensors.gguf", buf, sizeof(buf));
        assert(gguf_open("test_huge_tensors.gguf", &gf) == -1);
        unlink("test_huge_tensors.gguf");
    }

    // 3. A metadata array whose length would overflow the old
    //    "esz * arr_len" byte-count multiply (esz=4, arr_len=1<<62 wraps to
    //    0, so the old code would have accepted this file and handed a
    //    fabricated 2^62-entry array to a caller). This is the regression
    //    case: it must fail against the new GGUF_MAX_ARRAY cap.
    {
        u8 buf[64];
        u8* p = buf;
        u32 magic = GGUF_MAGIC, version = 3;
        u64 n_tensors = 0, n_metadata = 1;
        memcpy(p, &magic, 4); p += 4;
        memcpy(p, &version, 4); p += 4;
        memcpy(p, &n_tensors, 8); p += 8;
        memcpy(p, &n_metadata, 8); p += 8;
        u64 keylen = 1;
        memcpy(p, &keylen, 8); p += 8;
        *p++ = 'x';
        u32 type = GGUF_TYPE_ARRAY;
        memcpy(p, &type, 4); p += 4;
        u32 elem_type = GGUF_TYPE_FLOAT32;
        memcpy(p, &elem_type, 4); p += 4;
        u64 arr_len = 1ull << 62;
        memcpy(p, &arr_len, 8); p += 8;
        write_test_gguf("test_array_overflow.gguf", buf, (size_t)(p - buf));
        assert(gguf_open("test_array_overflow.gguf", &gf) == -1);
        unlink("test_array_overflow.gguf");
    }

    printf("all malformed files correctly rejected\n");
}
