#include "common.h"
#include "tensor.h"
#include "model.h"
#include "kv_cache.h"
#include "gguf.h"
#include "quant.h"
#include "tokenizer.h"
#include "generate.h"

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

// ==============================================
// Main chat loop: input -> real BPE tokenizer -> full transformer forward
// pass -> greedy decode -> output. Each turn is single-turn generation:
// every layer's KV cache is reset per turn rather than pretending to
// carry multi-turn context (see generate.h).
// ==============================================
void start_chat(LLaMAModel* model, KVCache* caches, Vocab* vocab) {
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

        u32 in_count = text_to_tokens(vocab, user_input, input_tokens, MAX_TURN_TOKENS);

        for (u32 l = 0; l < model->cfg.n_layers; l++) kv_cache_reset(&caches[l]);
        u32 total = generate_autoregressive(model, caches, input_tokens, in_count,
                                             out_tokens, MAX_GEN_TOKENS, vocab->eos_id);

        printf("Bot:");
        for (u32 i = in_count; i < total; i++) {
            if (token_to_text(vocab, out_tokens[i], word, sizeof(word)) > 0)
                printf("%s", word);
        }
        printf("\n\n");
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
        fprintf(stderr, "Usage:\n  %s model.gguf\n  %s --test\n", argv[0], argv[0]);
        return 1;
    }

    GGUFFile gf;
    if (gguf_open(argv[1], &gf) != 0) {
        fprintf(stderr, "GGUF load failed\n");
        return -1;
    }
    printf("Loaded GGUF v%u (%llu tensors, %llu metadata entries)\n",
           gf.hdr.version, (unsigned long long)gf.hdr.n_tensors, (unsigned long long)gf.hdr.n_metadata);

    // TinyLlama-1.1B-shaped defaults, used only for keys the file doesn't
    // define; every value the file does define overrides these.
    LLaMAConfig defaults = {2048, 22, 32, 32, 5632, 32000, MAX_SEQ_LEN, 1e-5f, 10000.0f};
    LLaMAConfig cfg = llama_config_from_gguf(&gf, defaults);
    printf("Config: dim=%u layers=%u heads=%u kv_heads=%u ffn=%u vocab=%u seq_len=%u\n",
           cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_heads_kv, cfg.ffn_dim, cfg.vocab_size, cfg.seq_len);

    LLaMAModel model;
    if (llama_model_init(&model, &cfg) != 0) { gguf_close(&gf); return -1; }

    if (llama_load_weights(&model, &gf) != 0) {
        fprintf(stderr, "warning: could not load model weights from this GGUF file "
                        "(missing tensor, or a quantization type this reader doesn't "
                        "support -- see README limits); running with zero-initialized "
                        "weights\n");
    }

    Vocab vocab;
    int have_vocab = (vocab_load_from_gguf(&gf, &vocab) == 0);
    if (!have_vocab)
        fprintf(stderr, "warning: no tokenizer vocab in this GGUF file; chat disabled\n");

    u32 head_dim = cfg.dim / cfg.n_heads;
    u32 kv_dim = cfg.n_heads_kv * head_dim;
    KVCache* caches = (KVCache*)calloc(cfg.n_layers, sizeof(KVCache));
    for (u32 l = 0; l < cfg.n_layers; l++) kv_cache_init(&caches[l], kv_dim, cfg.seq_len);

    if (have_vocab) start_chat(&model, caches, &vocab);

    for (u32 l = 0; l < cfg.n_layers; l++) { free(caches[l].key); free(caches[l].val); }
    free(caches);
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
    rope(q,5,8,4,10000.0f);
    for(int i=0;i<8;i++) printf("%.2f ",q[i]); printf("\n");
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
    LLaMAConfig cfg = {16, 1, 4, 2, 32, 8, 32, 1e-5f, 10000.0f};
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

    u32 head_dim = cfg.dim / cfg.n_heads;
    u32 kv_dim = cfg.n_heads_kv * head_dim;
    KVCache cache;
    kv_cache_init(&cache, kv_dim, cfg.seq_len);

    u32 in_tokens[4] = {1, 3, 4, 2};
    u32 out_tokens[4 + 4];
    u32 total = generate_autoregressive(&model, &cache, in_tokens, 4, out_tokens, 4, 999);
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
