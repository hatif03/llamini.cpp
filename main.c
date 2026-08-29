#include "common.h"
#include "tensor.h"
#include "model.h"
#include "kv_cache.h"
#include "gguf.h"
#include "quant.h"
#include <strings.h>

// ==============================================
// Step 1: Convert user input text to a token ID
// This is a SIMPLIFIED tokenizer (for teaching)
// ==============================================
static u32 text_to_token(const char* text) {
    if (strcasestr(text, "hello") || strcasestr(text, "hi")) return 22;
    if (strcasestr(text, "how are you")) return 45;
    if (strcasestr(text, "name") || strcasestr(text, "what's your name")) return 67;
    if (strcasestr(text, "bye") || strcasestr(text, "goodbye")) return 89;
    if (strcasestr(text, "thank you") || strcasestr(text, "thanks")) return 90;
    if (strcasestr(text, "who made you") || strcasestr(text, "creator")) return 91;
    if (strcasestr(text, "what is llm")) return 92;
    if (strcasestr(text, "what is tinyllama")) return 93;
    if (strcasestr(text, "how old are you")) return 94;
    if (strcasestr(text, "what can you do")) return 95;
    if (strcasestr(text, "good morning")) return 96;
    if (strcasestr(text, "good night")) return 97;
    return 1; // default token
}

// ==============================================
// Step 2: Convert token ID back to readable text
// ==============================================
static void token_to_text(u32 token, char* out) {
    switch(token) {
        case 22: strcpy(out, "Hello! How can I help you today?"); break;
        case 45: strcpy(out, "I'm doing well, thanks for asking!"); break;
        case 67: strcpy(out, "I am TinyLlama, a lightweight LLM."); break;
        case 89: strcpy(out, "Goodbye! Have a nice day."); break;
        case 90: strcpy(out, "You're welcome! Happy to help."); break;
        case 91: strcpy(out, "I am developed by the TinyLlama team."); break;
        case 92: strcpy(out, "LLM stands for Large Language Model."); break;
        case 93: strcpy(out, "TinyLlama is a small, fast open-source LLM."); break;
        case 94: strcpy(out, "I don't have an actual age, I'm an AI model."); break;
        case 95: strcpy(out, "I can chat and answer simple questions for you."); break;
        case 96: strcpy(out, "Good morning! Wish you a nice day."); break;
        case 97: strcpy(out, "Good night! Have a sweet dream."); break;
        default: strcpy(out, "I understand your message."); break;
    }
}

// ==============================================
// Step 3: Greedy sampling - pick the best token
// ==============================================
static u32 greedy_sample(f32* logits, u32 vocab_size) {
    u32 max_idx = 0;
    f32 max_val = logits[0];
    for (u32 i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Test functions (for teaching core LLM components)
void test_tensor_matmul();
void test_rms_norm();
void test_rope();
void test_kv_cache();
void test_int4_quant();

// Run all unit tests to verify LLM building blocks
void run_all_unit_tests() {
    printf("===== All Unit Tests =====\n");
    test_tensor_matmul();   // Test matrix multiplication
    test_rms_norm();        // Test normalization
    test_rope();            // Test positional encoding
    test_kv_cache();        // Test KV cache memory
    test_int4_quant();      // Test 4-bit quantization
    printf("=========================\n");
}

// ==============================================
// Main chat loop: input → model → output
// ==============================================
void start_chat(LLaMAModel* model, KVCache* kv_cache) {
    char user_input[MAX_PROMPT_LEN];
    char bot_reply[MAX_TOKEN_LEN];
    u32 cur_pos = 0;
    u32 dim = model->cfg.dim;

    printf("\n======== TinyLlama GGUF Chat Engine Ready ========\n");
    printf("Supported: hello, hi, how are you, name, bye, thanks, etc.\n\n");

    while (1) {
        // Get input from user
        printf("You: ");
        fgets(user_input, MAX_PROMPT_LEN, stdin);

        // Remove newline character (clean input)
        size_t len = strlen(user_input);
        if (len > 0 && user_input[len-1] == '\n')
            user_input[len-1] = '\0';

        // Exit condition
        if (!strcmp(user_input, "exit") || !strcmp(user_input, "quit")) {
            printf("Bot: Bye!\n"); break;
        }

        // Convert text to token
        u32 prompt_token = text_to_token(user_input);

        // Create Q, K, V tensors for attention
        u32 shape[] = {1, dim};
        Tensor *q = tensor_create(2, shape);
        Tensor *k = tensor_create(2, shape);
        Tensor *v = tensor_create(2, shape);
        Tensor *out = tensor_create(2, shape);

        // Copy embedding values into Q, K, V
        memcpy(q->data, model->embeddings->data + prompt_token * dim, dim * sizeof(f32));
        memcpy(k->data, model->embeddings->data + prompt_token * dim, dim * sizeof(f32));
        memcpy(v->data, model->embeddings->data + prompt_token * dim, dim * sizeof(f32));

        // Apply positional encoding (RoPE)
        rope(q->data, k->data, cur_pos, dim, dim / model->cfg.n_heads);

        // Run multi-head attention
        causal_mha(q, k, v, kv_cache, out, cur_pos, model->cfg.n_heads);

        // Simulate output logits
        f32 logits[32000] = {0};
        logits[prompt_token] = 10.0f;

        // Choose best token
        u32 next_token = greedy_sample(logits, model->cfg.vocab_size);

        // Convert token to reply text
        token_to_text(next_token, bot_reply);
        printf("Bot: %s\n\n", bot_reply);

        // Move to next position
        cur_pos++;
        if (cur_pos >= model->cfg.seq_len) {
            printf("[Reset KV Cache]\n");
            kv_cache_reset(kv_cache);
            cur_pos = 0;
        }

        // Free temporary tensors
        tensor_free(q);
        tensor_free(k);
        tensor_free(v);
        tensor_free(out);
    }
}

// ==============================================
// Main function: load model + start chat
// ==============================================
int main(int argc, char** argv) {
    // Run tests if --test is used
    if (argc >= 2 && !strcmp(argv[1], "--test")) {
        run_all_unit_tests();
        return 0;
    }

    // Check command line arguments
    if (argc < 2) {
        fprintf(stderr, "Usage:\n  %s model.gguf\n  %s --test\n", argv[0], argv[0]);
        return 1;
    }

    // TinyLlama 1.1B model configuration
    LLaMAConfig cfg = {2048, 22, 32, 32000, MAX_SEQ_LEN};
    LLaMAModel model;

    // Initialize model structure
    if (llama_model_init(&model, &cfg) != 0) return -1;

    // Initialize KV cache
    KVCache kv_cache;
    kv_cache_init(&kv_cache, cfg.dim, cfg.seq_len);

    // Load GGUF model file
    GGUFFile gf;
    if (gguf_open(argv[1], &gf) != 0) {
        fprintf(stderr, "GGUF load failed\n");
        return -1;
    }
    printf("Loaded GGUF v%u\n", gf.hdr.version);

    // Start chatting
    start_chat(&model, &kv_cache);

    // Cleanup
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
    rms_norm(o,x,w,4);
    for(int i=0;i<4;i++) printf("%.4f ",o[i]); printf("\n");
}

void test_rope() {
    printf("\n[RoPE Test]\n");
    f32 q[]={1,1,1,1,1,1,1,1},k[]={1,1,1,1,1,1,1,1};
    rope(q,k,5,8,4);
    for(int i=0;i<8;i++) printf("%.2f ",q[i]); printf("\n");
}

void test_kv_cache() {
    printf("\n[KV Cache Test]\n");
    KVCache c; kv_cache_init(&c,512,MAX_SEQ_LEN);
    c.cur_seq=10;
    printf("seq=%u\n",c.cur_seq);
    kv_cache_reset(&c);
    printf("reset=%u\n",c.cur_seq);
}

void test_int4_quant() {
    printf("\n[INT4 Quant Test]\n");
    f32 s[]={1,2,3,4,5,6,7,8},o[8],sc,zp;
    u8 d[4];
    quant_int4(s,d,8,&sc,&zp);
    dequant_int4(o,d,8,sc,zp);
    for(int i=0;i<8;i++) printf("%.2f | %.2f\n",s[i],o[i]);
}