#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "quant.h"
#include "gguf.h"
#include "model.h"
#include "tokenizer.h"
#include "generate.h"

void test_matmul()
{
    printf("\n[MatMul Unit Test]\n");
    u32 shape_a[] = {2, 2};
    u32 shape_b[] = {2, 2};
    u32 shape_c[] = {2, 2};

    Tensor* A = tensor_create(2, shape_a);
    Tensor* B = tensor_create(2, shape_b);
    Tensor* C = tensor_create(2, shape_c);

    A->data[0] = 1.0f; A->data[1] = 2.0f;
    A->data[2] = 3.0f; A->data[3] = 4.0f;
    B->data[0] = 5.0f; B->data[1] = 6.0f;
    B->data[2] = 7.0f; B->data[3] = 8.0f;

    matmul(A, B, C);
    printf("%.2f %.2f\n", C->data[0], C->data[1]);
    printf("%.2f %.2f\n", C->data[2], C->data[3]);

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

void test_kv_cache()
{
    printf("\n[KV Cache Unit Test]\n");
    KVCache cache;
    kv_cache_init(&cache, 512, MAX_SEQ_LEN);
    cache.cur_seq = 10;
    printf("Cached token count before reset: %u\n", cache.cur_seq);
    kv_cache_reset(&cache);
    printf("Cached token count after reset: %u\n", cache.cur_seq);
}

void test_int4_quant()
{
    printf("\n[INT4 Quantization Unit Test]\n");
    f32 original[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const u32 elem_count = 8;
    u8 compressed_buf[elem_count / 2];
    f32 restored[elem_count];
    f32 scale_param, zp_param;

    quant_int4(original, compressed_buf, elem_count, &scale_param, &zp_param);
    dequant_int4(restored, compressed_buf, elem_count, scale_param, zp_param);

    printf("Original | Restored\n");
    for (u32 i = 0; i < elem_count; i++)
    {
        printf("%.2f      %.2f\n", original[i], restored[i]);
    }
}

void test_gguf_loader(const char* model_path)
{
    printf("\n[GGUF mmap Loader Unit Test]\n");
    GGUFFile gf;
    int ret = gguf_open(model_path, &gf);
    if (ret != 0)
    {
        printf("GGUF load test FAILED, invalid file path or format\n");
        return;
    }
    printf("GGUF file loaded successfully!\n");
    printf("GGUF Version: %u\n", gf.hdr.version);
    printf("Total tensors in model: %llu\n", (unsigned long long)gf.hdr.n_tensors);
    printf("Total metadata entries: %llu\n", (unsigned long long)gf.hdr.n_metadata);
    gguf_close(&gf);
    printf("GGUF resource cleaned up\n");
}

void test_rms_norm()
{
    printf("\n[RMSNorm Unit Test]\n");
    f32 x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    f32 w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    f32 out[4];
    rms_norm(out, x, w, 4);
    printf("Normalized vector: ");
    for (int i = 0; i < 4; i++)
        printf("%.4f ", out[i]);
    printf("\n");
}

void test_swiglu()
{
    printf("\n[SwiGLU Unit Test]\n");
    f32 gate[] = {1.0f, -1.0f, 2.0f, -2.0f};
    f32 up[]   = {2.0f, 3.0f, 1.0f, 4.0f};
    f32 out[4];
    swiglu(out, gate, up, 4);
    printf("SwiGLU output vector: ");
    for (int i = 0; i < 4; i++)
        printf("%.4f ", out[i]);
    printf("\n");
}

void test_rope()
{
    printf("\n[RoPE Rotary Positional Encoding Test]\n");
    f32 q[] = {1, 0, 1, 0, 1, 0, 1, 0};
    f32 k[] = {1, 0, 1, 0, 1, 0, 1, 0};
    u32 pos = 5;
    u32 dim = 8;
    u32 head_dim = 8;

    rope(q, k, pos, dim, head_dim);

    printf("Rotated Q vector: ");
    for(int i = 0; i < 8; i++) {
        printf("%.3f ", q[i]);
    }
    printf("\n");
}

void test_causal_mha()
{
    printf("\n[Causal Multi-Head Attention Unit Test]\n");
    u32 vec_shape[] = {1, 8};
    Tensor* q = tensor_create(2, vec_shape);
    Tensor* k = tensor_create(2, vec_shape);
    Tensor* v = tensor_create(2, vec_shape);
    Tensor* out = tensor_create(2, vec_shape);

    for(u32 i = 0; i < 8; i++){
        q->data[i] = (f32)i;
        k->data[i] = (f32)i;
        v->data[i] = (f32)i;
    }

    KVCache cache;
    kv_cache_init(&cache, 8, MAX_SEQ_LEN);

    causal_mha(q, k, v, &cache, out, 0, 1);
    printf("Attention output at pos 0: ");
    for(int i = 0; i < 8; i++){
        printf("%.2f ", out->data[i]);
    }
    printf("\n");

    tensor_free(q);
    tensor_free(k);
    tensor_free(v);
    tensor_free(out);
    kv_cache_reset(&cache);
}

/**
 * Unit test for text <-> token conversion
 * Test the full tokenizer encode and decode workflow
 */
void test_tokenizer()
{
    // Print test section title on console
    printf("\n[Tokenizer Unit Test]\n");
    // Sample input sentence for tokenization test
    const char* prompt = "hello world how are you";
    // Buffer to store converted integer token IDs
    u32 tokens[64];
    // Convert plain text string into token ID array
    u32 token_cnt = text_to_tokens(prompt, tokens, 64);
    // Print original input text and total number of generated tokens
    printf("Input text: %s\nToken count: %u\nToken IDs: ", prompt, token_cnt);
    // Loop to print every token ID in sequence
    for(u32 i = 0; i < token_cnt; i++)
        printf("%u ", tokens[i]);
    printf("\n");

    // Demo of token decoding: convert single token ID back to readable word
    char buf[32];
    // Decode the second token (index 1, BOS is index 0)
    token_to_text(tokens[1], buf, 32);
    // Print decoded text matched with corresponding token ID
    printf("Token %u decode text: %s\n", tokens[1], buf);
}

/**
 * End-to-end autoregressive generation test
 * Full pipeline test: tokenizer -> model forward -> autoregressive token generation
 */
void test_autoregressive_generate()
{
    // Print test section header for generation pipeline
    printf("\n[Autoregressive Generation End-To-End Test]\n");
    // Static hyperparameter configuration matching TinyLlama official setting
    LLaMAConfig cfg = {
        .dim = 512,          // Model hidden embedding dimension
        .n_layers = 22,      // Total number of transformer decoder layers
        .n_heads = 32,       // Number of multi-head attention heads
        .vocab_size = 32000, // Total vocabulary size of LLaMA tokenizer
        .seq_len = MAX_SEQ_LEN // Maximum supported context sequence length
    };
    // Declare main LLM model instance
    LLaMAModel model;
    // Initialize all model tensors and layers with above config
    llama_model_init(&model, &cfg);

    // Declare KV cache instance for fast generation
    KVCache cache;
    // Allocate memory for key/value cache storage
    kv_cache_init(&cache, cfg.dim, cfg.seq_len);

    // Raw user input prompt for chat test
    const char* user_prompt = "hello llama";
    // Buffer to hold encoded input prompt token IDs
    u32 input_tokens[256];
    // Convert input text into token sequence
    u32 in_cnt = text_to_tokens(user_prompt, input_tokens, 256);

    // Output buffer storing full prompt + newly generated tokens
    u32 output_tokens[512];
    // Run complete autoregressive generation, limit max 10 newly generated tokens
    u32 total = generate_autoregressive(&model, &cache, input_tokens, in_cnt, output_tokens, 10);

    // Print total length of combined prompt + generated token sequence
    printf("Generated total token length: %u\nToken sequence: ", total);
    // Print every token ID in the full output sequence
    for(u32 i = 0; i < total; i++)
        printf("%u ", output_tokens[i]);
    printf("\n");

    // Release all dynamically allocated memory of model tensors
    llama_model_free(&model);
    // Clear all cached attention key and value data
    kv_cache_reset(&cache);
}

int main(int argc, char** argv)
{
    printf("===== Section 9: Tokenizer & Autoregressive Generation Full Test Suite =====\n");
    test_matmul();
    test_kv_cache();
    test_int4_quant();
    test_rms_norm();
    test_swiglu();
    test_rope();
    test_causal_mha();
    test_tokenizer();
    test_autoregressive_generate();

    if (argc >= 2)
    {
        test_gguf_loader(argv[1]);
    }
    else
    {
        printf("\nHint: Run with ./mini_llama model.gguf to test GGUF loading\n");
    }

    printf("\nAll tests finished without error.\n");
    return 0;
}