#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "quant.h"
#include "gguf.h"

/**
 * Unit test for matrix multiplication core tensor math
 */
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

/**
 * Unit test to verify KV Cache init and reset logic
 */
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

/**
 * Unit test for INT4 quantization & dequantization correctness
 */
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

/**
 * Minimal GGUF loader test: only validate header parsing
 * Pass your tinyllama .gguf file path as argument to run
 */
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

int main(int argc, char** argv)
{
    printf("===== Section 5: Tensor + KV Cache + INT4 Quant + GGUF Loader Test =====\n");
    test_matmul();
    test_kv_cache();
    test_int4_quant();

    // If user pass gguf file path, run GGUF test
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