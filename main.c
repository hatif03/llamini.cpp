#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "quant.h"

/**
 * Unit test for matrix multiplication core tensor math
 */
void test_matmul()
{
    printf("\n[MatMul Unit Test]\n");
    // Define matrix shapes: A(2x2), B(2x2), output C(2x2)
    u32 shape_a[] = {2, 2};
    u32 shape_b[] = {2, 2};
    u32 shape_c[] = {2, 2};

    // Create empty tensors
    Tensor* A = tensor_create(2, shape_a);
    Tensor* B = tensor_create(2, shape_b);
    Tensor* C = tensor_create(2, shape_c);

    // Fill test matrix values
    // A = [[1, 2], [3, 4]]
    A->data[0] = 1.0f; A->data[1] = 2.0f;
    A->data[2] = 3.0f; A->data[3] = 4.0f;
    // B = [[5, 6], [7, 8]]
    B->data[0] = 5.0f; B->data[1] = 6.0f;
    B->data[2] = 7.0f; B->data[3] = 8.0f;

    // Execute matrix multiplication
    matmul(A, B, C);

    // Print computed result matrix
    printf("%.2f %.2f\n", C->data[0], C->data[1]);
    printf("%.2f %.2f\n", C->data[2], C->data[3]);

    // Free allocated tensor memory
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
    // Initialize cache with hidden dim 512, max sequence from global macro
    kv_cache_init(&cache, 512, MAX_SEQ_LEN);

    // Simulate storing 10 tokens
    cache.cur_seq = 10;
    printf("Cached token count before reset: %u\n", cache.cur_seq);

    // Execute full cache reset
    kv_cache_reset(&cache);
    printf("Cached token count after reset: %u\n", cache.cur_seq);
}

/**
 * Unit test for INT4 quantization & dequantization correctness
 * Compare original float values with restored decompressed values
 */
void test_int4_quant()
{
    printf("\n[INT4 Quantization Unit Test]\n");
    // Sample input float vector
    f32 original[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const u32 elem_count = 8;
    u8 compressed_buf[elem_count / 2]; // 2 int4 per byte
    f32 restored[elem_count];
    f32 scale_param, zp_param;

    // Step 1: Quantize FP32 to packed INT4 bytes
    quant_int4(original, compressed_buf, elem_count, &scale_param, &zp_param);
    // Step 2: Decompress INT4 back to FP32
    dequant_int4(restored, compressed_buf, elem_count, scale_param, zp_param);

    // Print comparison of original vs recovered values
    printf("Original | Restored\n");
    for (u32 i = 0; i < elem_count; i++)
    {
        printf("%.2f      %.2f\n", original[i], restored[i]);
    }
}

int main(int argc, char** argv)
{
    printf("===== Section 4: Tensor + KV Cache + INT4 Quant Unit Test =====\n");
    test_matmul();
    test_kv_cache();
    test_int4_quant();
    printf("\nAll tests finished without error.\n");
    return 0;
}