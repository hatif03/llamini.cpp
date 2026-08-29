#include "common.h"
#include "tensor.h"

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

int main(int argc, char** argv)
{
    printf("===== Section 2: Tensor System Unit Test =====\n");
    test_matmul();
    printf("\nAll tensor tests finished without error.\n");
    return 0;
}