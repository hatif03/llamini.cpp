#include "tensor.h"

/**
 * Create tensor struct and zero-initialize data buffer with calloc
 */
Tensor* tensor_create(u32 ndim, u32* shape)
{
    // Allocate memory for Tensor header struct
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->size = 1;

    // Calculate total element count and copy dimension sizes
    for (u32 i = 0; i < ndim; i++)
    {
        t->dims[i] = shape[i];
        t->size *= shape[i];
    }

    // Allocate continuous zeroed float memory for tensor data
    t->data = (f32*)calloc(t->size, sizeof(f32));
    return t;
}

/**
 * Safe deallocation for tensor, null pointer guard to avoid crash
 */
void tensor_free(Tensor* t)
{
    if (t != NULL)
    {
        free(t->data);
        free(t);
    }
}

/**
 * Naive triple-loop matrix multiplication for teaching purposes
 * Not optimized for cache performance, easy to read & understand
 */
void matmul(Tensor* A, Tensor* B, Tensor* C)
{
    u32 m = A->dims[0]; // Row count of matrix A
    u32 k = A->dims[1]; // Column count of A / Row count of B
    u32 n = B->dims[1]; // Column count of matrix B

    // Iterate every output row
    for (u32 i = 0; i < m; i++)
    {
        // Iterate every output column
        for (u32 j = 0; j < n; j++)
        {
            f32 dot_sum = 0.0f;
            // Dot product over shared dimension k
            for (u32 l = 0; l < k; l++)
            {
                dot_sum += A->data[i * k + l] * B->data[l * n + j];
            }
            C->data[i * n + j] = dot_sum;
        }
    }
}

/**
 * Element-wise add two equal-length float vectors
 */
void vec_add(f32* out, const f32* a, const f32* b, u32 n)
{
    for (u32 i = 0; i < n; i++)
    {
        out[i] = a[i] + b[i];
    }
}

/**
 * Multiply every element of a vector by a single scalar value
 */
void vec_scale(f32* out, const f32* a, f32 s, u32 n)
{
    for (u32 i = 0; i < n; i++)
    {
        out[i] = a[i] * s;
    }
}