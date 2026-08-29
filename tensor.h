#ifndef TENSOR_H
#define TENSOR_H

#include "common.h"

/**
 * Multi-dimensional Tensor structure
 * Supports up to 4 dimensions; this course only uses 1D(vector) & 2D(matrix)
 */
typedef struct {
    u32 dims[4];    // Store size of each dimension
    u32 ndim;       // Count of valid dimensions (1 = vector, 2 = matrix)
    u32 size;       // Total number of floating-point elements in this tensor
    f32* data;      // Pointer to contiguous f32 data buffer
} Tensor;

/**
 * Allocate memory and initialize a new zero-filled tensor
 * @param ndim Number of target dimensions
 * @param shape Array storing length of each dimension
 * @return Pointer to newly created Tensor struct
 */
Tensor* tensor_create(u32 ndim, u32* shape);

/**
 * Release all heap memory occupied by a tensor to prevent memory leaks
 * @param t Target tensor pointer to destroy
 */
void tensor_free(Tensor* t);

/**
 * Standard matrix multiplication: C = A * B
 * A: m × k matrix, B: k × n matrix, C: m × n output matrix
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output result matrix
 */
void matmul(Tensor* A, Tensor* B, Tensor* C);

/**
 * Element-wise vector addition: out = a + b
 * @param out Output vector buffer
 * @param a First input vector
 * @param b Second input vector
 * @param n Total element count
 */
void vec_add(f32* out, const f32* a, const f32* b, u32 n);

/**
 * Vector scalar multiplication: out = a * scalar
 * @param out Output vector buffer
 * @param a Input source vector
 * @param s Scalar multiplier value
 * @param n Total element count
 */
void vec_scale(f32* out, const f32* a, f32 s, u32 n);

#endif