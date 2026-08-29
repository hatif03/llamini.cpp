#ifndef TENSOR_H
#define TENSOR_H

#include "common.h"

typedef struct {
    u32 dims[4];
    u32 ndim;
    u32 size;
    f32* data;
} Tensor;

Tensor* tensor_create(u32 ndim, u32* shape);
void tensor_free(Tensor* t);
void matmul(Tensor* A, Tensor* B, Tensor* C);
void vec_add(f32* out, const f32* a, const f32* b, u32 n);
void vec_scale(f32* out, const f32* a, f32 s, u32 n);

#endif