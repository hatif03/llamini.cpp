#include "tensor.h"

Tensor* tensor_create(u32 ndim, u32* shape) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->size = 1;
    for (u32 i = 0; i < ndim; i++) {
        t->dims[i] = shape[i];
        t->size *= shape[i];
    }
    t->data = (f32*)calloc(t->size, sizeof(f32));
    return t;
}

void tensor_free(Tensor* t) {
    if (t) {
        free(t->data);
        free(t);
    }
}

void matmul(Tensor* A, Tensor* B, Tensor* C) {
    u32 m = A->dims[0];
    u32 k = A->dims[1];
    u32 n = B->dims[1];

    for (u32 i = 0; i < m; i++) {
        for (u32 j = 0; j < n; j++) {
            f32 sum = 0.0f;
            for (u32 l = 0; l < k; l++) {
                sum += A->data[i * k + l] * B->data[l * n + j];
            }
            C->data[i * n + j] = sum;
        }
    }
}

void vec_add(f32* out, const f32* a, const f32* b, u32 n) {
    for (u32 i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void vec_scale(f32* out, const f32* a, f32 s, u32 n) {
    for (u32 i = 0; i < n; i++) out[i] = a[i] * s;
}