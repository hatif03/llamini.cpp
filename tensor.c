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
    t->src_gf = NULL;
    t->src_info = NULL;
    return t;
}

void tensor_free(Tensor* t) {
    if (t) {
        free(t->data); // no-op for a lazy tensor (data is already NULL)
        free(t);
    }
}

void tensor_get_row(const Tensor* t, u32 row, u32 row_len, f32* out) {
    if (t->data) {
        memcpy(out, t->data + (u64)row * row_len, row_len * sizeof(f32));
        return;
    }
    gguf_dequantize_rows(t->src_gf, t->src_info, row, row + 1, row_len, out);
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