#ifndef TENSOR_H
#define TENSOR_H

#include "common.h"
#include "gguf.h"

// A tensor is either eager (data holds size floats, calloc'd by
// tensor_create -- true for every synthetic/small tensor: norms, biases,
// and anything --test writes directly into) or lazy (data is NULL,
// src_gf/src_info identify where to dequantize rows from on demand --
// used only for the handful of large weight matrices where materializing
// the whole thing to f32 up front would multiply memory footprint several
// times over). A tensor is lazy iff src_gf != NULL; every existing caller
// of tensor_create is completely unaffected since those two fields default
// to NULL and data stays eagerly populated exactly as before.
typedef struct {
    u32 dims[4];
    u32 ndim;
    u32 size;
    f32* data;
    GGUFFile* src_gf;             // non-NULL iff this tensor is lazy
    const GGUFTensorInfo* src_info; // borrowed from gf->tensors[] -- never freed here
} Tensor;

Tensor* tensor_create(u32 ndim, u32* shape);
void tensor_free(Tensor* t);
void matmul(Tensor* A, Tensor* B, Tensor* C);
void vec_add(f32* out, const f32* a, const f32* b, u32 n);
void vec_scale(f32* out, const f32* a, f32 s, u32 n);

// Copies one logical row (row_len elements) out of a tensor into `out` --
// a memcpy for eager tensors, a single-row on-demand dequant for lazy ones.
// Used for embedding-table lookups, which index by row rather than doing a
// matmul (see linear() in model.h for the matmul case).
void tensor_get_row(const Tensor* t, u32 row, u32 row_len, f32* out);

#endif