#ifndef QUANT_H
#define QUANT_H

#include "common.h"

/**
 * Convert FP32 float array into packed INT4 bytes
 * Two 4-bit integers are packed into one single u8 byte
 * @param src Input original FP32 weight array
 * @param dst Output compressed byte buffer
 * @param n Total number of float elements
 * @param scale Output quantization scaling factor
 * @param zp Output zero point offset
 */
void quant_int4(const f32* src, u8* dst, u32 n, f32* scale, f32* zp);

/**
 * Unpack INT4 compressed bytes and restore back to FP32 floats for inference
 * @param dst Output decompressed FP32 array
 * @param src Input packed INT4 byte buffer
 * @param n Total number of float elements to restore
 * @param scale Precomputed scale from quantization stage
 * @param zp Precomputed zero point from quantization stage
 */
void dequant_int4(f32* dst, const u8* src, u32 n, f32 scale, f32 zp);

#endif