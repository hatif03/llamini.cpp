#ifndef QUANT_H
#define QUANT_H

#include "common.h"

void quant_int4(const f32* src, u8* dst, u32 n, f32* scale, f32* zp);
void dequant_int4(f32* dst, const u8* src, u32 n, f32 scale, f32 zp);

#endif // QUANT_H