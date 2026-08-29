#include "quant.h"

void quant_int4(const f32* src, u8* dst, u32 n, f32* scale, f32* zp) {
    f32 minv = src[0], maxv = src[0];
    for (u32 i = 0; i < n; i++) {
        if (src[i] < minv) minv = src[i];
        if (src[i] > maxv) maxv = src[i];
    }
    *scale = (maxv - minv) / 15.0f;
    *zp    = -minv / (*scale);

    for (u32 i = 0; i < n; i += 2) {
        u8 v1 = (u8)round((src[i] - minv) / (*scale));
        u8 v2 = 0;
        if (i+1 < n) v2 = (u8)round((src[i+1] - minv) / (*scale));
        dst[i/2] = (v1 & 0x0F) | ((v2 & 0x0F) << 4);
    }
}

void dequant_int4(f32* dst, const u8* src, u32 n, f32 scale, f32 zp) {
    for (u32 i = 0; i < n; i++) {
        u8 pack = src[i/2];
        u8 val = (i % 2 == 0) ? (pack & 0x0F) : ((pack >> 4) & 0x0F);
        dst[i] = (f32)val * scale - zp * scale;
    }
}