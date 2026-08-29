#include "quant.h"

/**
 * FP32 → INT4 affine quantization + bit packing
 * Map float range [min, max] to integer range [0, 15]
 * Pack two 4-bit values into one 8-bit unsigned char
 */
void quant_int4(const f32* src, u8* dst, u32 n, f32* scale, f32* zp)
{
    // Step 1: Find min and max value in source float array
    f32 minv = src[0];
    f32 maxv = src[0];
    for (u32 i = 0; i < n; i++)
    {
        if (src[i] < minv) minv = src[i];
        if (src[i] > maxv) maxv = src[i];
    }

    // Step 2: Calculate affine quantization parameters
    const f32 int_max = 15.0f; // Max value of unsigned 4-bit integer
    *scale = (maxv - minv) / int_max;
    *zp = -minv / (*scale);

    // Step3: Quantize floats and pack two INT4 into one byte
    for (u32 i = 0; i < n; i += 2)
    {
        // Quantize first element in pair
        u8 v1 = (u8)roundf((src[i] - minv) / (*scale));
        u8 v2 = 0;

        // Quantize second element if it exists
        if (i + 1 < n)
        {
            v2 = (u8)roundf((src[i+1] - minv) / (*scale));
        }

        // Pack: low 4 bits = v1, high 4 bits = v2
        dst[i / 2] = (v1 & 0x0F) | ((v2 & 0x0F) << 4);
    }
}

/**
 * INT4 unpack + dequantization back to FP32
 * Recover original approximate float values using stored scale and zero point
 */
void dequant_int4(f32* dst, const u8* src, u32 n, f32 scale, f32 zp)
{
    for (u32 i = 0; i < n; i++)
    {
        u8 packed_byte = src[i / 2];
        u8 int_val;

        // Extract 4-bit value from packed byte
        if (i % 2 == 0)
        {
            int_val = packed_byte & 0x0F; // Lower 4 bits for even index
        }
        else
        {
            int_val = (packed_byte >> 4) & 0x0F; // Upper 4 bits for odd index
        }

        // Affine dequant formula
        dst[i] = ((f32)int_val - zp) * scale;
    }
}