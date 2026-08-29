#ifndef COMMON_H
#define COMMON_H

// Standard system headers for IO, memory, string, math operations
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>  // Required for mmap() prototype

// Global constant limits matching TinyLlama 1.1B hyperparameters
#define MAX_DIM        4096    // Maximum hidden dimension for tensor buffers
#define MAX_SEQ_LEN    2048    // Maximum token context window length
#define MAX_PROMPT_LEN 1024    // Max character length of user input prompt
#define EPS            1e-8f   // Small epsilon to avoid division by zero in normalization

// Fixed-width cross-platform data types for consistent LLM computation
typedef float          f32;         // 32-bit single-precision floating point
typedef unsigned char  u8;          // 8-bit unsigned byte for quantized weights
typedef unsigned int   u32;         // 32-bit unsigned integer for indexes & lengths
typedef unsigned long long u64;     // 64-bit unsigned integer for GGUF file offsets

// Reserved special token IDs (placeholder for later tokenizer module)
#define TOKEN_BOS  1  // Beginning of sequence token
#define TOKEN_EOS  2  // End of sequence token

// GGUF magic signature ASCII "GGUF" hex value
#define GGUF_MAGIC 0x46554747

#endif