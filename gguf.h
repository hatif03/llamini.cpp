#ifndef GGUF_H
#define GGUF_H

#include "common.h"

// GGUF is the modern LLM model file format
#define GGUF_MAGIC 0x46554747 // "GGUF"

// GGUF file header
typedef struct {
    u32 magic;      // Must be GGUF
    u32 version;    // GGUF version
    u64 n_tensors;  // Number of tensors in model
    u64 n_metadata; // Metadata count
} GGUFHeader;

// GGUF File handle
typedef struct {
    int fd;             // File descriptor
    void* data;         // Mapped file data
    size_t size;        // File size
    GGUFHeader hdr;     // Header info
    u64 tensor_offset;  // Start of tensor data
} GGUFFile;

// Load and unload GGUF model
int gguf_open(const char* path, GGUFFile* gf);
void gguf_close(GGUFFile* gf);

// Read float32 weights from GGUF
void gguf_read_f32(GGUFFile* gf, u64 offset, f32* out, u64 elem_cnt);

#endif