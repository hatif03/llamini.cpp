#ifndef GGUF_H
#define GGUF_H

#include "common.h"
#include "tensor.h"

/**
 * Official GGUF v3 fixed header structure
 * Stores global metadata for entire model file
 */
typedef struct {
    u32 magic;        // File signature, must equal GGUF_MAGIC
    u32 version;      // GGUF format major version
    u64 n_tensors;    // Total count of weight tensors inside file
    u64 n_metadata;   // Total count of model hyperparameter metadata entries
} GGUFHeader;

/**
 * Global GGUF file handle
 * Manages file descriptor, mmap mapped memory and offsets
 */
typedef struct {
    int fd;             // System file descriptor for opened GGUF file
    void* data;         // Base pointer of mmap zero-copy mapped memory
    size_t size;        // Total byte size of GGUF file
    GGUFHeader hdr;     // Parsed GGUF header info
    u64 tensor_offset;  // Start byte offset of all tensor weight data
} GGUFFile;

/**
 * Open GGUF file, create mmap mapping and parse header
 * @param path Path to target GGUF model file
 * @param gf Empty GGUFFile handle to fill
 * @return 0 = success, -1 = any failure (file missing / bad magic / mmap fail)
 */
int gguf_open(const char* path, GGUFFile* gf);

/**
 * Release mmap buffer and close file descriptor to avoid resource leak
 * @param gf Loaded GGUF file handle
 */
void gguf_close(GGUFFile* gf);

/**
 * Read continuous FP32 weight block from mapped GGUF memory into tensor buffer
 * @param gf Active GGUF file handle
 * @param offset Target byte offset inside tensor data region
 * @param out Destination float array buffer
 * @param elem_cnt Total f32 elements to copy
 */
void gguf_read_f32(GGUFFile* gf, u64 offset, f32* out, u64 elem_cnt);

#endif