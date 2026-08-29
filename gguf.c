#include "gguf.h"

/**
 * Open GGUF file, create zero-copy mmap mapping and parse header information
 */
int gguf_open(const char* path, GGUFFile* gf)
{
    memset(gf, 0, sizeof(GGUFFile));

    // Step 1: Open model file read-only
    gf->fd = open(path, O_RDONLY);
    if (gf->fd < 0)
    {
        fprintf(stderr, "GGUF Error: Cannot open file %s\n", path);
        return -1;
    }

    // Step 2: Get total file size via stat
    struct stat st;
    if (fstat(gf->fd, &st) != 0)
    {
        close(gf->fd);
        fprintf(stderr, "GGUF Error: Failed to get file stat\n");
        return -1;
    }
    gf->size = st.st_size;

    // Step3: Create read-only private mmap zero-copy mapping
    // Cast last argument to off_t to eliminate type warning
    gf->data = mmap(NULL, gf->size, PROT_READ, MAP_PRIVATE, gf->fd, (off_t)0);
    if (gf->data == MAP_FAILED)
    {
        close(gf->fd);
        fprintf(stderr, "GGUF Error: mmap mapping failed\n");
        return -1;
    }

    // Step4: Copy header binary data from mapped memory
    memcpy(&gf->hdr, gf->data, sizeof(GGUFHeader));

    // Step5: Validate GGUF magic signature
    if (gf->hdr.magic != GGUF_MAGIC)
    {
        munmap(gf->data, gf->size);
        close(gf->fd);
        fprintf(stderr, "GGUF Error: Not a valid GGUF file (wrong magic)\n");
        return -1;
    }

    // Step6: Set tensor data starting offset (skip fixed header)
    gf->tensor_offset = sizeof(GGUFHeader);
    return 0;
}

/**
 * Clean up all system resources occupied by GGUF file
 */
void gguf_close(GGUFFile* gf)
{
    if (!gf) return;
    // Release mmap virtual memory
    if (gf->data)
    {
        munmap(gf->data, gf->size);
        gf->data = NULL;
    }
    // Close file descriptor
    if (gf->fd >= 0)
    {
        close(gf->fd);
        gf->fd = -1;
    }
}

/**
 * Copy continuous FP32 weight block from mapped GGUF memory to output buffer
 */
void gguf_read_f32(GGUFFile* gf, u64 offset, f32* out, u64 elem_cnt)
{
    u8* src_ptr = (u8*)gf->data + gf->tensor_offset + offset;
    memcpy(out, src_ptr, elem_cnt * sizeof(f32));
}