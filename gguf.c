#include "gguf.h"

int gguf_open(const char* path, GGUFFile* gf) {
    memset(gf, 0, sizeof(GGUFFile));
    gf->fd = open(path, O_RDONLY);
    if (gf->fd < 0) return -1;

    struct stat st;
    fstat(gf->fd, &st);
    gf->size = st.st_size;

    gf->data = mmap(NULL, gf->size, PROT_READ, MAP_PRIVATE, gf->fd, 0);
    if (gf->data == MAP_FAILED) { close(gf->fd); return -1; }

    memcpy(&gf->hdr, gf->data, sizeof(GGUFHeader));
    if (gf->hdr.magic != GGUF_MAGIC) {
        munmap(gf->data, gf->size);
        close(gf->fd);
        return -1;
    }
    // 简易偏移：跳过头部，指向张量区（教学简化实现）
    gf->tensor_offset = sizeof(GGUFHeader);
    return 0;
}

void gguf_close(GGUFFile* gf) {
    if (gf->data) munmap(gf->data, gf->size);
    if (gf->fd >= 0) close(gf->fd);
}

void gguf_read_f32(GGUFFile* gf, u64 offset, f32* out, u64 elem_cnt) {
    u8* src = (u8*)gf->data + gf->tensor_offset + offset;
    memcpy(out, src, elem_cnt * sizeof(f32));
}