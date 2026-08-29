#ifndef KV_CACHE_H
#define KV_CACHE_H

#include "common.h"
#include "tensor.h"

typedef struct {
    f32* key;
    f32* val;
    u32 dim;
    u32 max_seq;
    u32 cur_seq;
} KVCache;

void kv_cache_init(KVCache* cache, u32 dim, u32 max_seq);
void kv_cache_reset(KVCache* cache);

#endif // KV_CACHE_H