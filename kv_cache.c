#include "kv_cache.h"

void kv_cache_init(KVCache* cache, u32 dim, u32 max_seq) {
    cache->dim = dim;
    cache->max_seq = max_seq;
    cache->cur_seq = 0;
    cache->key = (f32*)calloc(dim * max_seq, sizeof(f32));
    cache->val = (f32*)calloc(dim * max_seq, sizeof(f32));
}

void kv_cache_reset(KVCache* cache) {
    cache->cur_seq = 0;
    memset(cache->key, 0, cache->dim * cache->max_seq * sizeof(f32));
    memset(cache->val, 0, cache->dim * cache->max_seq * sizeof(f32));
}