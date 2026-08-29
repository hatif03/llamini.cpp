#include "kv_cache.h"

/**
 * Pre-allocate fixed-size key & value memory buffer once on initialization
 * Use calloc to fill buffer with zero by default
 */
void kv_cache_init(KVCache* cache, u32 dim, u32 max_seq)
{
    // Save model dimension and context limit to struct
    cache->dim = dim;
    cache->max_seq = max_seq;
    cache->cur_seq = 0;

    // Allocate memory for all Key vectors
    cache->key = (f32*)calloc(dim * max_seq, sizeof(f32));
    // Allocate memory for all Value vectors
    cache->val = (f32*)calloc(dim * max_seq, sizeof(f32));
}

/**
 * Clear token counter and wipe entire K/V buffer memory
 * Called when starting a new chat session or hitting max context length
 */
void kv_cache_reset(KVCache* cache)
{
    // Reset token counter to zero
    cache->cur_seq = 0;
    // Fill key buffer with zero
    memset(cache->key, 0, cache->dim * cache->max_seq * sizeof(f32));
    // Fill value buffer with zero
    memset(cache->val, 0, cache->dim * cache->max_seq * sizeof(f32));
}