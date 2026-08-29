#ifndef KV_CACHE_H
#define KV_CACHE_H

#include "common.h"
#include "tensor.h"

/**
 * KV Cache structure for LLM autoregressive inference acceleration
 * Stores all historical Key and Value vectors to avoid recomputation
 */
typedef struct {
    f32* key;        // Continuous buffer to store all historical Key vectors
    f32* val;        // Continuous buffer to store all historical Value vectors
    u32 dim;         // Hidden dimension size of each K/V vector
    u32 max_seq;     // Maximum supported context window length
    u32 cur_seq;     // Current number of cached tokens
} KVCache;

/**
 * Initialize KV Cache and pre-allocate fixed memory buffer
 * @param cache Pointer to empty KVCache struct
 * @param dim Model hidden dimension
 * @param max_seq Maximum context token limit
 */
void kv_cache_init(KVCache* cache, u32 dim, u32 max_seq);

/**
 * Reset cache counter and clear buffer for a brand-new conversation
 * @param cache Target KVCache instance
 */
void kv_cache_reset(KVCache* cache);

#endif