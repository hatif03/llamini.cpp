#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "common.h"

// Simple mock tokenizer for demo, word -> token id mapping
#define MAX_TOKEN_VOCAB 128

/**
 * Convert simple input text string to token ID array
 * This is a minimal placeholder tokenizer for teaching demo
 * @param text Input user prompt string
 * @param tokens Output token id buffer
 * @param max_tokens Max capacity of token buffer
 * @return Actual number of generated tokens
 */
u32 text_to_tokens(const char* text, u32* tokens, u32 max_tokens);

/**
 * Convert single token ID back to readable text string
 * @param token Target token id
 * @param out_str Output character buffer
 * @param str_len Max length of output string
 */
void token_to_text(u32 token, char* out_str, u32 str_len);

#endif