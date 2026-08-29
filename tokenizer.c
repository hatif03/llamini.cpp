#include "tokenizer.h"

// Hardcoded tiny vocab lookup table for demo
// Static fixed vocabulary array storing word strings mapped to token IDs
static const char* vocab_table[MAX_TOKEN_VOCAB] = {
    "", "<bos>", "<eos>", "hello", "world", "how", "are", "you",
    "i", "am", "fine", "thank", "you", "what", "is", "llama"
};

/**
 * Simple naive tokenizer: split by space, map word to index
 * Encode raw input text into an array of integer token IDs
 * @param text Input human-readable text string
 * @param tokens Output buffer to store converted token IDs
 * @param max_tokens Maximum capacity limit of token output buffer
 * @return Total valid number of generated tokens
 */
u32 text_to_tokens(const char* text, u32* tokens, u32 max_tokens)
{
    // Counter to track how many valid tokens we have generated
    u32 token_cnt = 0;
    // Temporary character buffer to assemble single split word
    char buf[64];
    // Current write position index inside temporary word buffer
    u32 buf_idx = 0;

    // Prepend BOS token at the start of token sequence (begin-of-sequence marker)
    if (token_cnt < max_tokens)
        tokens[token_cnt++] = TOKEN_BOS;

    // Iterate through every character of input text until null terminator
    for (u32 i = 0; text[i] != '\0'; i++)
    {
        // Trigger word split when whitespace or newline character is detected
        if (text[i] == ' ' || text[i] == '\n')
        {
            // Skip empty buffer if no characters were collected for current word
            if (buf_idx == 0) continue;
            // Add null terminator to mark end of assembled word string
            buf[buf_idx] = '\0';
            // Reset buffer index for next word collection
            buf_idx = 0;

            // Look up matched word inside vocabulary table to get corresponding token ID
            u32 tid = 0;
            for (; tid < MAX_TOKEN_VOCAB; tid++)
            {
                if (strcmp(buf, vocab_table[tid]) == 0)
                    break;
            }
            // Fallback to token ID 0 if word is not found in vocabulary
            if (tid >= MAX_TOKEN_VOCAB) tid = 0;
            // Write token ID to output buffer if buffer capacity is not exceeded
            if (token_cnt < max_tokens)
                tokens[token_cnt++] = tid;
        }
        else
        {
            // Append current character to temporary word buffer, prevent buffer overflow
            if (buf_idx < 63)
                buf[buf_idx++] = text[i];
        }
    }

    // Process the last leftover word after loop ends (text without trailing space)
    if (buf_idx > 0)
    {
        buf[buf_idx] = '\0';
        u32 tid = 0;
        for (; tid < MAX_TOKEN_VOCAB; tid++)
        {
            if (strcmp(buf, vocab_table[tid]) == 0)
                break;
        }
        // Unknown word fallback to ID 0
        if (tid >= MAX_TOKEN_VOCAB) tid = 0;
        if (token_cnt < max_tokens)
            tokens[token_cnt++] = tid;
    }
    // Return total amount of converted tokens including BOS
    return token_cnt;
}

/**
 * Convert token ID back to text word
 * Decode single token ID back to readable vocabulary word string
 * @param token Input integer token ID to decode
 * @param out_str Output char buffer for decoded word
 * @param str_len Total allocated length of output string buffer
 */
void token_to_text(u32 token, char* out_str, u32 str_len)
{
    // Clear entire output buffer with zero bytes first
    memset(out_str, 0, str_len);
    // Exit early if token ID exceeds vocabulary range (invalid token)
    if (token >= MAX_TOKEN_VOCAB)
        return;
    // Copy matched vocabulary word into output buffer, reserve space for null terminator
    strncpy(out_str, vocab_table[token], str_len - 1);
}