#ifndef GGUF_H
#define GGUF_H

#include "common.h"

// GGUF is the modern LLM model file format
#define GGUF_MAGIC 0x46554747 // "GGUF"

// GGUF metadata value types (matches the on-disk format's type tags)
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFValueType;

// GGML tensor element types this reader can dequantize. Anything else
// (Q2_K, Q3_K, Q8_K, ...) is detected and reported, never silently
// misread -- see gguf_dequantize_tensor.
#define GGML_TYPE_F32   0
#define GGML_TYPE_F16   1
#define GGML_TYPE_Q4_0  2
#define GGML_TYPE_Q4_1  3
#define GGML_TYPE_Q5_0  6
#define GGML_TYPE_Q8_0  8
#define GGML_TYPE_Q4_K  12
#define GGML_TYPE_Q5_K  13
#define GGML_TYPE_Q6_K  14

// One parsed metadata key/value. Scalars are decoded eagerly; STRING and
// ARRAY payloads are left in place (value_off points at their encoded
// bytes in the mapped file) and decoded on demand by the accessors below,
// since a few of these (the tokenizer vocab) are tens of thousands of
// entries and most callers only ever want a handful of scalar keys.
typedef struct {
    char* key;          // null-terminated, owned copy
    u32   type;          // GGUFValueType
    u64   value_off;     // byte offset (from file start) of the raw value
    u32   arr_elem_type;  // only meaningful when type == GGUF_TYPE_ARRAY
    u64   arr_len;        // only meaningful when type == GGUF_TYPE_ARRAY
} GGUFMeta;

// One parsed tensor-info entry.
typedef struct {
    char* name;        // null-terminated, owned copy
    u32   n_dims;
    u64   dims[4];      // ne[0..n_dims-1], ggml order (fastest-varying first)
    u32   ggml_type;     // GGML_TYPE_*
    u64   offset;        // byte offset from the start of tensor data section
} GGUFTensorInfo;

// GGUF file header
typedef struct {
    u32 magic;      // Must be GGUF
    u32 version;    // GGUF version
    u64 n_tensors;  // Number of tensors in model
    u64 n_metadata; // Metadata count
} GGUFHeader;

// GGUF File handle
typedef struct {
    int fd;               // File descriptor
    void* data;           // Mapped file data
    size_t size;           // File size
    GGUFHeader hdr;        // Header info
    u64 tensor_offset;     // Start of the (alignment-padded) tensor data section

    GGUFMeta* kv;          // n_metadata parsed metadata entries
    u64 n_kv;

    GGUFTensorInfo* tensors; // n_tensors parsed tensor-info entries
    u64 n_tensors;
} GGUFFile;

// Load and unload GGUF model. gguf_open fully parses the metadata KV table
// and the tensor-info table (real GGUF layout: header, metadata, tensor
// table, alignment padding, tensor data), not just the fixed header.
int gguf_open(const char* path, GGUFFile* gf);
void gguf_close(GGUFFile* gf);

// Read raw float32 data at `offset` bytes into the tensor-data section.
// Returns 0 on success, -1 if the requested range runs past the mapped file.
int gguf_read_f32(GGUFFile* gf, u64 offset, f32* out, u64 elem_cnt);

// Metadata scalar accessors. Return 0/def_val if the key is absent or of
// a different type than requested -- callers pass sane architecture
// defaults rather than treating a missing key as fatal.
u32 gguf_get_u32(GGUFFile* gf, const char* key, u32 def_val);
f32 gguf_get_f32(GGUFFile* gf, const char* key, f32 def_val);

// Decodes a scalar STRING metadata value (e.g. "general.architecture")
// into a freshly malloc'd null-terminated buffer -- caller frees. Returns
// NULL if the key is missing or not a scalar string (an ARRAY of strings
// is a different type; use gguf_get_string_array for that).
char* gguf_get_str(GGUFFile* gf, const char* key);

// Returns the number of elements in a metadata ARRAY value (0 if the key
// is absent or not an array), and optionally its element type via
// *elem_type_out (may be NULL).
u64 gguf_meta_array_len(GGUFFile* gf, const char* key, u32* elem_type_out);

// Decodes a STRING-array metadata value (e.g. tokenizer.ggml.tokens) into
// `count` freshly malloc'd, null-terminated strings. Caller frees each
// string and the returned array. Returns NULL if the key is missing, not
// a string array, or out of range.
char** gguf_get_string_array(GGUFFile* gf, const char* key, u64* count_out);

// Decodes a FLOAT32-array metadata value into a freshly malloc'd array of
// `count` floats (caller frees). Returns NULL if the key is missing or
// not a float32 array.
f32* gguf_get_f32_array(GGUFFile* gf, const char* key, u64* count_out);

// Decodes an INT32-array metadata value the same way (token_type, etc).
i32* gguf_get_i32_array(GGUFFile* gf, const char* key, u64* count_out);

// Finds a tensor by exact name (e.g. "blk.0.attn_q.weight"). Returns NULL
// if not present.
const GGUFTensorInfo* gguf_find_tensor(GGUFFile* gf, const char* name);

// Total element count of a tensor (product of its dims).
u64 gguf_tensor_count(const GGUFTensorInfo* info);

// Dequantizes a tensor's data into `out` (must hold gguf_tensor_count(info)
// floats). Returns 0 on success; -1 if the tensor's bytes run past the
// mapped file, or if its ggml_type isn't one of the types this reader
// supports (see GGML_TYPE_* above) -- callers must treat -1 as "could not
// load this weight," never proceed with whatever `out` happened to hold.
int gguf_dequantize_tensor(GGUFFile* gf, const GGUFTensorInfo* info, f32* out, u64 out_count);

#endif
