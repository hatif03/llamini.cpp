# Zero dependency

Empty manifest. Never suggest a package, vcpkg, Conan, or a header-only drop-in.

- Allowed: libc, POSIX, C standard library, `-lm`, `gcc`/`make`.
- Forbidden: Boost, fmt, abseil, ggml, llama.cpp as a library, nlohmann/json, OpenSSL, any vendored third-party source.
- If C has no stdlib answer (JSON, HTTP, regex, crypto, hashmap), say so and write it.
- Do not copy library source into the tree to fake an empty manifest. Disclose any code not written this weekend in STDLIB.md.

```c
/* BAD — third-party */
#include <ggml.h>
#include "vendor/llama.h"

/* GOOD — libc / POSIX */
#include <stdio.h>
#include <sys/mman.h>
```
