---
name: zero-dep-hackathon
description: Enforces Zero Dependency hackathon rules for llamini.cpp (C libc/POSIX only). Use when implementing, reviewing, choosing libraries, or deciding whether a header or tool is allowed.
---

# Zero Dependency hackathon

Cheat-sheets: https://zerodepshack.com/cheatsheets

## The one rule

Zero third-party **runtime** dependencies. The shipped artifact's dependency manifest is empty.

C / C++: libc, POSIX, and the C++ standard library (libstdc++/libc++). This repo is **C99**, so C stdlib + POSIX only. No vendored third-party libraries, and no Boost, fmt, abseil, or header-only drop-ins. Header-only is still third-party.

`-lm` is libc. `gcc` and `make` are the toolchain and do not count.

If about to suggest installing a package, stop and find the standard-library answer. If C has no stdlib answer, say so and write it.

## Where the C stdlib stops

The cheat sheet marks these `none` — writing them is the project, not a reason to add a package:

JSON, HTTP, TLS, crypto, compression, test framework, argument parser, hashmap.

`pthreads` for concurrency; `<stdio.h>` and `<string.h>` for the rest.

## Legal substitutions (this repo)

| Would normally import | Use instead |
| --- | --- |
| ggml / llama.cpp GGUF loader | `open` + `fstat` + `mmap` + `memcpy` (`gguf.c`) |
| BLAS / ggml matmul | hand-rolled loops + `malloc` (`tensor.c`) |
| sentencepiece / tokenizers | write it; `argc`/`argv` for CLI |
| getopt packages | `argc` / `argv` + `strcmp` |
| OpenSSL | do not add; if a hash is needed, a documented FNV/djb2 in-tree, disclosed in STDLIB.md |
| chalk / color crates | raw ANSI + honour `NO_COLOR` |
| a test runner | `--test` + `printf` in `main.c` |

## Illegal

- Linking ggml or llama.cpp
- Copying `ggml.c` / llama.cpp internals into the tree
- vcpkg, Conan, CMake FetchContent, pip, npm “just for the demo”
- `system()` / `popen` of llama-cli, Python, or any separately installed tool
- Reading a user-supplied `.gguf` is fine; launching the tool that produced it is not

Vendoring source to keep the manifest empty is a dependency. Disclose any code not written this weekend in STDLIB.md or it scores against Zero-Dependency Craft.

## Scoring (do not optimize for the wrong bonus)

- Functionality & Usefulness 35%
- Zero-Dependency Craft 30% (STDLIB.md quality lives here)
- Code Quality & Idiom 25%
- Innovation 10%

Bonuses: Single File +5, Reproducible Build +5, Package Killer +3, STDLIB Log +3. Single File is natural in C — **do not flatten this tree unless the user asks**.

Track F: the README must argue what people would normally import (ggml, llama.cpp, sentencepiece) and what libc/POSIX replaced.
