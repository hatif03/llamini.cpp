---
name: hackathon-submit
description: Writes Zero Dependency hackathon submit docs for llamini.cpp (README, STDLIB.md, deps-proof, .zero-dep.toml, Makefile). Use when authoring or reviewing those files or the one-command build.
---

# Hackathon submit (Track F)

Do not invent `package.json`, `Cargo.toml`, `vcpkg.json`, or `conanfile`. C’s empty manifest is **Makefile only** — no vcpkg / Conan / CMake FetchContent.

## Required files

| File | Role |
| --- | --- |
| `README.md` | what it does, how to run, honest limits |
| `STDLIB.md` | every “I'd normally import X, instead I used stdlib Y” |
| `Makefile` | one command → runnable `mini_llama` |
| `deps-proof.txt` | output showing only libc/libm (and no package manifests) |
| `.zero-dep.toml` | track letter + one-line pitch |

Pin the toolchain in the README: `gcc`, C99 (`-std=c99`), POSIX, `-lm`. Example run: `make` then `./mini_llama model.gguf` (or `--test`).

This repo currently has `README.MD`. Prefer `README.md` for judges; do not leave two conflicting READMEs.

## `.zero-dep.toml`

```toml
track = "F"
pitch = "A C99 mini llama.cpp: GGUF mmap and transformer ops with libc and POSIX only."
```

## STDLIB.md rows this repo can honestly claim

Write real substitutions with a one-line rationale. Empty bullets do not count.

| Instead of | Used | Rationale |
| --- | --- | --- |
| ggml / llama.cpp GGUF I/O | `open` / `mmap` / `memcpy` | POSIX file + mapping, no GGUF library |
| BLAS / ggml matmul | hand loops in `tensor.c` | libc `malloc` only |
| getopt / CLI packages | `argc` / `argv` + `strcmp` | C has no stdlib arg parser; keep it minimal |
| CTest / Unity / Check | `--test` + `printf` | C has no stdlib test framework; in-process only |
| sentencepiece | (when written) in-tree tokenizer | no tokenizer package |

Do not claim Q4_K_M, a full GGUF table parse, or a real forward pass until the code does that.

## `deps-proof.txt`

Capture commands and output, for example:

- Linux: `ldd mini_llama` — only libc, libm, libpthread, ld-linux
- macOS: `otool -L mini_llama`
- Windows (MinGW): `ldd mini_llama.exe` or dumpbin `/dependents`

Also record: no `vcpkg.json`, no `conanfile`, no `CMakeLists.txt` with FetchContent.

## README limits (required)

Name what is still a stub: keyword chat in `main.c`, skipped GGUF metadata, unlinked `tokenizer.c` / `generate.c`, generic INT4 vs `Q4_K_M`. Honest and correct beats fast and hand-wavy.
