# llamini.cpp

C99 mini llama.cpp for the Zero Dependency hackathon (Track F). libc, POSIX, and the C standard library only.

## Hard rule

The shipped artifact has an empty dependency manifest. `-lm` is libc. Out: Boost, fmt, abseil, ggml-as-a-library, nlohmann/json, vcpkg, Conan, header-only drop-ins, and vendoring llama.cpp or ggml source. Copying a library into the tree to fake an empty manifest is a dependency; disclose it in STDLIB.md or it scores against Zero-Dependency Craft.

If you are about to suggest installing a package, stop. If C has no standard-library answer, say so and write it.

## No hidden deps

Do not `system()` or `popen` llama-cli, Python, or any separately installed tool. Reading a user-supplied `.gguf` is allowed (parse the file; do not launch the tool that produced it).

## New code this weekend

Do not copy llama.cpp internals. If any pre-existing snippet is kept, disclose it in STDLIB.md.

## Idiom

- C99, POSIX (`mmap`, `open`, `fstat`).
- Use the typedefs in `common.h` (`f32`, `u32`, …).
- One concern per `.c` / `.h` pair.
- Tests stay in-process (`--test` + `printf`). C has no stdlib test framework; do not add a test package.

## Honesty

Stubs (keyword chat, skipped GGUF metadata, unlinked tokenizer/generate units) must be named in README limits, not papered over.

## Skills

- Event brief and C substitutions: `zero-dep-hackathon`
- Module map and current vs target: `llamini-architecture`
- README, STDLIB.md, deps-proof, `.zero-dep.toml`: `hackathon-submit`
