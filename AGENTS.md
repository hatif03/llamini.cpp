# llamini.cpp

llamini.cpp: a minimal, from-scratch C99 reimplementation of llama.cpp's core, for the Zero Dependency hackathon (Track F). libc, POSIX, and the C standard library only.

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

Stubs and known limits (see README "Limits" for the current list — e.g. no multi-turn memory, partial quant coverage, an unverified chat template) must be named there, not papered over. When a stub gets fixed, update README instead of leaving stale disclosure text.

## Build log — keep it current

Every substantive change (a feature, a real bug fix, a failed experiment, a
wrong assumption caught by testing) gets an entry appended to `BUILDLOG.md`
in the same sitting as the change — before or right after its commit, not
reconstructed from memory later. State what was tried and what actually
happened, including when something made things worse or had to be
re-fixed; do not smooth this over. `BUILDLOG.md` is the raw source material
for `WRITEUP.md` (the hackathon Write-Up side quest post) and any future
write-up — keeping it current as you go is what makes that possible without
reconstructing history after the fact.

## Skills

- Event brief and C substitutions: `zero-dep-hackathon`
- Module map and current vs target: `llamini-architecture`
- README, STDLIB.md, deps-proof, `.zero-dep.toml`: `hackathon-submit`
