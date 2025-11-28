# Hashhat (CUDA GPU Password Cracker)

GPU Programming 2025 final project. Build a CUDA-based brute-force password cracker. Target platform: CSC GPUs (Puhti/Mahti); develop locally with CUDA where possible. CLI-first; optional curses-style dashboard for progress/H/s/ETA.

Team: Inka Byskata, Max Söderholm, Niina Rahko, Rasmus Lagerqvist, Tobias Holm

```
                    _.--._
               _.-.'      `.-._
             .' ./`--...--' \  `.
    .-.      `.'.`--.._..--'   .'
_..'.-.`-._.'( (-..__    __..-'
 >.'   `-...' ) )    ````
 '           / /     "brute force, but with a hat"
        .._.'.'    
         >.-'   v0.1.0 - Åbo Akademi University, 2025
         '
  /\  /\__ _ ___| |__ | |__   __ _| |_ 
 / /_/ / _` / __| '_ \| '_ \ / _` | __|
/ __  / (_| \__ \ | | | | | | (_| | |_ 
\/ /_/ \__,_|___/_| |_|_| |_|\__,_|\__|
```

## What this tool does

Hashhat takes a hashed password, generates lots of password guesses on the GPU, hashes each guess, and stops when a guess matches the target hash. You choose which characters to allow (lower/upper/numbers/symbols) and the length range to try. The GPU’s massive parallelism lets us test millions of guesses at once, so we can measure how quickly a given hash falls to brute force. This is useful for security audits and for demonstrating GPU parallel programming: the same idea as Hashcat, but scoped to our course project with clear hooks for experimentation and optimization.

## Quick references to get started

- What is the MD5 Algorithm? — <https://www.geeksforgeeks.org/computer-networks/what-is-the-md5-algorithm/>
- SHA (Secure Hash Algorithms) — <https://en.wikipedia.org/wiki/Secure_Hash_Algorithms>
- Overview (readable): Wikipedia “Cryptographic hash function” — <https://en.wikipedia.org/wiki/Cryptographic_hash_function>

## Goals

- Recover passwords by brute-forcing a hash (start with MD5; stretch goal: SHA-1/SHA-256).
- Generate candidate strings on-GPU (index-to-string) without CPU intervention.
- Support adjustable charset (lower/upper/numbers/symbols) and length.
- Benchmark GPU vs single-threaded CPU; report hashes/sec and scaling on Puhti/Mahti.
- Optimize kernels (constant memory, loop unrolling, minimizing divergence) and document gains.

## Scope and Difficulty

- Baseline: crack fixed-length lowercase passwords with MD5 on GPU.
- Medium: add upper/lower/numbers/symbols toggle and variable length (e.g., 4-8 chars).
- Stretch: multiple hash algorithms (MD5, SHA)

## Workstreams (assignable, plain language)

- **1) Password generator + kernel hookup**: turn thread IDs into password guesses on the GPU, hash them, and check vs target. Decide grid/block sizes and how to handle length/charset.
- **2) Hash functions**: write MD5 on GPU first; later add SHA-1/SHA-256. Also provide a CPU version to verify correctness.
- **3) Host runner + CLI**: read the target hash and options, launch the GPU work, stop when found, print stats. Keep flags simple.
- **4) Optional terminal dashboard**: a curses UI that shows progress, hashes/sec, ETA, and the current length being tested.
- **5) Speed tuning + profiling**: move constants to fast memory, unroll loops, adjust occupancy; measure with Nsight/nvprof.
- **6) Benchmarks + writeup**: run on Puhti/Mahti, compare GPU vs CPU, and summarize results with graphs.

### Current TODO tags in code

- TODO 1: Wire GPU kernel launch (hash generation + compare) using parsed options.
- TODO 2: Add CPU baseline for correctness and simple benchmark output.
- TODO 3: Integrate curses dashboard when using the flag `--ui=curses`.
- TODO 4: Add configurable hash algorithm selection (MD5 now, SHA variants later).
- TODO 5: Move charset/hash data to constant memory and measure speedups.
- TODO 6: Add test hooks for small keyspaces to validate end-to-end (nice-to-have; tests are optional add-ons).

## Pick a TODO and claim it

Add your name next to a TODO here when you start:

- TODO 1:
- TODO 2:
- TODO 3:
- TODO 4:
- TODO 5:
- TODO 6:

## Architecture (proposed)

- `src/host/`: CLI, config parsing, kernel launches, CPU baseline.
- `src/device/`: CUDA kernels, device hash implementations, constant-memory setup.
- `tests/`: minimal correctness checks (GPU vs CPU hash, small search spaces).
- `docs/`: benchmark results, run logs, final report material.

## Development Setup

- CUDA toolkit (12.x recommended), CMake or `nvcc`.
- Access to a CUDA-capable GPU locally or CSC Puhti/Mahti.
- Language: C++17 + CUDA.
- On macOS (no CUDA): use the host-only build to test CLI parsing (`make host`).

## How to Run (early sketch)

```bash
# GPU build (uses nvcc; set ARCH as needed, e.g., sm_80)
make cuda

# host-only build (no CUDA; e.g., macOS) to test CLI parsing
make host

# run (GPU build)
./bin/hashhat --hash <md5_hex> --charset lower,upper,num,sym --min-len 4 --max-len 8

# optional dashboard (placeholder; GPU build)
./bin/hashhat --hash <md5_hex> --ui curses
```

(CLI flags are placeholders)

## Working independently

- Build and smoke-test: `make host && ./bin/hashhat_host --help` (mac/CPU) or `make cuda && ./bin/hashhat --help` (CUDA).
- Grab a TODO from the list and work in your area; leave the rest stubbed.
- Keep the boundary simple: host code should call a single launcher (e.g., `launch_cracker(options)`); device code implements the heavy lifting.
- Tests are optional; if you add them, keep them small and focused (e.g., “MD5 matches CPU for known input”).
- Leave short comments near your changes about assumptions or expected inputs so others can plug in their parts without conflicts.
