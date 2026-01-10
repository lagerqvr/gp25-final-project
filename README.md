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

- Recover passwords by brute-forcing a hash (start with MD5/SHA-1).
- Generate candidate strings on-GPU (index-to-string) without CPU intervention.
- Support adjustable charset (lower/upper/numbers/symbols) and length.
- Benchmark GPU vs single-threaded CPU; report hashes/sec and scaling on Puhti/Mahti.
- Optimize kernels (constant memory, loop unrolling, minimizing divergence) and document gains.

## Scope and Difficulty

- Baseline: crack fixed-length lowercase passwords with MD5 on GPU.
- Medium: add upper/lower/numbers/symbols toggle and variable length (e.g., 4-8 chars).
- Stretch: multiple hash algorithms (MD5, SHA)

## Contributions

- Rasmus Lagerqvist: Project skeleton/structure, team setup, SHA-1 CPU/GPU cracker, benchmark harness, analysis writeup.
- Tobias Holm: Cursor UI implementation and local gpu testing

## Project structure

- `src/host/`: CLI, config parsing, kernel launches, CPU baseline, GPU kernels.
- `src/common/`: shared headers (e.g., options).
- `tests/`: minimal correctness checks (optional).
- `results/`: benchmarking logs and analysis notes.

## Build + CLI Usage Examples

```bash
# GPU build (uses nvcc; set ARCH as needed, e.g., sm_80)
make cuda

# host-only build (no CUDA; e.g., macOS) to test CLI parsing
make host

# run (GPU build) — SHA-1 hash
./bin/hashhat --algo sha1 --hash <sha1_hex> --charset lower,upper,num,sym --min-len 4 --max-len 8

# optional dashboard (placeholder; GPU build)
./bin/hashhat --hash <sha1_hex> --ui curses

# benchmark (CPU + GPU SHA-1; logs to results/ANALYSIS.md). If no hash provided, uses SHA-1("aaa").
./bin/hashhat --benchmark --algo sha1 --charset lower --min-len 3 --max-len 5

# crack a specific SHA-1 hash (GPU build). Example hashes: echo -n "aaa" | sha1sum
./bin/hashhat --algo sha1 --hash <sha1_hex> --charset lower --min-len 3 --max-len 5
```

(CLI flags are placeholders)
