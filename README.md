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

## Workstreams (what each person can own)

- **1) GPU search loop**: map thread IDs to password guesses, hash, compare. Grid/block sizing and length/charset handling. **Status:** done (SHA-1)
- **2) Hash functions**: add MD5 (CPU+GPU); SHA-1 is done. **Status:** MD5 pending (Max)
- **3) Host runner + CLI**: parse flags, launch GPU work, stop when found, print stats. **Status:** done
- **4) Terminal dashboard (optional)**: curses UI for progress/H/s/ETA. **Status:** not started
- **5) Speed tuning**: constants/occupancy/unrolling; profile with Nsight. **Status:** partial (constants, block=256, chunking)
- **6) Benchmarks + report**: run on Puhti/Mahti, log CPU vs GPU, summarize. **Status:** done (SHA-1)
- **7) Analysis**: bottlenecks, optimization notes, benchmark results in `results/ANALYSIS.md`. **Status:** done (Rasmus)

### Current TODO tags in code

- TODO 1: Wire GPU kernel launch (hash generation + compare) using parsed options. **Status:** done (SHA-1)
- TODO 2: Add CPU baseline for correctness and simple benchmark output. **Status:** done (SHA-1); MD5 pending
- TODO 3: Integrate curses dashboard when using the flag `--ui=curses`. **Status:** not started
- TODO 4: Add configurable hash algorithm selection (MD5 now, SHA variants later). **Status:** SHA-1 done; MD5 pending (Max)
- TODO 5: Move charset/hash data to constant memory and measure speedups. **Status:** partial (constants used)
- TODO 6: Add test hooks for small keyspaces to validate end-to-end (nice-to-have; tests are optional add-ons). **Status:** not started
- TODO 7: Benchmark harness + analysis writeup (CPU vs GPU timing, bottlenecks, results in `results/ANALYSIS.md`). **Status:** done (Rasmus)

## Pick a TODO and claim it (Name / Status)

- TODO 1: (Rasmus) done (SHA-1)
- TODO 2: (Rasmus; MD5 by Max) SHA-1 done; MD5 not started
- TODO 3: -
- TODO 4: (Rasmus; MD5 by Max) SHA-1 done; MD5 not started
- TODO 5: -
- TODO 6: -
- TODO 7: (Rasmus) done

## Contributions

- Rasmus Lagerqvist: Project skeleton/structure, team setup, SHA-1 CPU/GPU cracker, benchmark harness, analysis writeup.
- Tobias Holm: Cursor UI implementation and local gpu testing

## Project status

- CPU SHA-1 brute force implemented (capped workload for quick runs).
- GPU SHA-1 brute force implemented (lengths up to 8, simple kernel); requires CUDA build.
- Benchmark harness (`--benchmark`) runs CPU+GPU and logs to `results/ANALYSIS.md`.
- Default benchmark hash: SHA-1("aaa") if no `--hash` is provided.
- MD5 support pending (Max to implement under TODO 2/4).

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

## Ways of working

- Grab a TODO from the list and work in your area; leave the rest stubbed.
- Leave comments near your changes about assumptions or expected inputs so others can plug in their parts without conflicts.
- If you’re doing analysis on your results, add notes in `results/ANALYSIS.md`.
