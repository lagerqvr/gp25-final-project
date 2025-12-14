# Hashhat Performance & Analysis (SHA-1)

Owner: Rasmus Lagerqvist
Goal: Performance report and bottleneck analysis.

## What ran (Mahti A100, CUDA 11.5.0)

- Device: A100 (gpusmall), block=256 threads, chunk=10M candidates per launch, target/charset in `__constant__`, lengths capped at 8.
- Hash: SHA-1 only (MD5 pending); charset parsing supports lower/upper/num/sym.
- Default benchmark target: SHA-1("aaa") if no `--hash` is given.

### Run A — Benchmark (default target)

- Command: `./bin/hashhat --benchmark --algo sha1 --charset lower --min-len 3 --max-len 5`
- Target: SHA-1("aaa")
- Results: CPU ≈ 2.09M H/s, GPU ≈ 33.8M H/s → ~16.2x speedup.
- Notes: good steady throughput.

### Run B — Crack a harder hash (`t0m8`, mixed charset)

- Password: `t0m8`
- Hash (SHA-1): `0554cb9ed835ba8cff065406ad639156c18b8435` (from `echo -n "t0m8" | sha1sum`)
- Command: `./bin/hashhat --algo sha1 --hash 0554cb9ed835ba8cff065406ad639156c18b8435 --charset lower,num --min-len 4 --max-len 4`
- Results: CPU ≈ 2.06M H/s, GPU ≈ 1.55e9 H/s → ~750x speedup; GPU found `t0m8` quickly (small keyspace).

## Bottlenecks and analysis (SHA-1)

Hashhat is a CUDA-based password brute forcer: threads map indices to candidate strings, hash them on GPU, and compare to a target. We finished the SHA-1 path (CPU+GPU), added a benchmark flag, and ran on Mahti’s A100. In a small benchmark (lowercase, len 3–5, default SHA-1 “aaa”), GPU hit ~33.8M H/s vs CPU ~2.09M (~16x faster). In a mixed charset test (`t0m8`, lower+numbers, len 4), GPU found it quickly at ~1.55B H/s vs CPU ~2.06M. Kernel uses constant memory for target/charset, a simple idx→string mapper, block=256, 10M-candidate chunks. Bottlenecks: length cap at 8, single hash per run, host sync each chunk (no overlap), early exits can inflate H/s on tiny spaces, MD5 not implemented yet. Next steps: add MD5, support longer lengths, overlap launches or tune chunking, allow multiple targets per run, add a small terminal dashboard for progress/ETA.

## Reproduce on Mahti (A100)

```bash
source /appl/profile/zz-csc-env.sh
srun --account=project_2016196 --partition=gpusmall --gres=gpu:a100:1 \
     --time=00:05:00 --ntasks=1 --cpus-per-task=4 --mem=4G \
     bash -lc '
       module load cuda/11.5.0
       cd ~/hashhat
       make cuda
       # Benchmark (Run A)
       ./bin/hashhat --benchmark --algo sha1 --charset lower --min-len 3 --max-len 5
       # Crack t0m8 (Run B)
       ./bin/hashhat --algo sha1 --hash 0554cb9ed835ba8cff065406ad639156c18b8435 \
                     --charset lower,num --min-len 4 --max-len 4
     '
```

## Raw numbers (for reference)

- Run A: CPU 2.09e6 H/s, GPU 3.38e7 H/s, speedup ~16.2x (target SHA-1("aaa"), lower 3–5).
- Run B: CPU 2.06e6 H/s, GPU 1.55e9 H/s, speedup ~750x (target `t0m8`, lower+num len=4).
