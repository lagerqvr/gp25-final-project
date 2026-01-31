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

## MD4 and NTLM benchmarking

device=NVIDIA A100-SXM4-40GB MIG 1g.5gb, len=4, charset=36, target=2947d49576676f9aaa39d830c756effd, block=256, grid_first=6561, chunk=10000000
Date: 2026-01-11T19:37:37Z
Charset/len: lower,num / 3-5
CPU H/s: 2185361.829982
GPU H/s: 2276244550.927892
Speedup (GPU/CPU): 1041.59x
Notes: found=t0m8

device=NVIDIA A100-SXM4-40GB MIG 1g.5gb, len=4, charset=36, target=6f0142ff3493987cf859ed183411d969, block=256, grid_first=6561, chunk=10000000
Date: 2026-01-11T19:42:09Z
Charset/len: lower,num / 3-5
CPU H/s: 1652830.685992
GPU H/s: 2256812512.746498
Speedup (GPU/CPU): 1365.42x
Notes: found=t0m8

## MD4 and NTLM analysis

Looking at the case above where the GPU is fully utlizied with the password t0m8, we can see that the NTLM case is faster than the md4, even if the NTLM requires an extra step for encoding. This is because the UTF16-LE encoding sorts bytes to be more accessible for the GPU, resulting in a faster throughput. In a case with an easier password, the overhead of the kernel is high and both MD4 and NTLM have a lower speedup, but still shows the trend of the NTLM being faster. 


## MD5 benchmarks

To test the implementation of the MD5 algorithm, the same benchmarking tests were performed. Differing from the other benchmarks, the MD5 algorithm was
tested locally to get some variation to the tests and results.

### Benchmark run

- Device:  NVIDIA GeForce RTX 4070 Ti SUPER

         
Simple benchmark, target `aaa`.

- Command: `./bin/hashhat --benchmark --algo md5 --charset lower --min-len 3 --max-len 5`
- CPU H/S: **658328**
- GPU H/S: **2.18986e+07**
- Speedup: **33.2639x**


Benchmark for the target `t0m8`.

- Command: `./bin/hashhat --algo md5 --hash 76f5237d8f33341b5d686bae8631d65e --charset lower,num --min-len 4 --max-len 4`
- CPU H/S: **5.05406e+06**
- GPU H/S: **2.81819e+09**
- Speedup: **~550x**

### Analysis

As anticipated, the GPU performs better in the benchmarks compared to the CPU. We can see an increase in hashes per second
for both the CPU and GPU when the complexity of the target increases. This is most likely due to the fact that the word
`aaa` is the first combination and most of the execution time is spent on setting up the computation. Once we move to a
more computational task, such as cracking the second hash, we spend much more time doing work, reducing the percentage of
overhead.

An improvement to the implementation that was made to try to reduce the computations per hash was to replace the calculations
for the variable `g` with a table with pre-calculated values. \
What was done in practice was:

- (5 * i + 1) % 16 -> d_g[i]
- (3 * i + 5) % 16 -> d_g[i]

By making this small change, the H/s, averaging **~1.6e9** for the target `t0m8` was increased to averaging over **2.0e9** H/s.
