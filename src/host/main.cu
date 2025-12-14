#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "common/options.h"

Options parse_args(int argc, char **argv)
{
    Options opts;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto consume_value = [&](const std::string &key) -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value for " + key);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--help" || arg == "-h")
        {
            opts.show_help = true;
        }
        else if (arg == "--benchmark")
        {
            opts.run_benchmark = true;
        }
        else if (arg == "--hash")
        {
            opts.hash_hex = consume_value(arg);
        }
        else if (arg == "--algo")
        {
            opts.algo = consume_value(arg);
        }
        else if (arg == "--charset")
        {
            opts.charset = consume_value(arg);
        }
        else if (arg == "--min-len")
        {
            opts.min_len = std::stoi(consume_value(arg));
        }
        else if (arg == "--max-len")
        {
            opts.max_len = std::stoi(consume_value(arg));
        }
        else if (arg == "--ui")
        {
            opts.ui_mode = consume_value(arg);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return opts;
}

void print_ascii_banner()
{
    std::cout << R"(
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
)" << "\n";
}

void print_help()
{
    print_ascii_banner();
    std::cout << "Hashhat - CUDA GPU Password Cracker (CLI scaffold)\n\n";
    std::cout << "Usage: hashhat [--hash <hex>] [--algo sha1|md5] [--charset list] [--min-len N] [--max-len N] [--ui curses|none]\n";
    std::cout << "       hashhat --benchmark\n";
    std::cout << "\n";
    std::cout << "Flags (current placeholders):\n";
    std::cout << "  --hash <hex>        Target hash in hex (MD5 initially)\n";
    std::cout << "  --algo name         sha1 (now) | md5 (WIP)\n";
    std::cout << "  --charset list      Comma list: lower,upper,num,sym (default: lower)\n";
    std::cout << "  --min-len N         Minimum password length (default: 1)\n";
    std::cout << "  --max-len N         Maximum password length (default: 1)\n";
    std::cout << "  --ui <mode>         Optional UI mode: none|curses (default: none)\n";
    std::cout << "  --benchmark         Run benchmark harness (CPU vs GPU), writes summary to results/ANALYSIS.md\n";
    std::cout << "  --help              Show this help\n";
    std::cout << "\n";
    std::cout << "Roadmap hooks:\n";
    std::cout << "  - TODO: wire GPU kernel launch here\n";
    std::cout << "  - TODO: add CPU baseline and correctness checks\n";
    std::cout << "  - TODO: integrate curses UI when --ui=curses\n";
    std::cout << "  - TODO: implement benchmark harness (CPU vs GPU timing)\n";
}

void print_options(const Options &o)
{
    std::cout << "Hashhat options:\n";
    std::cout << "  hash:     " << (o.hash_hex.empty() ? "(none)" : o.hash_hex) << "\n";
    std::cout << "  algo:     " << o.algo << "\n";
    std::cout << "  charset:  " << o.charset << "\n";
    std::cout << "  min-len:  " << o.min_len << "\n";
    std::cout << "  max-len:  " << o.max_len << "\n";
    std::cout << "  ui:       " << o.ui_mode << "\n";
    std::cout << "  bench:    " << (o.run_benchmark ? "yes" : "no") << "\n";
}

std::optional<double> run_gpu_baseline(const Options &);

std::string now_iso8601()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

void append_analysis_entry(const Options &o, const std::optional<double> &cpu_hps, const std::optional<double> &gpu_hps)
{
    std::ofstream out("results/ANALYSIS.md", std::ios::app);
    if (!out)
    {
        std::cerr << "[benchmark] Could not open results/ANALYSIS.md for append.\n";
        return;
    }

    out << "\n---\n";
    out << "Date: " << now_iso8601() << "\n";
    out << "Charset/len: " << o.charset << " / " << o.min_len << "-" << o.max_len << "\n";
    out << "CPU H/s: " << (cpu_hps ? std::to_string(*cpu_hps) : std::string("N/A")) << "\n";
    out << "GPU H/s: " << (gpu_hps ? std::to_string(*gpu_hps) : std::string("N/A")) << "\n";

    if (cpu_hps && gpu_hps && *cpu_hps > 0)
    {
        double speedup = *gpu_hps / *cpu_hps;
        out << "Speedup (GPU/CPU): " << speedup << "x\n";
    }
    else
    {
        out << "Speedup (GPU/CPU): N/A\n";
    }

    out << "Notes: auto-generated from --benchmark (fill in HW details and kernel config).\n";
}

// SHA-1 implementation for CPU baseline
namespace sha1
{
    inline uint32_t rotl(uint32_t value, uint32_t bits)
    {
        return (value << bits) | (value >> (32 - bits));
    }

    std::vector<uint8_t> hash(const std::vector<uint8_t> &data)
    {
        uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8;
        std::vector<uint8_t> msg(data);
        msg.push_back(0x80);
        while ((msg.size() % 64) != 56)
        {
            msg.push_back(0x00);
        }
        for (int i = 7; i >= 0; --i)
        {
            msg.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFF));
        }

        uint32_t h0 = 0x67452301;
        uint32_t h1 = 0xEFCDAB89;
        uint32_t h2 = 0x98BADCFE;
        uint32_t h3 = 0x10325476;
        uint32_t h4 = 0xC3D2E1F0;

        for (size_t chunk = 0; chunk < msg.size(); chunk += 64)
        {
            uint32_t w[80];
            for (int i = 0; i < 16; ++i)
            {
                w[i] = (msg[chunk + 4 * i] << 24) | (msg[chunk + 4 * i + 1] << 16) |
                       (msg[chunk + 4 * i + 2] << 8) | (msg[chunk + 4 * i + 3]);
            }
            for (int i = 16; i < 80; ++i)
            {
                w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
            }

            uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
            for (int i = 0; i < 80; ++i)
            {
                uint32_t f, k;
                if (i < 20)
                {
                    f = (b & c) | ((~b) & d);
                    k = 0x5A827999;
                }
                else if (i < 40)
                {
                    f = b ^ c ^ d;
                    k = 0x6ED9EBA1;
                }
                else if (i < 60)
                {
                    f = (b & c) | (b & d) | (c & d);
                    k = 0x8F1BBCDC;
                }
                else
                {
                    f = b ^ c ^ d;
                    k = 0xCA62C1D6;
                }
                uint32_t temp = rotl(a, 5) + f + e + k + w[i];
                e = d;
                d = c;
                c = rotl(b, 30);
                b = a;
                a = temp;
            }

            h0 += a;
            h1 += b;
            h2 += c;
            h3 += d;
            h4 += e;
        }

        std::vector<uint8_t> digest(20);
        uint32_t hs[5] = {h0, h1, h2, h3, h4};
        for (int i = 0; i < 5; ++i)
        {
            digest[4 * i] = (hs[i] >> 24) & 0xFF;
            digest[4 * i + 1] = (hs[i] >> 16) & 0xFF;
            digest[4 * i + 2] = (hs[i] >> 8) & 0xFF;
            digest[4 * i + 3] = hs[i] & 0xFF;
        }
        return digest;
    }
} // namespace sha1

std::vector<uint8_t> hex_to_bytes(const std::string &hex)
{
    if (hex.size() % 2 != 0)
        return {};
    std::vector<uint8_t> out(hex.size() / 2);
    auto nibble = [](char c) -> int
    {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < out.size(); ++i)
    {
        int hi = nibble(hex[2 * i]);
        int lo = nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0)
            return {};
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return out;
}

std::string bytes_to_hex(const std::vector<uint8_t> &bytes)
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint8_t b : bytes)
    {
        oss << std::setw(2) << static_cast<int>(b);
    }
    return oss.str();
}

std::string build_charset(const std::string &spec)
{
    std::string charset;
    std::istringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (token == "lower")
            charset += "abcdefghijklmnopqrstuvwxyz";
        else if (token == "upper")
            charset += "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        else if (token == "num" || token == "digit")
            charset += "0123456789";
        else if (token == "sym" || token == "symbols")
            charset += "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        else if (!token.empty())
            charset += token; // treat as literal characters
    }
    if (charset.empty())
        charset = "abcdefghijklmnopqrstuvwxyz";
    return charset;
}

struct BenchmarkResult
{
    double hashes_per_second;
    size_t candidates_tested;
    bool found{false};
    std::string found_word;
    std::string target_hex;
    double seconds{0.0};
};

BenchmarkResult run_cpu_bruteforce_sha1(const Options &opts)
{
    BenchmarkResult res{0.0, 0, false, ""};
    const std::string charset = build_charset(opts.charset);
    const size_t charset_len = charset.size();
    if (charset_len == 0)
        return res;

    const size_t max_work = 5'000'000; // cap to keep CPU benchmark quick
    int eff_min = opts.min_len;
    int eff_max = opts.max_len;

    auto total_for_len = [&](int len) -> size_t
    {
        size_t total = 1;
        for (int i = 0; i < len; ++i)
            total *= charset_len;
        return total;
    };

    size_t planned = 0;
    for (int l = eff_min; l <= eff_max; ++l)
    {
        planned += total_for_len(l);
    }
    while (planned > max_work && eff_max > eff_min)
    {
        planned -= total_for_len(eff_max);
        --eff_max;
    }

    std::vector<uint8_t> target_bytes;
    std::string target_hex = opts.hash_hex;
    if (!opts.hash_hex.empty())
        target_bytes = hex_to_bytes(opts.hash_hex);
    if (target_bytes.empty())
    {
        std::vector<uint8_t> def{'a', 'a', 'a'};
        target_bytes = sha1::hash(def);
        target_hex = bytes_to_hex(target_bytes);
    }
    if (target_bytes.size() != 20)
    {
        std::cout << "[benchmark][cpu] invalid target hash length\n";
        return res;
    }

    auto start = std::chrono::steady_clock::now();
    std::string candidate;

    std::function<void(int)> gen = [&](int remaining)
    {
        if (remaining == 0)
        {
            std::vector<uint8_t> bytes(candidate.begin(), candidate.end());
            auto digest = sha1::hash(bytes);
            ++res.candidates_tested;
            if (!target_bytes.empty() && digest == target_bytes)
            {
                res.found = true;
                res.found_word = candidate;
            }
            return;
        }
        for (char c : charset)
        {
            candidate.push_back(c);
            gen(remaining - 1);
            candidate.pop_back();
            if (res.found)
                return;
        }
    };

    for (int len = eff_min; len <= eff_max && !res.found; ++len)
    {
        candidate.clear();
        gen(len);
    }

    auto end = std::chrono::steady_clock::now();
    res.seconds = std::chrono::duration<double>(end - start).count();
    res.target_hex = target_hex;
    if (res.seconds > 0.0)
        res.hashes_per_second = static_cast<double>(res.candidates_tested) / res.seconds;

    std::cout << "[benchmark][cpu] charset=" << charset_len << " chars, len " << eff_min << "-" << eff_max
              << ", tested=" << res.candidates_tested
              << ", found=" << (res.found ? "yes" : "no")
              << ", H/s=" << res.hashes_per_second
              << ", target=" << res.target_hex << "\n";
    if (res.found)
    {
        std::cout << "[benchmark][cpu] found word: " << res.found_word << "\n";
    }

    if (planned > max_work)
    {
        std::cout << "[benchmark][cpu] capped workload to " << res.candidates_tested << " candidates to stay fast.\n";
    }

    return res;
}

std::optional<double> run_cpu_baseline(const Options &opts)
{
    if (opts.algo != "sha1")
    {
        std::cout << "[benchmark][cpu] algo=" << opts.algo << " not implemented (MD5 pending; hook here for Max).\n";
        return std::nullopt;
    }
    BenchmarkResult r = run_cpu_bruteforce_sha1(opts);
    if (r.candidates_tested == 0)
        return std::nullopt;
    return r.hashes_per_second;
}

#ifdef __CUDACC__

#define CUDA_CHECK(x)                              \
    do                                             \
    {                                              \
        cudaError_t err__ = (x);                   \
        if (err__ != cudaSuccess)                  \
        {                                          \
            std::cerr << "CUDA error: "            \
                      << cudaGetErrorString(err__) \
                      << " at " << __FILE__ << ":" \
                      << __LINE__ << "\n";         \
            return std::nullopt;                   \
        }                                          \
    } while (0)

__constant__ uint8_t d_target[20];
__constant__ char d_charset[128];
__device__ int d_charset_len;

__device__ inline uint32_t rotl_dev(uint32_t v, uint32_t bits)
{
    return (v << bits) | (v >> (32 - bits));
}

__device__ void sha1_device(const char *msg, int len, uint8_t out[20])
{
    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
    uint8_t block[64] = {0};

    for (int i = 0; i < len; ++i)
        block[i] = static_cast<uint8_t>(msg[i]);
    block[len] = 0x80;
    block[63] = static_cast<uint8_t>(bit_len & 0xFF);
    block[62] = static_cast<uint8_t>((bit_len >> 8) & 0xFF);
    block[61] = static_cast<uint8_t>((bit_len >> 16) & 0xFF);
    block[60] = static_cast<uint8_t>((bit_len >> 24) & 0xFF);

    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
    {
        w[i] = (block[4 * i] << 24) | (block[4 * i + 1] << 16) | (block[4 * i + 2] << 8) | block[4 * i + 3];
    }
    for (int i = 16; i < 80; ++i)
    {
        w[i] = rotl_dev(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    uint32_t a = 0x67452301;
    uint32_t b = 0xEFCDAB89;
    uint32_t c = 0x98BADCFE;
    uint32_t d = 0x10325476;
    uint32_t e = 0xC3D2E1F0;

    for (int i = 0; i < 80; ++i)
    {
        uint32_t f, k;
        if (i < 20)
        {
            f = (b & c) | ((~b) & d);
            k = 0x5A827999;
        }
        else if (i < 40)
        {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        }
        else if (i < 60)
        {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        }
        else
        {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }
        uint32_t temp = rotl_dev(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = rotl_dev(b, 30);
        b = a;
        a = temp;
    }

    uint32_t hs[5] = {a + 0x67452301, b + 0xEFCDAB89, c + 0x98BADCFE, d + 0x10325476, e + 0xC3D2E1F0};
    for (int i = 0; i < 5; ++i)
    {
        out[4 * i + 0] = (hs[i] >> 24) & 0xFF;
        out[4 * i + 1] = (hs[i] >> 16) & 0xFF;
        out[4 * i + 2] = (hs[i] >> 8) & 0xFF;
        out[4 * i + 3] = hs[i] & 0xFF;
    }
}

__device__ void idx_to_string(uint64_t idx, int len, char *out)
{
    int base = d_charset_len;
    for (int pos = len - 1; pos >= 0; --pos)
    {
        out[pos] = d_charset[idx % base];
        idx /= base;
    }
}

__global__ void sha1_bruteforce_kernel(uint64_t start_idx, uint64_t total, int len, char *found_word, int *found_flag)
{
    uint64_t gid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x + start_idx;
    if (gid >= total || *found_flag)
        return;

    char candidate[8]; // supports lengths up to 8
    idx_to_string(gid, len, candidate);

    uint8_t digest[20];
    sha1_device(candidate, len, digest);

    bool match = true;
    for (int i = 0; i < 20; ++i)
    {
        if (digest[i] != d_target[i])
        {
            match = false;
            break;
        }
    }
    if (match)
    {
        if (!atomicExch(found_flag, 1))
        {
            for (int i = 0; i < len; ++i)
            {
                found_word[i] = candidate[i];
            }
        }
    }
}

std::optional<double> run_gpu_baseline(const Options &opts)
{
    if (opts.algo != "sha1")
    {
        std::cout << "[benchmark][gpu] algo=" << opts.algo << " not implemented (MD5 pending; slot for Max).\n";
        return std::nullopt;
    }
    const std::string charset = build_charset(opts.charset);
    const int charset_len = static_cast<int>(charset.size());
    if (charset_len == 0)
    {
        std::cout << "[benchmark][gpu] empty charset\n";
        return std::nullopt;
    }

    if (opts.max_len > 8)
    {
        std::cout << "[benchmark][gpu] max_len > 8 not supported in simple kernel; capping to 8.\n";
    }
    int eff_min = std::max(1, opts.min_len);
    int eff_max = std::min(8, opts.max_len);

    std::vector<uint8_t> target_bytes;
    if (!opts.hash_hex.empty())
        target_bytes = hex_to_bytes(opts.hash_hex);
    if (target_bytes.empty())
    {
        std::vector<uint8_t> def{'a', 'a', 'a'};
        target_bytes = sha1::hash(def);
    }
    std::string target_hex = bytes_to_hex(target_bytes);
    if (target_bytes.size() != 20)
    {
        std::cout << "[benchmark][gpu] invalid target hash length\n";
        return std::nullopt;
    }

    int device = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    CUDA_CHECK(cudaMemcpyToSymbol(d_charset, charset.data(), charset_len));
    CUDA_CHECK(cudaMemcpyToSymbol(d_charset_len, &charset_len, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_target, target_bytes.data(), 20));

    char *d_found_word = nullptr;
    int *d_found_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_found_word, 8));
    CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));
    int false_flag = 0;
    CUDA_CHECK(cudaMemcpy(d_found_flag, &false_flag, sizeof(int), cudaMemcpyHostToDevice));

    const int threads = 256;
    const uint64_t chunk = 10'000'000ULL;
    uint64_t total_tested = 0;
    bool found = false;
    std::string found_word;

    auto start = std::chrono::steady_clock::now();
    for (int len = eff_min; len <= eff_max && !found; ++len)
    {
        uint64_t total = 1;
        for (int i = 0; i < len; ++i)
            total *= static_cast<uint64_t>(charset_len);

        uint64_t first_work = std::min(chunk, total);
        unsigned int grid_first = static_cast<unsigned int>((first_work + threads - 1) / threads);
        std::cout << "[benchmark][gpu] device=" << prop.name
                  << ", len=" << len << ", charset=" << charset_len
                  << ", target=" << target_hex
                  << ", block=" << threads << ", grid_first=" << grid_first
                  << ", chunk=" << chunk << "\n";

        for (uint64_t offset = 0; offset < total && !found; offset += chunk)
        {
            uint64_t work = std::min(chunk, total - offset);
            dim3 block(threads);
            dim3 grid(static_cast<unsigned int>((work + threads - 1) / threads));
            sha1_bruteforce_kernel<<<grid, block>>>(offset, total, len, d_found_word, d_found_flag);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int host_found = 0;
            CUDA_CHECK(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost));
            total_tested += work;

            if (host_found)
            {
                found = true;
                char buf[8] = {0};
                CUDA_CHECK(cudaMemcpy(buf, d_found_word, len, cudaMemcpyDeviceToHost));
                found_word.assign(buf, buf + len);
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    cudaFree(d_found_word);
    cudaFree(d_found_flag);

    if (!found)
    {
        std::cout << "[benchmark][gpu] target not found in searched space.\n";
    }
    else
    {
        std::cout << "[benchmark][gpu] found=" << found_word << ", target=" << target_hex << "\n";
    }

    if (seconds <= 0.0 || total_tested == 0)
        return std::nullopt;

    double hps = static_cast<double>(total_tested) / seconds;
    std::cout << "[benchmark][gpu] tested=" << total_tested << " in " << seconds << "s, H/s=" << hps << "\n";
    return hps;
}

#else
std::optional<double> run_gpu_baseline(const Options &opts)
{
    std::cout << "[benchmark][gpu] CUDA build required (algo=" << opts.algo << ").\n";
    return std::nullopt;
}
#endif

void run_benchmark(const Options &opts)
{
    std::cout << "\n[benchmark] Starting benchmark harness...\n";
    auto cpu_hps = run_cpu_baseline(opts);
    auto gpu_hps = run_gpu_baseline(opts);

    if (cpu_hps && gpu_hps && *cpu_hps > 0)
    {
        double speedup = *gpu_hps / *cpu_hps;
        std::cout << "[benchmark] CPU: " << *cpu_hps << " H/s, GPU: " << *gpu_hps << " H/s, speedup: " << speedup << "x\n";
    }
    else
    {
        std::cout << "[benchmark] Missing CPU/GPU implementations or search exhausted.\n";
    }

    append_analysis_entry(opts, cpu_hps, gpu_hps);
    std::cout << "[benchmark] Wrote summary to results/ANALYSIS.md\n\n";
}
int main(int argc, char **argv)
{
    try
    {
        Options opts = parse_args(argc, argv);

        if (opts.show_help)
        {
            print_help();
            return 0;
        }

        if (opts.min_len > opts.max_len)
        {
            throw std::runtime_error("min-len cannot exceed max-len");
        }

        print_options(opts);

        if (opts.run_benchmark)
        {
            run_benchmark(opts);
            return 0;
        }

        if (!opts.hash_hex.empty())
        {
            std::cout << "[crack] Starting CPU + GPU search for provided hash.\n";
            auto cpu_hps = run_cpu_baseline(opts);
            auto gpu_hps = run_gpu_baseline(opts);
            if (cpu_hps)
                std::cout << "[crack][cpu] H/s=" << *cpu_hps << "\n";
            if (gpu_hps)
                std::cout << "[crack][gpu] H/s=" << *gpu_hps << "\n";
            return 0;
        }

        // TODO 3: Integrate curses dashboard when opts.ui_mode == "curses".
        // TODO 4: Add configurable hash algorithm selection (MD5 now, SHA variants later).
        // TODO 5: Move charset/hash data to constant memory and measure speedups.
        // TODO 6: Add test hooks for small keyspaces to validate end-to-end.

        std::cout << "\n[stub] No cracking implemented yet. See TODOs in source.\n";
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        std::cerr << "Use --help for usage.\n";
        return 1;
    }
}
