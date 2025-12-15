#include <cstdint>

__constant__ std::uint32_t K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

__constant__ std::uint32_t S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
};

inline __device__ void pad_input(const char *input, std::size_t length, std::uint8_t *padded_input) {
    int i;
    for (i = 0; i < length; i++) {
        padded_input[i] = (std::uint8_t)input[i];
    }
    padded_input[length] = 0x80;
    std::uint64_t length_in_bits = (std::uint64_t)length * 8;
    for (i = 0; i < 8; i++) {
        // Storing length in bits as big endian
        padded_input[56 + i] = (std::uint8_t)(length_in_bits >> (i * 8));
    }
}

inline __device__ void process_chunk(std::uint8_t *padded_input, std::uint32_t *a0, std::uint32_t *b0, std::uint32_t *c0, std::uint32_t *d0) {
    std::uint32_t M[16] = {0};
    for (int i=0; i < 16; i++) {
        // reversing order from little endian to big endian
        M[i] = (padded_input[i * 4 + 3] << 24) | (padded_input[i * 4 + 2] << 16) |
                (padded_input[i * 4 + 1] << 8) | (padded_input[i * 4]);
    }
    std::uint32_t A = *a0;
    std::uint32_t B = *b0;
    std::uint32_t C = *c0;
    std::uint32_t D = *d0;
    
    for (int i = 0; i < 64; i++) {
        std::uint32_t F;
        std::uint32_t g;
        if (i < 16) {
            F = (B & C) | ((~B) & D);
            g = i;
        }
        else if (i < 32) {
            F = (D & B) | ((~D) & C);
            g = (5 * i + 1) % 16;
        }
        else if (i < 48) {
            F = B ^ C ^ D;
            g = (3 * i + 5) % 16;
        }
        else {
            F = C ^ (B | (~D));
            g = (7 * i) % 16;
        }
        F = F + A + K[i] + M[g];
        A = D;
        D = C;
        C = B;
        B = B + ((F << S[i]) | (F >> (32 - S[i])));
    }
    *a0 += A;
    *b0 += B;
    *c0 += C;
    *d0 += D;
}




inline __device__ void hash_md5(const char *input, const std::size_t input_length, std::uint8_t *hash) {
    std::uint32_t a0 = 0x67452301;
    std::uint32_t b0 = 0xefcdab89;
    std::uint32_t c0 = 0x98badcfe;
    std::uint32_t d0 = 0x10325476;

    std::uint8_t padded_input[64] = {0};
    pad_input(input, input_length, padded_input);
    // Assuming a password cannot exceed 55 chars, we only
    // process one chunk of 64 bytes.
    process_chunk(padded_input, &a0, &b0, &c0, &d0);

    // At this stage a0, b0, c0 and d0 should contain the hash uint8s
    // The hash being 16 bytes so 4 bytes in a0, 4bytes in b0 etc

    // Stores bytes in little endian
    for (int i = 0; i < 4; i++) {
        hash[i] = ((a0 >> (8 * i)) & 0xff) ;
        hash[i + 4] = ((b0 >> (8 * i)) & 0xff);
        hash[i + 8] = ((c0 >> (8 * i)) & 0xff);
        hash[i + 12] = ((d0 >> (8 * i)) & 0xff);
    }

}


