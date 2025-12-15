#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}




const uint32_t s[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
};

const uint32_t K[64] = {
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


void pad_input(const char *input, uint8_t **padded_input, size_t *padded_length) {

    size_t input_length = strlen(input);
    size_t add_1_length = input_length + 1;

    size_t z_padding;
    size_t modulo = add_1_length % 64;
    if (modulo <= 56) {
        z_padding = 56 - modulo;
    }
    else {
        z_padding = 64 + 56 - modulo;
    }

    uint64_t input_len_in_bits = (uint64_t)input_length * 8;
    size_t bits_length = 8;


    size_t z_padded_length = add_1_length + z_padding;
    *padded_length = add_1_length + z_padding + bits_length;
    *padded_input = (uint8_t*)calloc(*padded_length, 1);
    if (*padded_input == NULL) {
        perror("calloc()");
        *padded_length = 0;
        return;
    }
    memcpy(*padded_input, input, input_length);
    (*padded_input)[input_length] = 0x80;
    for (int i = 0; i < 8; i++) {
        (*padded_input)[z_padded_length + i] = (uint8_t)(input_len_in_bits >> (i * 8));
    }
}

void process_chunks(uint8_t *padded_input, const size_t padded_length, uint32_t *a0, uint32_t *b0, uint32_t *c0, uint32_t *d0)
{
    for (int chunk_offset = 0; chunk_offset < padded_length; chunk_offset += 64) {
        uint8_t *chunk = padded_input + chunk_offset;
        uint32_t M[16] = {0};
        for (int i=0; i < 16; i++) {
            M[i] = (chunk[i * 4 + 3] << 24) | (chunk[i * 4 + 2] << 16) |
                    (chunk[i * 4 + 1] << 8) | (chunk[i * 4]);
        }

        uint32_t A = *a0;
        uint32_t B = *b0;
        uint32_t C = *c0;
        uint32_t D = *d0;
        
        for (int i = 0; i < 64; i++) {
            uint32_t F;
            uint32_t g;
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
            B = B + ((F << s[i]) | (F >> (32 - s[i])));
        }
        *a0 += A;
        *b0 += B;
        *c0 += C;
        *d0 += D;
    }
}

void append_digest(uint8_t digest[16], uint32_t a0, uint32_t b0, uint32_t c0, uint32_t d0)
{
    for (int i = 0; i < 4; i++) {
        digest[i] = (a0 >> (8 * i)) & 0xFF;
        digest[4 + i] = (b0 >> (8 * i)) & 0xFF;
        digest[8 + i] = (c0 >> (8 * i)) & 0xFF;
        digest[12 + i] = (d0 >> (8 * i)) & 0xFF;
    }
}

void cpu_md5(const char *input, uint8_t digest[16])
{
    uint8_t *padded_input = NULL;
    size_t padded_length = 0;
    pad_input(input, &padded_input, &padded_length);
    if (padded_input == NULL) {
        return;
    }
    uint32_t a0 = 0x67452301;
    uint32_t b0 = 0xefcdab89;
    uint32_t c0 = 0x98badcfe;
    uint32_t d0 = 0x10325476;
    process_chunks(padded_input, padded_length, &a0, &b0, &c0, &d0);
    append_digest(digest, a0, b0, c0, d0);
    free(padded_input);
    padded_input = NULL;
}

void hash_to_hex(const uint8_t digest[16], char hex_string[33]) {
    for (int i = 0; i < 16; i++) {
        sprintf(hex_string + i*2, "%02x", digest[i]);
    }
    hex_string[32] = '\0';
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("arg1 = input word\n");
        printf("arg2 = expected hash\n");
        return 1;
    }

    Timer timer;

    char *input = argv[1];
    char *expected_hash = argv[2];
    char calculated_hash[33] = {0};

    uint8_t digest[16] = {0};
    
    startTime(&timer);
    cpu_md5(input, digest);
    stopTime(&timer);
    float time = elapsedTime(timer);

    hash_to_hex(digest, calculated_hash);

    printf("\n----------------------------------------------------------\n");
    printf("Input word:      %s\n", input);
    printf("Expected hash:   %s\n", expected_hash);
    printf("Calculated hash: %s\n\n", calculated_hash);

    if (strcmp(expected_hash, calculated_hash) == 0) {
        printf("\033[32mHashes match!\033[0m\n");
    } 
    else {
        printf("\033[31mHashes do not match\033[0m\n");
    }

    printf("\nTime elapsed: %f s\n", time);

    printf("----------------------------------------------------------\n");

    return 0;
}