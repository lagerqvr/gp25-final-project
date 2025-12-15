#include <iostream>
#include <cstdint>
#include <iomanip>
#include "md5_gpu.cu"


__constant__ char alphabet[27] = "abcdefghijklmnopqrstuvwxyz";
__constant__ std::uint8_t input_hash[16];

__global__ void run_hash(char *found_word, int *found_flag, const int length) {
    long long combinations = 1;
    for (int i = 0; i < length; i++) {
        combinations *= 26;
    }

    unsigned thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned stride = blockDim.x * gridDim.x;

    for (long long i = thread_id; i < combinations; i += stride) {

        long long temp_i = i;
        char guess[64] = {0};
        
        for (int l_index = length - 1; l_index >= 0; l_index--) {
            guess[l_index] = alphabet[temp_i % 26];
            temp_i = temp_i / 26;
        }
        std::uint8_t hash[16];
        hash_md5(guess, length, hash);

        int found = 1;
        for (int j = 0; j < length; j++) {
            if (hash[j] != input_hash[j]) found = 0;
        }
        

        if (found) {
            int prev_flag_value = atomicExch(found_flag, 1);
            if (prev_flag_value == 0) {
                for (int i = 0; i < length; i++) {
                    found_word[i] = guess[i];
                }
            }
        }

        if (*found_flag) {
            return;
        }
    }
}



int main() {
    cudaError_t cuda_ret;

    size_t length = 8;
    std::uint8_t input_hash_h[16] = {
        0xd6, 0x5a, 0x72, 0x6b, 0x3c, 0xd9, 0xd7, 0x93,
        0xa4, 0xa6, 0x7c, 0xae, 0x4a, 0x02, 0xda, 0xf8
    };
    char found_word_h[length + 1] = {0};

    char *found_word_d;
    int *found_flag_d;

    cuda_ret = cudaMalloc(&found_word_d, length * sizeof(char));
    if (cuda_ret != cudaSuccess) {
        std::cout << "Failed to malloc found word array\n";
        return 1;
    }

    cudaDeviceSynchronize();

    cuda_ret = cudaMemcpy(found_word_d, found_word_h, length * sizeof(char), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) {
        std::cout << "Failed memcpy \n";
        cudaFree(found_word_d);
        return 1;
    }

    cudaDeviceSynchronize();

    cuda_ret = cudaMalloc(&found_flag_d, sizeof(int));
    if (cuda_ret != cudaSuccess) {
        std::cout << "Failed to malloc found flag array\n";
        return 1;
    }

    cudaDeviceSynchronize();

    cuda_ret = cudaMemset(found_flag_d, 0, sizeof(int));
    if (cuda_ret != cudaSuccess) {
        std::cout << "Failed to set 0 to flag\n";
        return 1;
    }

    cudaDeviceSynchronize();

    cuda_ret = cudaMemcpyToSymbol(input_hash, input_hash_h, 16 * sizeof(std::uint8_t));
    if (cuda_ret != cudaSuccess) {
        std::cout << "Failed to malloc input hash array\n";
        return 1;
    }

    cudaDeviceSynchronize();

    run_hash<<<80192,128>>>(found_word_d, found_flag_d, length);

    cudaDeviceSynchronize();

    cuda_ret = cudaMemcpy(found_word_h, found_word_d, length * sizeof(char), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) {
        std::cout << "Failed memcpy \n";
        cudaFree(found_word_d);
        return 1;
    }
    cudaDeviceSynchronize();

    cudaFree(found_word_d);
    cudaFree(found_flag_d);

    cudaDeviceSynchronize();

    std::cout << found_word_h << "\n";

    /*std::cout << std::hex << std::setw(2) << std::setfill('0');
    for (const std::uint8_t byte : out_hash_h) {
        std::cout << static_cast<int>(byte);
    }
    std::cout << std::dec << "\n";*/

    return 0;
}