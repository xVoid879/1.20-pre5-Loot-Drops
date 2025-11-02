// 1nvcc -O3 -arch=sm_75 backwheat.cu -o backwheat

#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define GOLDEN_RATIO_64  0x9e3779b97f4a7c15ULL
#define SILVER_RATIO_64  0x6a09e667f3bcc909ULL
#define SHORT_GRASS_MD5_0 0x8bbf3c7183e5ac1eULL
#define SHORT_GRASS_MD5_1 0x0e588cdc9bfba209ULL
#define STAFFORD_MIX_1   0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2   0x94d049bb133111ebULL
#define SHORT_GRASS_CONST 2097152ULL
#define MIN_ZEROES 13 // min to output

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

struct Xoroshiro128PlusPlus {
    uint64_t l, h;

    __device__ __forceinline__
    Xoroshiro128Plus(uint64_t seed) {
        l = seed ^ SILVER_RATIO_64;
        h = l + GOLDEN_RATIO_64;
        l ^= SHORT_GRASS_MD5_0;
        h ^= SHORT_GRASS_MD5_1;

        l = ((l ^ (l >> 30)) * STAFFORD_MIX_1);
        h = ((h ^ (h >> 30)) * STAFFORD_MIX_1);
        l = ((l ^ (l >> 27)) * STAFFORD_MIX_2);
        h = ((h ^ (h >> 27)) * STAFFORD_MIX_2);
        l ^= (l >> 31);
        h ^= (h >> 31);
    }

    __device__ __forceinline__
    uint64_t next_long() {
        uint64_t n = (rotl64((l + h), 17) + l);
        h ^= l;
        uint64_t l_new = (rotl64(l, 49) ^ h ^ (h << 21));
        uint64_t h_new = rotl64(h, 28);
        l = l_new;
        h = h_new;
        return n;
    }
};

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        Xoroshiro128Plus rng(seed);

        int streak = 0;
        while (true) {
            uint64_t n = rng.next_long();
            if ((n >> 40) < SHORT_GRASS_CONST) {
                streak++;
                rng.next_long();
            } else {
                break;
            }
        }

        if (streak >= MIN_ZEROES) {
            printf("Seed %llu with %d\n", seed, streak);
        }
    }
}

int main() {
    uint64_t start = 10000000000000ULL;
    uint64_t end   = 10000000000001ULL;

    int threads = 256;
    int blocks  = 256;

    search<<<blocks, threads>>>(start, end);
    cudaDeviceSynchronize();

    return 0;
}
