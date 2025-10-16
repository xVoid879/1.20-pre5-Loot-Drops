// !nvcc -O3 -arch=sm_75 unluckygun.cu -o unluckygun
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define MASK_64         0xFFFFFFFFFFFFFFFFULL

#define GOLDEN_RATIO_64 0x9e3779b97f4a7c15ULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define SUBTRACT_CONSTANT 0x61C8864680B583EBULL
#define CREEPER_MD5_0   0x6863479bde978baeULL
#define CREEPER_MD5_1   0xea09ca04385aacb4ULL
#define STAFFORD_MIX_1  0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2  0x94d049bb133111ebULL
#define MIN_ZEROS       23   // min it outputs

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int r) {
    return ((x << r) & MASK_64) | (x >> (64 - r));
}

__device__ __forceinline__ uint64_t mix_stafford13(uint64_t seed) {
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64;
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64;
    seed ^= seed >> 31;
    return seed & MASK_64;
}

struct Xoroshiro128Plus {
    uint64_t l, h;

    __device__ __forceinline__
    Xoroshiro128Plus(uint64_t seed) {
        l = seed ^ SILVER_RATIO_64;
        h = l - SUBTRACT_CONSTANT;
        l ^= CREEPER_MD5_0;
        h ^= CREEPER_MD5_1;
        l = mix_stafford13(l);
        h = mix_stafford13(h);
        if ((l | h) == 0ULL) {
            l = GOLDEN_RATIO_64;
            h = SILVER_RATIO_64;
        }
        this->l = l;
        this->h = h;
    }

    __device__ __forceinline__
    uint64_t nextLong() {
        uint64_t n = rotl64((l + h) & MASK_64, 17) + l;
        h ^= l;
        uint64_t l_new = (rotl64(l, 49) ^ h ^ ((h << 21) & MASK_64)) & MASK_64;
        uint64_t h_new = rotl64(h, 28) & MASK_64;
        l = l_new;
        h = h_new;
        return n;
    }
};

__device__ __forceinline__
int java_next_int(Xoroshiro128Plus &rng, int bound) {
    while (true) {
        int32_t i = (int32_t)(rng.nextLong() & 0xFFFFFFFFULL);
        uint64_t j = (uint64_t)(uint32_t)i * (uint64_t)bound;
        uint32_t k = (uint32_t)j;
        if (k < (uint32_t)bound) {
            uint32_t l = (-(uint32_t)bound) % (uint32_t)bound;
            while (k < l) {
                i = (int32_t)(rng.nextLong() & 0xFFFFFFFFULL);
                j = (uint64_t)(uint32_t)i * (uint64_t)bound;
                k = (uint32_t)j;
            }
        }
        return (int)((j >> 32) & 0xFFFFFFFFULL);
    }
}

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        Xoroshiro128Plus rng(seed);
        int curStreak = 0;

        while (true) {
            int drop = java_next_int(rng, 3); // 0 is no
            if (drop == 0) {
                curStreak++;
            } else {
                break;
            }
        }

        if (curStreak >= MIN_ZEROS) {
            printf("Seed %llu with %d\n", seed, curStreak);
        }
    }
}

int main() {
    uint64_t start = 0ULL;
    uint64_t end   = 1000000000000ULL;

    int threads = 256;
    int blocks  = 256;

    search<<<blocks, threads>>>(start, end);
    cudaDeviceSynchronize();
    return 0;
}
