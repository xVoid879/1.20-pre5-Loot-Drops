// !nvcc -O3 -arch=sm_75 7.cu -o 7
// Finding back-to-back melon blocks giving 7 melons
#include <cstdio>
#include <cstdint>

#define MASK_64         0xFFFFFFFFFFFFFFFFULL
#define GOLDEN_RATIO_64 0x9e3779b97f4a7c15ULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define SUBTRACT_CONST  0x61C8864680B583EBULL
#define MELON_MD5_0     0x45e0c0d79db027a8ULL
#define MELON_MD5_1     0xafe0a44cafba7e37ULL
#define STAFFORD_MIX_1  0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2  0x94d049bb133111ebULL

#define MIN_SEQUENCE 17  // min it outputs

__device__ inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

__device__ inline uint64_t mix_stafford13(uint64_t seed) {
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1;
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2;
    return seed ^ (seed >> 31);
}

struct LootTableRNG {
    uint64_t seedLo;
    uint64_t seedHi;

    __device__ void set_seed(uint64_t seed) {
        uint64_t l2 = seed ^ SILVER_RATIO_64;
        uint64_t l3 = l2 + -SUBTRACT_CONST;

        seedLo = mix_stafford13(l2 ^ MELON_MD5_0);
        seedHi = mix_stafford13(l3 ^ MELON_MD5_1);

        if ((seedLo | seedHi) == 0) {
            seedLo = GOLDEN_RATIO_64;
            seedHi = SILVER_RATIO_64;
        }
    }

    __device__ uint64_t next_long() {
        uint64_t l = seedLo;
        uint64_t m = seedHi;
        uint64_t n = (rotl64(l + m, 17) + l) & MASK_64;

        m ^= l;
        seedLo = (rotl64(l, 49) ^ m ^ (m << 21)) & MASK_64;
        seedHi = rotl64(m, 28) & MASK_64;

        return n;
    }

    __device__ int next_int(int bound) {
        uint64_t l = next_long() & 0xFFFFFFFF;
        uint64_t m = l * bound;
        uint64_t low = m & 0xFFFFFFFF;
        if (low < (uint64_t)bound) {
            uint64_t t = (-bound) % bound;
            while (low < t) {
                l = next_long() & 0xFFFFFFFF;
                m = l * bound;
                low = m & 0xFFFFFFFF;
            }
        }
        return (int)(m >> 32);
    }

    __device__ int melon_drop() {
        int val = next_int(5) + 3; // (0 to 4) + 3 = drop (think looting makes 9 idk)
        next_long();
        return val;
    }
};

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        LootTableRNG rng;
        rng.set_seed(seed);

        int streak = 0;
        while (rng.melon_drop() == 7) { // change 7 to 3, 4, 5, or 6 if u want to search for that but 7 is the best imo
            streak++;
        }

        if (streak >= MIN_SEQUENCE) {
            printf("Seed %llu with %d\n", seed, streak);
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