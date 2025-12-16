// !nvcc -O3 -arch=sm_75 4.cu -o 4
#include <cstdio>
#include <cstdint>

#define MASK_64 0xFFFFFFFFFFFFFFFFULL

#define GOLDEN_RATIO_64 0x9e3779b97f4a7c15ULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define SUBTRACT_CONST 0x61C8864680B583EBULL
#define GLOWSTONE_MD5_0 0x83196c8056477ab2ULL
#define GLOWSTONE_MD5_1 0x41f5812fc482f9acULL
#define STAFFORD_MIX_1 0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2 0x94d049bb133111ebULL

#define MIN_SEQUENCE 25

__device__ inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

__device__ inline uint64_t mix_stafford13(uint64_t x) {
    x = (x ^ (x >> 30)) * STAFFORD_MIX_1;
    x = (x ^ (x >> 27)) * STAFFORD_MIX_2;
    return x ^ (x >> 31);
}

struct GlowstoneRNG {
    uint64_t s0;
    uint64_t s1;

    __device__ void set_seed(uint64_t seed) {
        uint64_t lo = seed ^ SILVER_RATIO_64;
        uint64_t hi = lo - SUBTRACT_CONST;

        s0 = mix_stafford13(lo ^ GLOWSTONE_MD5_0);
        s1 = mix_stafford13(hi ^ GLOWSTONE_MD5_1);

        if ((s0 | s1) == 0) {
            s0 = GOLDEN_RATIO_64;
            s1 = SILVER_RATIO_64;
        }
    }

    __device__ uint64_t next_long() {
    uint64_t l = s0;
    uint64_t m = s1;

    uint64_t result = (rotl64(l + m, 17) + l) & MASK_64;

    m ^= l;
    s0 = (rotl64(l, 49) ^ m ^ (m << 21)) & MASK_64;
    s1 = rotl64(m, 28) & MASK_64;

    return result;
    }  

    __device__ int next_int(int bound) {
        while (true) {
            uint32_t bits = (uint32_t)(next_long() & 0xFFFFFFFF);
            uint64_t m = (uint64_t)bits * bound;
            uint32_t low = (uint32_t)m;

            if (low >= (uint32_t)(-bound % bound))
                return (int)(m >> 32);
        }
    }

    __device__ int glowstone_drop() {
        int drop = next_int(3) + 2;
        next_long(); // (burn when dropped)
        return drop;
    }
};

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        GlowstoneRNG rng;
        rng.set_seed(seed);

        int streak = 0;
        while (rng.glowstone_drop() == 4) {
            streak++;
        }

        if (streak >= MIN_SEQUENCE) {
            printf("Seed %llu with %d\n", seed, streak);
        }
    }
}

int main() {
    uint64_t start = 5000000000000ULL;
    uint64_t end   = 5000000000001ULL;

    int threads = 256;
    int blocks  = 256;

    search<<<blocks, threads>>>(start, end);
    cudaDeviceSynchronize();
    return 0;
}
