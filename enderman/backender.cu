// !nvcc -O3 -arch=sm_75 backender.cu -o backender
// Finding back-to-back pearls
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define MASK_64         0xFFFFFFFFFFFFFFFFULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define SUBTRACT_CONST  0x61C8864680B583EBULL
#define ENDERMAN_MD5_0  0xa6dd84b6981b3414ULL
#define ENDERMAN_MD5_1  0x4bec48c8de2830b3ULL
#define STAFFORD_MIX_1  0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2  0x94d049bb133111ebULL
#define MIN_SEQUENCE    39

struct Xoroshiro128Plus {
    uint64_t l, h;

    __device__ __forceinline__ Xoroshiro128Plus(uint64_t seed) {
        uint64_t unmixed_lo = seed ^ SILVER_RATIO_64;
        uint64_t unmixed_hi = (unmixed_lo - SUBTRACT_CONST) & MASK_64;

        l = mix_stafford(unmixed_lo ^ ENDERMAN_MD5_0);
        h = mix_stafford(unmixed_hi ^ ENDERMAN_MD5_1);
    }

    __device__ __forceinline__ uint64_t mix_stafford(uint64_t x) {
        x = (x ^ (x >> 30)) * STAFFORD_MIX_1;
        x = (x ^ (x >> 27)) * STAFFORD_MIX_2;
        x ^= x >> 31;
        return x & MASK_64;
    }

    __device__ __forceinline__ uint64_t rotl(uint64_t x, int k) {
        return ((x << k) | (x >> (64 - k))) & MASK_64;
    }

    __device__ __forceinline__ uint64_t next_long() {
        uint64_t s0 = l;
        uint64_t s1 = h;
        uint64_t result = rotl((s0 + s1) & MASK_64, 17) + s0;
        s1 ^= s0;
        l = rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK_64);
        h = rotl(s1, 28);
        return result & MASK_64;
    }

    __device__ __forceinline__ uint32_t next_int32() {
        return next_long() & 0xFFFFFFFFULL;
    }

    __device__ __forceinline__ bool next_ender_pearl() {
        return ((next_int32() >> 31) % 2) == 1;
    }
};

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        Xoroshiro128Plus rng(seed);

        int streak = 0;
        while (rng.next_ender_pearl()) {
            streak++;
        }

        if (streak >= MIN_SEQUENCE) {
            printf("Seed %" PRIu64 " with %d\n", seed, streak);
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