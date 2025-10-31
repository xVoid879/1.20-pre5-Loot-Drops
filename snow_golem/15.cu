// !nvcc -O3 -arch=sm_75 15.cu -o 15
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define MASK_64 0xFFFFFFFFFFFFFFFFULL
#define GOLDEN_RATIO_64 0x9e3779b97f4a7c15ULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define SNOW_GOLEM_MD5_0 0x151b724962fd1257ULL
#define SNOW_GOLEM_MD5_1 0xc4e9b2afbaf5b9f6ULL
#define STAFFORD_MIX_1 0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2 0x94d049bb133111ebULL
#define MIN_SEQUENCE 10 // min output

struct Xoroshiro128Plus {
    uint64_t l, h;

    __device__ __forceinline__ Xoroshiro128Plus(uint64_t seed) {
        uint64_t lo = (seed ^ SILVER_RATIO_64) & MASK_64;
        uint64_t hi = (lo + GOLDEN_RATIO_64) & MASK_64;

        l = mix_stafford(lo ^ SNOW_GOLEM_MD5_0);
        h = mix_stafford(hi ^ SNOW_GOLEM_MD5_1);
    }

    __device__ __forceinline__ uint64_t mix_stafford(uint64_t x) {
        x = (x ^ (x >> 30)) * STAFFORD_MIX_1;
        x = (x ^ (x >> 27)) * STAFFORD_MIX_2;
        x ^= (x >> 31);
        return x & MASK_64;
    }

    __device__ __forceinline__ uint64_t rotl(uint64_t x, int k) {
        return ((x << k) | (x >> (64 - k))) & MASK_64;
    }

    __device__ __forceinline__ uint64_t next_long() {
        uint64_t s0 = l;
        uint64_t s1 = h;
        uint64_t result = ((rotl((s0 + s1), 17) + s0) & MASK_64);
        s1 ^= s0;
        l = (rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK_64)) & MASK_64;
        h = rotl(s1, 28) & MASK_64;
        return result;
    }

    __device__ __forceinline__ int java_next_int(int bound) {
        uint64_t i, j, k;
        int val;
        do {
            i = next_long() & 0xFFFFFFFFULL;
            j = i * bound;
            k = j & 0xFFFFFFFFULL;
            val = (int)((j >> 32) & 0xFFFFFFFFULL);
        } while (k < bound && k < ((-bound) % bound));
        return val;
    }

    __device__ __forceinline__ int snowball_drop() {
        return java_next_int(16);
    }
};

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        Xoroshiro128Plus rng(seed);

        int streak = 0;
        while (true) {
            int drop = rng.snowball_drop();
            if (drop == 15) { // this is 15 snowballs. change to 0-14 if u want.
                streak++;
            } else {
                break;
            }
        }

        if (streak >= MIN_SEQUENCE) {
            printf("Seed %" PRIu64 " with %d\n", seed, streak);
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
