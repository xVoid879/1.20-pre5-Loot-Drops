// !nvcc -O3 -arch=sm_75 etrim.cu -o etrim
// Look at line 99 to change to longest streak without trim
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define MASK_64 0xFFFFFFFFFFFFFFFFULL

#define GOLDEN_RATIO_64  0x9e3779b97f4a7c15ULL
#define SILVER_RATIO_64  0x6a09e667f3bcc909ULL
#define ELDER_MD5_0      0xb29ffa7d6b2d2181ULL
#define ELDER_MD5_1      0x23990926ca2a6adfULL
#define STAFFORD_MIX_1   0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2   0x94d049bb133111ebULL

#define MIN_STREAK 15 // change if needed (most likely needed to change if longest streak)

struct Xoroshiro128Plus {
    uint64_t lo, hi;

    __device__ __forceinline__
    static uint64_t rotl(uint64_t x, int r) {
        return ((x << r) | (x >> (64 - r))) & MASK_64;
    }

    __device__ __forceinline__
    static uint64_t mix_stafford(uint64_t x) {
        x = (x ^ (x >> 30)) * STAFFORD_MIX_1 & MASK_64;
        x = (x ^ (x >> 27)) * STAFFORD_MIX_2 & MASK_64;
        return (x ^ (x >> 31)) & MASK_64;
    }

    __device__ __forceinline__
    Xoroshiro128Plus(uint64_t world_seed) {
        uint64_t l2 = (world_seed ^ SILVER_RATIO_64) & MASK_64;
        uint64_t l3 = (l2 + GOLDEN_RATIO_64) & MASK_64;

        lo = mix_stafford(l2 ^ ELDER_MD5_0);
        hi = mix_stafford(l3 ^ ELDER_MD5_1);

        if ((lo | hi) == 0) {
            lo = GOLDEN_RATIO_64;
            hi = SILVER_RATIO_64;
        }
    }

    __device__ __forceinline__
    uint64_t next_long() {
        uint64_t s0 = lo;
        uint64_t s1 = hi;
        uint64_t result = (rotl((s0 + s1) & MASK_64, 17) + s0) & MASK_64;
        s1 ^= s0;
        lo = (rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK_64)) & MASK_64;
        hi = rotl(s1, 28) & MASK_64;
        return result;
    }

    __device__ __forceinline__
    uint32_t next_int(uint32_t bound) {
        while (true) {
            uint32_t bits = next_long() & 0xFFFFFFFFULL;
            uint32_t val = (uint64_t(bits) * bound) >> 32;
            if (val < bound) return val;
        }
    }

    __device__ __forceinline__
    float next_float() {
        return ((next_long() >> 11) & ((1ULL << 53) - 1)) * (1.0 / (1ULL << 53));
    }
};

__device__ __forceinline__
bool next_is_trim(Xoroshiro128Plus &rng)
{   // burn
    int shards = rng.next_int(3); // shard
    int roll2 = rng.next_int(6); // cod crystal
    if (rng.next_float() < 0.025) { // fish
        int f = rng.next_int(100);
    }

    // 1 in 5 for armor
    int trim = rng.next_int(5);
    return trim == 4;
}

__global__ void search(uint64_t start, uint64_t end)
{
    uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + id; seed < end; seed += stride) {

        Xoroshiro128Plus rng(seed);

        int streak = 0;
        bool report = false;

        while (next_is_trim(rng)) { // if you want to search for longest streak without, put a ! before next_is_trim
            streak++;
            if (streak >= MIN_STREAK)
                report = true;
        }

        if (report) {
            printf("Seed %llu with %d\n", (unsigned long long)seed, streak);
        }
    }
}

int main()
{
    uint64_t start = 0;
    uint64_t end   = 1000000000000ULL;

    int threads = 256;
    int blocks = 256;

    search<<<blocks, threads>>>(start, end);
    cudaDeviceSynchronize();
    return 0;
}
