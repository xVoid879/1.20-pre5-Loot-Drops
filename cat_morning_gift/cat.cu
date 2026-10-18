// !nvcc -O3 -arch=sm_75 cat.cu -o catgift
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define MASK_64 0xFFFFFFFFFFFFFFFFULL
#define GOLDEN_RATIO_64 0x9e3779b97f4a7c15ULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define CAT_GIFT_MD5_0 0xdd853a8166273d58ULL
#define CAT_GIFT_MD5_1 0xd544fa93206d0bb3ULL
#define STAFFORD_MIXING_1 0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIXING_2 0x94d049bb133111ebULL
#define MIN_STREAK 6 // min back-to-back phantom membranes to print

struct Item {
    const char* name;
    int weight;
};

__constant__ Item Items[7] = {
    {"rabbit_hide",10},
    {"rabbit_foot",10},
    {"chicken",10},
    {"feather",10},
    {"rotten_flesh",10},
    {"string",10},
    {"phantom_membrane",2} // we care about this
};

__constant__ int Total_Weight = 62;

struct Xoroshiro128Plus {
    uint64_t l, h; // lo hi

    __device__ __forceinline__ Xoroshiro128Plus(uint64_t seed) {
        uint64_t unmixed_lo = (seed ^ SILVER_RATIO_64) & MASK_64;
        uint64_t unmixed_hi = (unmixed_lo + GOLDEN_RATIO_64) & MASK_64;
        l = mix_stafford(unmixed_lo ^ CAT_GIFT_MD5_0);
        h = mix_stafford(unmixed_hi ^ CAT_GIFT_MD5_1);
    }

    __device__ __forceinline__ uint64_t mix_stafford(uint64_t x) {
        x = (x ^ (x >> 30)) * STAFFORD_MIXING_1;
        x = (x ^ (x >> 27)) * STAFFORD_MIXING_2;
        x ^= x >> 31;
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

    __device__ __forceinline__ uint32_t next_int32() {
        return next_long() & 0xFFFFFFFFULL;
    }

    __device__ __forceinline__ uint32_t next_bound(uint32_t bound) {
        while (true) {
            uint32_t bits = next_int32();
            uint32_t val = (uint64_t(bits) * bound) >> 32;
            if (val < bound) return val;
        }
    }

    __device__ __forceinline__ int roll_item() {
        int j = next_bound(Total_Weight);
        for (int i = 0; i < 7; i++) {
            j -= Items[i].weight;
            if (j < 0) return i;
        }
        return -1;
    }

    __device__ __forceinline__ bool next_is_phantom_membrane() {
        return roll_item() == 6; // index 6 is phantom membrane
    }
};

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        Xoroshiro128Plus rng(seed);

        int streak = 0;
        bool report = false;

        while (rng.next_is_phantom_membrane()) {
            streak++;
            if (streak >= MIN_STREAK) report = true;
        }

        if (report) {
            printf("Seed %" PRIu64 " with %d\n", seed, streak);
        }
    }
}

int main() {
    uint64_t start = 1000000000000ULL;
    uint64_t end   = 1000000000001ULL;

    int threads = 256;
    int blocks  = 256;

    search<<<blocks, threads>>>(start, end);
    cudaDeviceSynchronize();
    return 0;
}
