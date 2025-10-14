// !nvcc -O3 -arch=sm_75 unluckydead.cu -o unluckydead
// GPU: Finding longest streak without stick
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define GOLDEN_RATIO_64 0x9e3779b97f4a7c15ULL // step 1
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL // step 1
#define DEAD_BUSH_MD5_0 0xa00e2b904f10208bULL // setp 2
#define DEAD_BUSH_MD5_1 0x61ced0c97606ee23ULL // step 2
#define STAFFORD_MIX_1  0xbf58476d1ce4e5b9ULL // step 3
#define STAFFORD_MIX_2  0x94d049bb133111ebULL // step 3
#define MASK_64         0xFFFFFFFFFFFFFFFFULL
#define MIN_ZEROS       25   

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int r) {
    return ((x << r) & MASK_64) | (x >> (64 - r));
}

struct Xoroshiro128Plus { // get 64 bit # from world seed
    uint64_t l, h;

    __device__ __forceinline__
    Xoroshiro128Plus(uint64_t seed) { // hash hash hash hash
        l = seed ^ SILVER_RATIO_64;
        h = l + GOLDEN_RATIO_64;
        l ^= DEAD_BUSH_MD5_0;
        h ^= DEAD_BUSH_MD5_1;
        l = (l ^ (l >> 30)) * STAFFORD_MIX_1;
        h = (h ^ (h >> 30)) * STAFFORD_MIX_1;
        l = (l ^ (l >> 27)) * STAFFORD_MIX_2;
        h = (h ^ (h >> 27)) * STAFFORD_MIX_2;
        l ^= l >> 31;
        h ^= h >> 31;
    }

    __device__ __forceinline__
    uint64_t nextLong() { // turn l and h into 64 bit #
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
            uint32_t l = (- (uint32_t)bound) % (uint32_t)bound;
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
        int minStreak = INT32_MAX; 
        bool found = false;

        while (true) {
            int sticks = java_next_int(rng, 3); 
            if (sticks == 0) {
                curStreak++;
            } else {
                if (curStreak > 0) {
                    if (curStreak < minStreak) minStreak = curStreak;
                    found = true;
                }
                curStreak = 0;
                break; 
            }
        }

        if (curStreak > 0) {
            if (curStreak < minStreak) minStreak = curStreak;
            found = true;
        }

        if (found && minStreak >= MIN_ZEROS) {
            printf("Seed %llu with %d\n", seed, minStreak);
        }
    }
}

int main() {
    uint64_t start = 0ULL;
    uint64_t end   = 1ULL; 

    int threads = 256;
    int blocks  = 256;

    search<<<blocks, threads>>>(start, end);
    cudaDeviceSynchronize();
    return 0;
}