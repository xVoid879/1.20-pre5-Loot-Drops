// !nvcc -O3 -arch=sm_75 backwheat.cu -o backwheat
// Finding back-to-back wheat seeds from short grass
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define GOLDEN_RATIO_64 0x9e3779b97f4a7c15ULL // step 1
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL // step 1
#define SHORT_GRASS_MD5_0 0x8bbf3c7183e5ac1eULL // step 2
#define SHORT_GRASS_MD5_1 0x0e588cdc9bfba209ULL // step 2
#define STAFFORD_MIX_1  0xbf58476d1ce4e5b9ULL // step 3
#define STAFFORD_MIX_2  0x94d049bb133111ebULL // step 3
#define SHORT_GRASS_CONST 2097152ULL            // 10% of 2^24
#define MIN_ZEROS       5                    // min # it outputs

struct Xoroshiro128Plus { // generate random 64 bit # from doing some hashing steps on the world seed (every mine = new one generated)
    uint64_t l, h;

    __device__ __forceinline__
    Xoroshiro128Plus(uint64_t seed) {
        // hash hash hash hash
        l = seed ^ SILVER_RATIO_64;
        h = l + GOLDEN_RATIO_64;
        l ^= SHORT_GRASS_MD5_0;
        h ^= SHORT_GRASS_MD5_1;

        l = (l ^ (l >> 30)) * STAFFORD_MIX_1;
        h = (h ^ (h >> 30)) * STAFFORD_MIX_1;
        l = (l ^ (l >> 27)) * STAFFORD_MIX_2;
        h = (h ^ (h >> 27)) * STAFFORD_MIX_2;
        l ^= l >> 31;
        h ^= h >> 31;
    }

    __device__ __forceinline__
    uint64_t next_long() {
        // get the 64 bit random number with the state (l and h) using xNextLong()
        uint64_t n = ((l + h) << 17 | (l + h) >> (64-17)) + l;
        h ^= l;
        l = (l << 49 | l >> (64-49)) ^ h ^ (h << 21);
        h = (h << 28) | (h >> (64-28));
        return n;
    }
};

// search search search search
__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        Xoroshiro128Plus rng(seed); // get random 64 bit #
        int zeros = 0;

        while (SHORT_GRASS_CONST > (rng.next_long() >> 40)) { // shift right (64-40=24) to get upper 24 to compare against the inequality
            zeros++;
        }

        if (zeros >= MIN_ZEROS) {
            printf("Seed %llu with %d\n", seed, zeros);
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
