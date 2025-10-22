// !nvcc -O3 -arch=sm_75 unluckyskyblock.cu -o unluckyskyblock
#include <cstdio>
#include <cstdint>
#include <inttypes.h>
#include <cuda_runtime.h>

#define MASK_64          0xFFFFFFFFFFFFFFFFULL
#define SILVER_RATIO_64  0x6A09E667F3BCC909ULL
#define SUBTRACT_CONST   0x61C8864680B583EBULL
#define OAK_LEAVES_MD5_0 0xEF6489BEC2529E35ULL
#define OAK_LEAVES_MD5_1 0x1F1AB2C703AA2B5DULL
#define STAFFORD_MIX_1   0xBF58476D1CE4E5B9ULL
#define STAFFORD_MIX_2   0x94D049BB133111EBULL
#define MIN_STREAK       1200  // min to print

struct Xoroshiro128Plus {
    uint64_t l, h; // state

    __device__ __forceinline__ Xoroshiro128Plus(uint64_t seed) { // xoroshiro and md5 hashing
        uint64_t unmixed_lo = seed ^ SILVER_RATIO_64;
        uint64_t unmixed_hi = (unmixed_lo - SUBTRACT_CONST) & MASK_64;
        l = mix_stafford(unmixed_lo ^ OAK_LEAVES_MD5_0);
        h = mix_stafford(unmixed_hi ^ OAK_LEAVES_MD5_1);
    }

    __device__ __forceinline__ uint64_t mix_stafford(uint64_t x) { // mix it up, mix it up
        x = (x ^ (x >> 30)) * STAFFORD_MIX_1;
        x = (x ^ (x >> 27)) * STAFFORD_MIX_2;
        x ^= x >> 31;
        return x & MASK_64;
    }

    __device__ __forceinline__ uint64_t next_long() { // turn state to 64 bit #
        uint64_t s0 = l, s1 = h;
        uint64_t result = ((s0 + s1) << 17 | (s0 + s1) >> (47)) + s0;
        s1 ^= s0;
        l = ((s0 << 49 | s0 >> (15)) ^ s1 ^ (s1 << 21)) & MASK_64;
        h = (s1 << 28 | s1 >> (36)) & MASK_64;
        return result & MASK_64;
    }

    __device__ __forceinline__ float next_float24() {
        return (next_long() >> 40) / float(1 << 24);
    }

    __device__ __forceinline__ bool next_drop(bool &got_stick) {
        next_float24();
        const float stick_chance[5] = {0.02f, 0.022222223f, 0.025f, 0.033333335f, 0.1f};
        int fortune = 0; // in case some one asks
        got_stick = next_float24() < stick_chance[fortune];
        next_float24();
        return false;
    }
};

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        Xoroshiro128Plus rng(seed);

        int streak = 0;
        while (true) {
            bool got_stick = false;
            rng.next_drop(got_stick);
            if (got_stick) break; 
            streak++;
        }

        if (streak >= MIN_STREAK) {
            printf("Seed %" PRIu64 " with %d\n", seed, streak+1); // cause it prints one less
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
