// !nvcc -O3 -arch=sm_75 backslime.cu -o backslime
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define MASK_64 0xFFFFFFFFFFFFFFFFULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define SUBTRACT_CONST 0x61C8864680B583EBULL
#define PANDA_SNEEZE_MD5_0 0x72a8f9a37353b657ULL
#define PANDA_SNEEZE_MD5_1 0xbc01ceb398ec0fe6ULL
#define STAFFORD_MIX_1 0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2 0x94d049bb133111ebULL
#define MIN_SEQUENCE 4 // min it outputs

struct Xoroshiro128Plus {
    uint64_t l, h;

    __device__ __forceinline__ Xoroshiro128Plus(uint64_t seed_lo, uint64_t seed_hi) {
        l = seed_lo & MASK_64;
        h = seed_hi & MASK_64;
    }

    __device__ __forceinline__ uint64_t rotl(uint64_t x, int k) {
        return ((x << k) | (x >> (64 - k))) & MASK_64;
    }

    __device__ __forceinline__ uint64_t next_long() {
        uint64_t s0 = l;
        uint64_t s1 = h;
        uint64_t result = (s0 + s1) & MASK_64;
        result = (rotl(result, 17) + s0) & MASK_64;

        s1 ^= s0;
        l = rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK_64);
        l &= MASK_64;
        h = rotl(s1, 28) & MASK_64;
        return result;
    }

    __device__ __forceinline__ int next_int(int bound) {
        uint64_t bits, val, product;
        do {
            bits = next_long() & 0xFFFFFFFFULL;
            product = bits * bound;
            val = (product >> 32) & 0xFFFFFFFFULL;
        } while ((product & 0xFFFFFFFFULL) < bound && (product & 0xFFFFFFFFULL) < ((-bound) % bound));
        return (int)val;
    }
};

__device__ __forceinline__ uint64_t mix_stafford(uint64_t x) {
    x ^= (x >> 30) & MASK_64;
    x = (x * STAFFORD_MIX_1) & MASK_64;
    x ^= (x >> 27) & MASK_64;
    x = (x * STAFFORD_MIX_2) & MASK_64;
    x ^= (x >> 31) & MASK_64;
    return x & MASK_64;
}

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        uint64_t lo = (seed ^ SILVER_RATIO_64) & MASK_64;
        uint64_t hi = (lo - SUBTRACT_CONST) & MASK_64;
        uint64_t mixed_lo = mix_stafford(lo ^ PANDA_SNEEZE_MD5_0);
        uint64_t mixed_hi = mix_stafford(hi ^ PANDA_SNEEZE_MD5_1);

        Xoroshiro128Plus rng(mixed_lo, mixed_hi);

        int streak = 0;
        while (true) {
            int drop = rng.next_int(700); // 0 to 699
            if (drop == 0) { // 0 drops slime
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
    uint64_t start = 1000000000000ULL;
    uint64_t end   = 1000000000001ULL; // example range

    int threads = 256;
    int blocks  = 256;

    search<<<blocks, threads>>>(start, end);
    cudaDeviceSynchronize();

    return 0;
}
