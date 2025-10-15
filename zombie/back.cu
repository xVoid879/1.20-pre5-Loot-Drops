// !nvcc -O3 -arch=sm_75 back.cu -o back
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <inttypes.h>

#define MASK_64         0xFFFFFFFFFFFFFFFFULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define SUBTRACT_CONST  0x61C8864680B583EBULL
#define ZOMBIE_MD5_0    0xf5549fb67eceeb03ULL
#define ZOMBIE_MD5_1    0x4c45b69e40dba4ceULL
#define STAFFORD_MIX_1  0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2  0x94d049bb133111ebULL
#define MIN_SEQUENCE    5       // min+ backtoback to print

struct Xoroshiro128Plus {
    uint64_t l, h;

    __device__ __forceinline__ Xoroshiro128Plus(uint64_t seed) {
        uint64_t unmixed_lo = seed ^ SILVER_RATIO_64;
        uint64_t unmixed_hi = (unmixed_lo - SUBTRACT_CONST) & MASK_64;

        l = mix_stafford(unmixed_lo ^ ZOMBIE_MD5_0);
        h = mix_stafford(unmixed_hi ^ ZOMBIE_MD5_1);
    }

    __device__ __forceinline__ uint64_t mix_stafford(uint64_t x) {
        x = (x ^ (x >> 30)) * STAFFORD_MIX_1;
        x = (x ^ (x >> 27)) * STAFFORD_MIX_2;
        x ^= x >> 31;
        return x & MASK_64;
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

    __device__ __forceinline__ double next_float53() {
        return (next_long() >> 11) * (1.0 / (1ULL << 53));
    }

    __device__ __forceinline__ uint32_t next_int_bound(uint32_t bound) {
        uint64_t l, m, low;
        do {
            l = next_long() & 0xFFFFFFFFULL;
            m = l * bound;
            low = m & 0xFFFFFFFFULL;
        } while (low < bound && low < ((-bound) % bound));
        return (uint32_t)((m >> 32) & 0xFFFFFFFFULL);
    }

    __device__ __forceinline__ void roll_rotten_flesh() {
        double c = next_float53() * 2.0;
        (void)((int)(c + 0.5));
    }

    __device__ __forceinline__ int drop_rare() {
        if (next_float53() < 0.025)
            return next_int_bound(3);
        return -1;
    }

    __device__ __forceinline__ uint64_t rotl(uint64_t x, int k) {
        return ((x << k) | (x >> (64 - k))) & MASK_64;
    }
};

__global__ void search(uint64_t start, uint64_t end, int target_item) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        Xoroshiro128Plus rng(seed);

        int streak = 0;
        while (true) {
            rng.roll_rotten_flesh();
            int drop = rng.drop_rare();
            if (drop == target_item) {
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

int main(int argc, char** argv) {
    if (argc < 3 || strcmp(argv[1], "-n") != 0) {
        printf("Use: ./back -n iron, carrot, or potato\n");
        return 0;
    }

    int target_item = 2;
    if (strcmp(argv[2], "iron") == 0) target_item = 0;
    else if (strcmp(argv[2], "carrot") == 0) target_item = 1;
    else if (strcmp(argv[2], "potato") == 0) target_item = 2;
    else {
        printf("Wrong Name!\n");
        return 0;
    }

    uint64_t start = 1000000000000ULL;
    uint64_t end   = 1000000000001ULL;

    int threads = 256;
    int blocks  = 256;

    search<<<blocks, threads>>>(start, end, target_item);
    cudaDeviceSynchronize();

    return 0;
}