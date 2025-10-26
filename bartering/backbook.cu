// !nvcc -O3 -arch=sm_75 backbook.cu -o backbook
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

#define MASK_64 0xFFFFFFFFFFFFFFFFULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define SUBTRACT_CONST 0x61C8864680B583EBULL
#define BARTERING_MD5_0 0xf79b444cdb83b923ULL
#define BARTERING_MD5_1 0xe09fad0dcb68166aULL
#define STAFFORD_MIXING_1 0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIXING_2 0x94d049bb133111ebULL
#define MIN_STREAK 6 // min back to back books to print

struct Item {
    const char* name;
    int min;
    int max;
    int weight;
    bool roll_amount;
    bool roll_ss;
};

__constant__ Item Items[18] = {
    {"book",1,1,5,false,true}, // we care about this
    {"iron_boots",1,1,8,false,true},
    {"potion",1,1,8,false,false},
    {"splash_potion",1,1,8,false,false},
    {"potion",1,1,10,false,false},
    {"iron_nugget",10,36,10,true,false},
    {"ender_pearl",2,4,10,true,false},
    {"string",3,9,20,true,false},
    {"quartz",5,12,20,true,false},
    {"obsidian",1,1,40,false,false},
    {"crying_obsidian",1,3,40,true,false},
    {"fire_charge",1,1,40,false,false},
    {"leather",2,4,40,true,false},
    {"soul_sand",2,8,40,true,false},
    {"nether_brick",2,8,40,true,false},
    {"spectral_arrow",6,12,40,true,false},
    {"gravel",8,16,40,true,false},
    {"blackstone",8,16,40,true,false}
};

__constant__ int Total_Weight = 459; // 5 + 8 + 8 + 8 + 10 + 10 + 10 + 20 + 20 + 40 + 40 + 40 + 40 + 40 + 40 + 40 + 40 + 40

struct Xoroshiro128Plus {
    uint64_t l, h; // lo hi

    __device__ __forceinline__ Xoroshiro128Plus(uint64_t seed) {
        uint64_t unmixed_lo = seed ^ SILVER_RATIO_64;
        uint64_t unmixed_hi = (unmixed_lo - SUBTRACT_CONST) & MASK_64;
        l = mix_stafford(unmixed_lo ^ BARTERING_MD5_0);
        h = mix_stafford(unmixed_hi ^ BARTERING_MD5_1);
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
        for (int i = 0; i < 18; i++) {
            j -= Items[i].weight;
            if (j < 0) return i;
        }
        return -1;
    }

    __device__ __forceinline__ bool next_is_book() {
        int idx = roll_item();
        if (idx < 0) return false;
        if (Items[idx].roll_ss) {
            next_bound(1);
            next_bound(3);
        }
        if (Items[idx].roll_amount) {
            next_bound(Items[idx].max - Items[idx].min + 1); // 0-2 + 1 = 1-3 enchantment
        }
        return idx == 0; // index 0 is book. change it if u want to not do book
    }
};

__global__ void search(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t seed = start + idx; seed < end; seed += stride) {
        Xoroshiro128Plus rng(seed);

        int streak = 0;
        bool report = false;

        while (rng.next_is_book()) {
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
