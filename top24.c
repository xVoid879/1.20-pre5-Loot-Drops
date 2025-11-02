// gcc -O3 -o top24 top24.c
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>

#define MASK_64 0xFFFFFFFFFFFFFFFFULL

#define GOLDEN_RATIO_64 0x9e3779b97f4a7c15ULL
#define SILVER_RATIO_64 0x6a09e667f3bcc909ULL
#define MD5_0 0x8bbf3c7183e5ac1eULL // This is Short Grass. Change to gravel if you want.
#define MD5_1 0x0e588cdc9bfba209ULL // This is Short Grass. Change to gravel if you want.
#define STAFFORD_MIX_1  0xbf58476d1ce4e5b9ULL
#define STAFFORD_MIX_2  0x94d049bb133111ebULL

uint64_t rotl64(uint64_t x, int r) {
    return ((x << r) & MASK_64) | (x >> (64 - r));
}

uint64_t xNext64(uint64_t state[2]) {
    uint64_t l = state[0];
    uint64_t h = state[1];
    uint64_t n = (rotl64((l + h) & MASK_64, 17) + l) & MASK_64;

    h ^= l;
    uint64_t l_new = (rotl64(l, 49) ^ h ^ ((h << 21) & MASK_64)) & MASK_64;
    uint64_t h_new = rotl64(h, 28) & MASK_64;

    state[0] = l_new;
    state[1] = h_new;

    return n;
}

void initialize_state(uint64_t seed, uint64_t state[2]) {
    uint64_t l = (seed ^ SILVER_RATIO_64) & MASK_64;
    uint64_t h = (l + GOLDEN_RATIO_64) & MASK_64;

    l ^= MD5_0;
    h ^= MD5_1;

    l = ((l ^ (l >> 30)) * STAFFORD_MIX_1) & MASK_64;
    h = ((h ^ (h >> 30)) * STAFFORD_MIX_1) & MASK_64;
    l = ((l ^ (l >> 27)) * STAFFORD_MIX_2) & MASK_64;
    h = ((h ^ (h >> 27)) * STAFFORD_MIX_2) & MASK_64;
    l ^= (l >> 31);
    h ^= (h >> 31);

    state[0] = l;
    state[1] = h;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("./top24 TOP24 START END\n", argv[0]);
        return 1;
    }

    uint64_t target = strtoull(argv[1], NULL, 0);
    uint64_t start_seed = (argc > 2) ? strtoull(argv[2], NULL, 0) : 0;
    uint64_t max_seeds = (argc > 3) ? strtoull(argv[3], NULL, 0) : 0xFFFFFFFFULL;
    uint64_t found_count = 0;

    for (uint64_t seed = start_seed; seed < start_seed + max_seeds; seed++) {
        uint64_t state[2];
        initialize_state(seed, state);
        uint64_t rnd = xNext64(state);
        uint64_t top24 = rnd >> 40;

        if (top24 == target) {
            printf("Found seed: %" PRIu64 "\n", seed);
            found_count++;
        }
    }

    printf("Search complete");
    return 0;
}
