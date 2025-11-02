# Print the top 24 bits from the xoroshiro output that determines wheat seeds from short grass
import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF
GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
GRAVEL_MD5_0 = 0x8bbf3c7183e5ac1e
GRAVEL_MD5_1 = 0x0e588cdc9bfba209
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb

SHORT_GRASS_CONST = 2097152


def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))


def xNext64(state):
    l, h = state
    n = (rotl64((l + h) & MASK_64, 17) + l) & MASK_64
    h ^= l
    l_new = (rotl64(l, 49) ^ h ^ ((h << 21) & MASK_64)) & MASK_64
    h_new = rotl64(h, 28) & MASK_64
    state[0], state[1] = l_new, h_new
    return n


def initialize_state(seed):
    l = (seed ^ SILVER_RATIO_64) & MASK_64
    h = (l + GOLDEN_RATIO_64) & MASK_64
    l ^= GRAVEL_MD5_0
    h ^= GRAVEL_MD5_1
    l = ((l ^ (l >> 30)) * STAFFORD_MIX_1) & MASK_64
    h = ((h ^ (h >> 30)) * STAFFORD_MIX_1) & MASK_64
    l = ((l ^ (l >> 27)) * STAFFORD_MIX_2) & MASK_64
    h = ((h ^ (h >> 27)) * STAFFORD_MIX_2) & MASK_64
    l ^= (l >> 31)
    h ^= (h >> 31)
    return [l, h]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print top 24 bits from short grass's xoroshiro output of a seed.")
    parser.add_argument("seed", type=int)
    parser.add_argument("-n", type=int, default=20)
    args = parser.parse_args()

    state = initialize_state(args.seed)

    i = 0
    while i < args.n:
        rnd64 = xNext64(state)
        top24 = rnd64 >> 40
        if top24 < SHORT_GRASS_CONST:
            print(f"{i + 1}: {top24} (burn next)")
            xNext64(state)  # burn
        else:
            print(f"{i + 1}: {top24}")
        i += 1
