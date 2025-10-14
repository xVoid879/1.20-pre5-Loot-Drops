# Print the top 24 bits from the xoroshiro output that determines flint or gravel
import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
GRAVEL_MD5_0 = 0xef6489bec2529e35
GRAVEL_MD5_1 = 0x1f1ab2c703aa2b5d
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def xNext64(state):
    # convert l and h (state) into 64 bit number
    l, h = state
    n = (rotl64((l + h) & MASK_64, 17) + l) & MASK_64
    h ^= l
    l_new = (rotl64(l, 49) ^ h ^ ((h << 21) & MASK_64)) & MASK_64
    h_new = rotl64(h, 28) & MASK_64
    state[0], state[1] = l_new, h_new
    return n

def initialize_state(seed):
    # hash hash hash hash (this should be beofre xNext64 but oh well)
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
    parser = argparse.ArgumentParser(description="Print value coming from nextFloat()")
    parser.add_argument("seed", type=int)
    parser.add_argument("-n", "--count", type=int, default=20)
    args = parser.parse_args()

    state = initialize_state(args.seed)

    for i in range(args.count):
        rnd64 = xNext64(state)
        top24 = rnd64 >> 40  # extract top 24 bits
        print(f"{i+1}: {top24}")