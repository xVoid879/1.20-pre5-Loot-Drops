# Inital Created by xpple

import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF
MAX_SEQUENCE = 20
GRAVEL_CONST = 1677721 # Fortune 1: 2396745 | Fortune 2: 4194304

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
GRAVEL_MD5_0 = 0x2fedfb509401412f
GRAVEL_MD5_1 = 0x6b4882392a3638a0
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb

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

def gravel_sequence(seed, max_seq=MAX_SEQUENCE):
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

    state = [l, h]
    sequence = []

    for _ in range(max_seq):
        count = 0
        while True:
            if (xNext64(state) >> 40) < GRAVEL_CONST:
                break
            count += 1
        sequence.append(count)
    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gravel sequences from a seed.")
    parser.add_argument("seed", type=int, help="World Seed")
    parser.add_argument("-d", "--depth", type=int, default=MAX_SEQUENCE,
                        help=f"Number of sequences to generate (default {MAX_SEQUENCE})")
    
    args = parser.parse_args()

    seq = gravel_sequence(args.seed, args.depth)
    print(f"Seed {args.seed} gravel sequence: {seq}")