import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF
MAX_SEQUENCE = 20
SHORT_GRASS_CONST = 2097152

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SHORT_GRASS_MD5_0 = 0x8bbf3c7183e5ac1e
SHORT_GRASS_MD5_1 = 0x0e588cdc9bfba209
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

def short_grass_sequence(seed, max_seq=MAX_SEQUENCE):
    l = (seed ^ SILVER_RATIO_64) & MASK_64
    h = (l + GOLDEN_RATIO_64) & MASK_64
    l ^= SHORT_GRASS_MD5_0
    h ^= SHORT_GRASS_MD5_1
    l = ((l ^ (l >> 30)) * STAFFORD_MIX_1) & MASK_64
    h = ((h ^ (h >> 30)) * STAFFORD_MIX_1) & MASK_64
    l = ((l ^ (l >> 27)) * STAFFORD_MIX_2) & MASK_64
    h = ((h ^ (h >> 27)) * STAFFORD_MIX_2) & MASK_64
    l ^= (l >> 31)
    h ^= (h >> 31)

    state = [l, h]
    sequence = []

    for i in range(max_seq):
        count = 0
        while True:
            if (xNext64(state) >> 40) < SHORT_GRASS_CONST:
                break
            count += 1
        sequence.append(count)

        xNext64(state) # burn a call because they do that for some reason (same reason as 1.20-pre3 ig)

    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Short Grass")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    
    args = parser.parse_args()

    seq = short_grass_sequence(args.seed, args.d)
    print(f"Seed {args.seed} short grass wheat seed sequence: {seq}")
