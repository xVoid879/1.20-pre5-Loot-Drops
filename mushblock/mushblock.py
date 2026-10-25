import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
BROWN_MUSHROOM_MD5_0 = 0x91fa4b17458f0323
BROWN_MUSHROOM_MD5_1 = 0x580b19a848cd2c89
RED_MUSHROOM_MD5_0 = 0x245a17efb6417bec
RED_MUSHROOM_MD5_1 = 0x679b8d25d5c6022d
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

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

def next_int(state, bound):
    while True:
        bits = xNext64(state) & 0xFFFFFFFF
        val = (bits * bound) >> 32
        if val < bound:
            return val

def mushroom_sequence(seed, color="brown", max_seq=MAX_SEQUENCE):
    if color == "brown":
        lo, hi = BROWN_MUSHROOM_MD5_0, BROWN_MUSHROOM_MD5_1
    elif color == "red":
        lo, hi = RED_MUSHROOM_MD5_0, RED_MUSHROOM_MD5_1
    else:
        raise ValueError(f"Wrong color: {color}. Must be brown or red.")

    l = (seed ^ SILVER_RATIO_64) & MASK_64
    h = (l + GOLDEN_RATIO_64) & MASK_64
    l ^= lo
    h ^= hi
    l = ((l ^ (l >> 30)) * STAFFORD_MIX_1) & MASK_64
    h = ((h ^ (h >> 30)) * STAFFORD_MIX_1) & MASK_64
    l = ((l ^ (l >> 27)) * STAFFORD_MIX_2) & MASK_64
    h = ((h ^ (h >> 27)) * STAFFORD_MIX_2) & MASK_64
    l ^= (l >> 31)
    h ^= (h >> 31)

    state = [l, h]
    sequence = []

    for _ in range(max_seq):
        count = next_int(state, 9) - 6
        sequence.append(max(0, count))

    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Red and Brown Mushroom Block Drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-c", type=str, required=True, choices=["brown", "red"])
    args = parser.parse_args()

    seq = mushroom_sequence(args.seed, color=args.c, max_seq=args.d)
    print(f"Seed {args.seed} {args.c} mushroom block sequence: {seq}")
