import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF


SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
NAUTILUS_MD5_0 = 0xe479df711dbbedcc
NAUTILUS_MD5_1 = 0x218bc079ba6ac103
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20


def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))


def xNext64(state):
    l, h = state
    n = (rotl64((l + h) & MASK_64, 17) + l) & MASK_64
    h ^= l
    state[0] = (rotl64(l, 49) ^ h ^ ((h << 21) & MASK_64)) & MASK_64
    state[1] = rotl64(h, 28) & MASK_64
    return n


def next_float(state):
    return (xNext64(state) >> 11) * (1.0 / (1 << 53))


def nautilus_kills(seed, num_drops):
    l = (seed ^ SILVER_RATIO_64) & MASK_64
    h = (l - SUBTRACT_CONSTANT) & MASK_64

    l ^= NAUTILUS_MD5_0
    h ^= NAUTILUS_MD5_1

    l = ((l ^ (l >> 30)) * STAFFORD_MIX_1) & MASK_64
    h = ((h ^ (h >> 30)) * STAFFORD_MIX_1) & MASK_64
    l = ((l ^ (l >> 27)) * STAFFORD_MIX_2) & MASK_64
    h = ((h ^ (h >> 27)) * STAFFORD_MIX_2) & MASK_64
    l ^= (l >> 31)
    h ^= (h >> 31)

    state = [l, h]

    kills_between = []
    count = 0
    drops = 0

    while drops < num_drops:
        count += 1
        if next_float(state) < 0.05:
            kills_between.append(count)
            count = 0
            drops += 1

    return kills_between


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nautlius Mob Drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)

    args = parser.parse_args()

    result = nautilus_kills(args.seed, args.d)
    print(f"Seed {args.seed} nautilus's nautilus shell sequence: {result}")
