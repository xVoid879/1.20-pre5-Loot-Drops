import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
TWISTING_VINES_MD5_0 = 0xf60dab45daa5ab18
TWISTING_VINES_MD5_1 = 0xdc85972562e778ce
WEEPING_VINES_MD5_0 = 0xc999e4df84992279
WEEPING_VINES_MD5_1 = 0x53efb26f53ee37df
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

Blocks = { # less lines instead of defining in simulate
    "twisting": (TWISTING_VINES_MD5_0, TWISTING_VINES_MD5_1, 0.33), # 33%
    "weeping": (WEEPING_VINES_MD5_0, WEEPING_VINES_MD5_1, 0.33) # 33%
}

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def mix_stafford(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
    seed ^= (seed >> 31)
    return seed & MASK_64

def xNext64(state):
    l, h = state
    n = (rotl64((l + h) & MASK_64, 17) + l) & MASK_64
    h ^= l
    l_new = (rotl64(l, 49) ^ h ^ ((h << 21) & MASK_64)) & MASK_64
    h_new = rotl64(h, 28) & MASK_64
    state[0], state[1] = l_new, h_new
    return n

def next_float(state):
    return (xNext64(state) >> 11) * (1.0 / (1 << 53))

def simulate(seed, block_type, max_seq=MAX_SEQUENCE):
    if block_type not in Blocks:
        raise ValueError(f"? {block_type}")
    md5_lo, md5_hi, drop_chance = Blocks[block_type]

    lo = (seed ^ SILVER_RATIO_64) & MASK_64
    hi = (lo - SUBTRACT_CONSTANT) & MASK_64
    lo = mix_stafford(lo ^ md5_lo)
    hi = mix_stafford(hi ^ md5_hi)
    state = [lo, hi]

    drops = []
    for _ in range(max_seq):
        drops.append(1 if next_float(state) < drop_chance else 0)
    return drops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twisting/Weeping Vine Drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-i", type=str, required=True, choices=Blocks.keys())
    parser.add_argument("-n", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    drops = simulate(args.seed, args.i, args.n)
    print(f"Seed {args.seed} {args.item} vines sequence: {drops}")
