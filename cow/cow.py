import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
COW_MD5_0 = 0xf18d816fdfeb04d1
COW_MD5_1 = 0xd0edefc7dbb80042
MOOSHROOM_MD5_0 = 0x4588e30d5e7d376a
MOOSHROOM_MD5_1 = 0x486b61c7427861fb
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def mix_stafford13(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
    return seed ^ (seed >> 31)

class LootTableRNG:
    def __init__(self, md5_lo, md5_hi):
        self.seedLo = 0
        self.seedHi = 0
        self.seedLoHash = md5_lo
        self.seedHiHash = md5_hi

    def set_seed(self, seed):
        l2 = seed ^ SILVER_RATIO_64
        l3 = l2 - SUBTRACT_CONSTANT
        self.seedLo = mix_stafford13(l2 ^ self.seedLoHash)
        self.seedHi = mix_stafford13(l3 ^ self.seedHiHash)
        if (self.seedLo | self.seedHi) == 0:
            self.seedLo = GOLDEN_RATIO_64
            self.seedHi = SILVER_RATIO_64

    def next_long(self):
        l = self.seedLo
        m = self.seedHi
        n = (rotl64(l + m, 17) + l) & MASK_64
        m ^= l
        self.seedLo = (rotl64(l, 49) ^ m ^ (m << 21)) & MASK_64
        self.seedHi = rotl64(m, 28) & MASK_64
        return n

    def next_int(self, bound):
        l = self.next_long() & 0xFFFFFFFF
        m = l * bound
        low = m & 0xFFFFFFFF
        if low < bound:
            t = (-bound) % bound
            while low < t:
                l = self.next_long() & 0xFFFFFFFF
                m = l * bound
                low = m & 0xFFFFFFFF
        return (m >> 32) & 0xFFFFFFFF

def cow_drop(rng):
    leather_count = rng.next_int(3)  # 0 to 2
    beef_count = rng.next_int(3) + 1  # 1 to 3
    return {"leather": leather_count, "beef": beef_count}

def cow_sequence(seed, cow_type="cow", max_seq=MAX_SEQUENCE):
    if cow_type == "cow":
        md5_lo, md5_hi = COW_MD5_0, COW_MD5_1
    elif cow_type == "mooshroom":
        md5_lo, md5_hi = MOOSHROOM_MD5_0, MOOSHROOM_MD5_1
    else:
        raise ValueError(f"Wrong type: {cow_type}")

    rng = LootTableRNG(md5_lo, md5_hi)
    rng.set_seed(seed)
    sequence = []

    for _ in range(max_seq):
        sequence.append(cow_drop(rng))
    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cow/Mooshroom Drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-t", choices=["cow", "mooshroom"], required=True)
    parser.add_argument("-i", choices=["leather", "beef"], required=True)

    args = parser.parse_args()

    seq = cow_sequence(args.seed, cow_type=args.t, max_seq=args.d)
    item_seq = [drops[args.i] for drops in seq]

    print(f"Seed {args.seed} {args.t} {args.i} sequence: {item_seq}")
