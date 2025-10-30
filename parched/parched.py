import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
PARCHED_MD5_0 = 0xf4160c92ec413859
PARCHED_MD5_1 = 0x882ea8e77272a5f5
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
    def __init__(self):
        self.seedLo = 0
        self.seedHi = 0
        self.seedLoHash = PARCHED_MD5_0
        self.seedHiHash = PARCHED_MD5_1

    def set_seed(self, seed):
        l2 = seed ^ SILVER_RATIO_64
        l3 = l2 - SUBTRACT_CONSTANT
        self.seedLo = mix_stafford13(l2 ^ self.seedLoHash)
        self.seedHi = mix_stafford13(l3 ^ self.seedHiHash)
        if (self.seedLo | self.seedHi) == 0:
            self.seedLo = GOLDEN_RATIO_64 & MASK_64
            self.seedHi = SILVER_RATIO_64 & MASK_64

    def next_long(self):
        l = self.seedLo
        m = self.seedHi
        n = (rotl64(l + m, 17) + l) & MASK_64
        m ^= l
        self.seedLo = (rotl64(l, 49) ^ m ^ (m << 21)) & MASK_64
        self.seedHi = rotl64(m, 28) & MASK_64
        return n

    def next_int(self, bound):
        if bound <= 0:
            raise ValueError("bound must be positive")
        while True:
            bits = self.next_long() & 0xFFFFFFFF
            val = (bits * bound) >> 32
            if val < bound:
                return val

def tipped_count(rng):
    base = rng.next_int(2)
    return base

def parched_drop(rng):
    arrows = rng.next_int(3)
    bones = rng.next_int(3)
    tipped_arrow = tipped_count(rng)
    return arrows, bones, tipped_arrow

def parched_sequence(seed, max_seq=MAX_SEQUENCE):
    rng = LootTableRNG()
    rng.set_seed(seed)
    arrows_seq = []
    bones_seq = []
    tipped_seq = []
    for _ in range(max_seq):
        a, b, t = parched_drop(rng)
        arrows_seq.append(a)
        bones_seq.append(b)
        tipped_seq.append(t)
    return arrows_seq, bones_seq, tipped_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parched Drop Simulation")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-n", choices=["arrows", "bones", "tipped"], default="arrows")

    args = parser.parse_args()
    arrows_seq, bones_seq, tipped_seq = parched_sequence(args.seed, args.d)

    if args.n == "arrows":
        print(f"Seed {args.seed} parched arrow sequence: {arrows_seq}")
    elif args.n == "bones":
        print(f"Seed {args.seed} parched bone sequence: {bones_seq}")
    elif args.n == "tipped":
        print(f"Seed {args.seed} parched tipped arrow sequence: {tipped_seq}")
