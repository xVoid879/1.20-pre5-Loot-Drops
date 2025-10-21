import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF
GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
ZOMBIFIED_PIGLIN_MD5_0 = 0x793941d11bfc05c2
ZOMBIFIED_PIGLIN_MD5_1 = 0x557e817ca51ae2d0
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20


def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))


def mix_stafford13(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
    return (seed ^ (seed >> 31)) & MASK_64


class LootTableRNG:
    def __init__(self, md5_lo, md5_hi):
        self.seedLo = 0
        self.seedHi = 0
        self.md5_lo = md5_lo
        self.md5_hi = md5_hi

    def set_seed(self, world_seed):
        l2 = (world_seed ^ SILVER_RATIO_64) & MASK_64
        l3 = (l2 - SUBTRACT_CONSTANT) & MASK_64
        self.seedLo = mix_stafford13(l2 ^ self.md5_lo)
        self.seedHi = mix_stafford13(l3 ^ self.md5_hi)
        if (self.seedLo | self.seedHi) == 0:
            self.seedLo = GOLDEN_RATIO_64 & MASK_64
            self.seedHi = SILVER_RATIO_64 & MASK_64

    def next_long(self):
        l, m = self.seedLo, self.seedHi
        n = (rotl64((l + m) & MASK_64, 17) + l) & MASK_64
        m ^= l
        self.seedLo = (rotl64(l, 49) ^ m ^ ((m << 21) & MASK_64)) & MASK_64
        self.seedHi = rotl64(m, 28) & MASK_64
        return n

    def next_int(self, bound):
        if bound <= 0:
            raise ValueError("bound must be positive")
        l = self.next_long() & 0xFFFFFFFF
        m = (l * bound) & MASK_64
        low = m & 0xFFFFFFFF
        if low < bound:
            t = (-bound) % bound
            while low < t:
                l = self.next_long() & 0xFFFFFFFF
                m = (l * bound) & MASK_64
                low = m & 0xFFFFFFFF
        return (m >> 32) & 0xFFFFFFFF


def simulate_zombified_piglin(world_seed, max_seq=MAX_SEQUENCE):
    rng = LootTableRNG(ZOMBIFIED_PIGLIN_MD5_0, ZOMBIFIED_PIGLIN_MD5_1)
    rng.set_seed(world_seed)

    fleshes = []
    nuggets = []
    ingots = []

    for _ in range(max_seq):
        fleshes.append(rng.next_int(2))
        nuggets.append(rng.next_int(2))
        ingots.append(1 if rng.next_int(1000) < 25 else 0)

    return fleshes, nuggets, ingots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zombified Piglin Drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-i", required=True, choices=["flesh", "nugget", "ingot"])
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    flesh_seq, nugget_seq, ingot_seq = simulate_zombified_piglin(args.seed, args.d)

    if args.i == "flesh":
        print(f"Seed {args.seed} zombified_piglin rotten_flesh drops: {flesh_seq}")
    elif args.i == "nugget":
        print(f"Seed {args.seed} zombified_piglin gold_nugget drops: {nugget_seq}")
    elif args.i == "ingot":
        print(f"Seed {args.seed} zombified_piglin gold_ingot drops: {ingot_seq}")
