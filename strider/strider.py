import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
STRIDER_MD5_0 = 0xe7e763a8d3ee683c
STRIDER_MD5_1 = 0xe6ea2c524c7e0b9f
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def mix_stafford13(seed: int) -> int:
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
    return (seed ^ (seed >> 31)) & MASK_64

class LootTableRNG:
    def __init__(self, md5_lo: int, md5_hi: int):
        self.seedLo = 0
        self.seedHi = 0
        self.md5_lo = md5_lo
        self.md5_hi = md5_hi

    def set_seed(self, world_seed: int):
        l2 = (world_seed ^ SILVER_RATIO_64) & MASK_64
        l3 = (l2 - SUBTRACT_CONSTANT) & MASK_64
        self.seedLo = mix_stafford13(l2 ^ self.md5_lo)
        self.seedHi = mix_stafford13(l3 ^ self.md5_hi)
        if (self.seedLo | self.seedHi) == 0:
            self.seedLo = GOLDEN_RATIO_64 & MASK_64
            self.seedHi = SILVER_RATIO_64 & MASK_64

    def next_long(self) -> int:
        l, m = self.seedLo, self.seedHi
        n = (rotl64((l + m) & MASK_64, 17) + l) & MASK_64
        m ^= l
        self.seedLo = (rotl64(l, 49) ^ m ^ ((m << 21) & MASK_64)) & MASK_64
        self.seedHi = rotl64(m, 28) & MASK_64
        return n

    def next_int(self, bound: int) -> int:
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

def simulate_strider(world_seed: int, depth: int = MAX_SEQUENCE):
    rng = LootTableRNG(STRIDER_MD5_0, STRIDER_MD5_1)
    rng.set_seed(world_seed)

    strings = []
    for _ in range(depth):
        strings.append(rng.next_int(4) + 2) # 2 to 5
    return strings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strider drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    seq = simulate_strider(args.seed, args.d)
    print(f"Seed {args.seed} strider string drops: {seq}")
