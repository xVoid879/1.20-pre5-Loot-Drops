import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
GUARDIAN_MD5_0 = 0xe862d54db922510d
GUARDIAN_MD5_1 = 0x9219364a4a525ce6
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def rotl(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def mix_stafford13(seed: int) -> int:
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
    return (seed ^ (seed >> 31)) & MASK_64

class LootTableRNG:
    def __init__(self, md5_lo: int, md5_hi: int):
        self.seedLo = 0
        self.seedHi = 0
        self.seedLoHash = md5_lo
        self.seedHiHash = md5_hi

    def set_seed(self, world_seed: int):
        l2 = (world_seed ^ SILVER_RATIO_64) & MASK_64
        l3 = (l2 + GOLDEN_RATIO_64) & MASK_64
        self.seedLo = mix_stafford13(l2 ^ self.seedLoHash)
        self.seedHi = mix_stafford13(l3 ^ self.seedHiHash)
        if (self.seedLo | self.seedHi) == 0:
            self.seedLo = GOLDEN_RATIO_64
            self.seedHi = SILVER_RATIO_64

    def next_long(self) -> int:
        l = self.seedLo
        m = self.seedHi
        n = (rotl((l + m) & MASK_64, 17) + l) & MASK_64
        m ^= l
        self.seedLo = (rotl(l, 49) ^ m ^ ((m << 21) & MASK_64)) & MASK_64
        self.seedHi = rotl(m, 28) & MASK_64
        return n

    def next_float(self) -> float:
        return ((self.next_long() >> 11) & ((1 << 53) - 1)) * (1.0 / (1 << 53))

    def next_int(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        while True:
            bits = self.next_long() & 0xFFFFFFFF
            val = (bits * bound) >> 32
            if val < bound:
                return val

def roll(rng, min_val, max_val):
    return min_val + rng.next_int(max_val - min_val + 1)

def simulate_guardian(world_seed: int):
    rng = LootTableRNG(GUARDIAN_MD5_0, GUARDIAN_MD5_1)
    rng.set_seed(world_seed)

    results = []

    for i in range(MAX_SEQUENCE):
        drops = []

        prismarine_shard = roll(rng, 0, 2)
        if prismarine_shard > 0:
            drops.append(("prismarine_shard", prismarine_shard))

        roll2 = rng.next_int(5)
        if roll2 in (0, 1):
            drops.append(("cod", 1))
        elif roll2 in (2, 3):
            drops.append(("prismarine_crystals", 1))

        if rng.next_float() < 0.025:
            fish_roll = rng.next_int(100)
            if fish_roll < 60:
                drops.append(("cod", 1))
            elif fish_roll < 85:
                drops.append(("salmon", 1))
            elif fish_roll < 87:
                drops.append(("tropical_fish", 1))
            else:
                drops.append(("pufferfish", 1))

        results.append((i + 1, drops))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guardian Drops")
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    results = simulate_guardian(args.seed)

    for kill_num, drops in results:
        print(f"Kill {kill_num}")
        if drops:
            for item, count in drops:
                print(f"  {item} x{count}")
        else:
                 print("  (no drops)")
        print()
