import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
WITCH_MD5_0 = 0x4be9888321dc796f
WITCH_MD5_1 = 0xf221b22df3592ddc
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

    def set_seed(self, world_seed: int):
        l2 = world_seed ^ SILVER_RATIO_64
        l3 = (l2 - SUBTRACT_CONSTANT) & MASK_64
        self.seedLo = mix_stafford13(l2 ^ self.seedLoHash)
        self.seedHi = mix_stafford13(l3 ^ self.seedHiHash)
        if (self.seedLo | self.seedHi) == 0:
            self.seedLo = GOLDEN_RATIO_64 & MASK_64
            self.seedHi = SILVER_RATIO_64 & MASK_64

    def next_long(self):
        l = self.seedLo
        m = self.seedHi
        n = (rotl64((l + m) & MASK_64, 17) + l) & MASK_64
        m ^= l
        self.seedLo = (rotl64(l, 49) ^ m ^ ((m << 21) & MASK_64)) & MASK_64
        self.seedHi = rotl64(m, 28) & MASK_64
        return n

    def next_float(self):
        return (self.next_long() >> 11) * (1.0 / (1 << 53))

    def next_int(self, bound):
        if bound <= 0:
            raise ValueError("bound must be positive")
        l = self.next_long() & 0xFFFFFFFF
        m = (l * bound) & 0xFFFFFFFFFFFFFFFF
        low = m & 0xFFFFFFFF
        if low < bound:
            t = (-bound) % bound
            while low < t:
                l = self.next_long() & 0xFFFFFFFF
                m = (l * bound) & 0xFFFFFFFFFFFFFFFF
                low = m & 0xFFFFFFFF
        return (m >> 32) & 0xFFFFFFFF

MAIN_TOTAL_WEIGHT = 7

MAIN_ITEMS = ["glowstone_dust", "sugar", "spider_eye", "glass_bottle", "gunpowder", "stick"]

def simulate_witch_sequence(world_seed: int, item_name: str, max_seq: int = MAX_SEQUENCE):
    rng = LootTableRNG(WITCH_MD5_0, WITCH_MD5_1)
    rng.set_seed(world_seed)

    seq = []

    for _ in range(max_seq):
        rolls = rng.next_int(3) + 1
        counts = {
            "glowstone_dust": 0,
            "sugar": 0,
            "spider_eye": 0,
            "glass_bottle": 0,
            "gunpowder": 0,
            "stick": 0
        }
        for _r in range(rolls):
            sel = rng.next_int(MAIN_TOTAL_WEIGHT)
            if sel == 0:
                pick = "glowstone_dust"
            elif sel == 1:
                pick = "sugar"
            elif sel == 2:
                pick = "spider_eye"
            elif sel == 3:
                pick = "glass_bottle"
            elif sel == 4:
                pick = "gunpowder"
            else:
                pick = "stick"
            cnt = rng.next_int(3)
            counts[pick] += cnt

        redstone_count = rng.next_int(5) + 4

        if item_name == "redstone":
            seq.append(redstone_count)
        elif item_name in counts:
            seq.append(counts[item_name])
        else:
            raise ValueError(f"Unknown item '{item_name}'")

    return seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Witch drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-i", required=True, choices=["glowstone_dust", "sugar", "spider_eye", "glass_bottle", "gunpowder", "stick", "redstone"])
    args = parser.parse_args()

    seq = simulate_witch_sequence(args.seed, args.i, args.d)
    print(f"Seed {args.seed} witch {args.i} sequence: {seq}")
