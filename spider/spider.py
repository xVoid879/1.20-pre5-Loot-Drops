import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
SPIDER_MD5_0 = 0xa156e1cbe4b9ddcf
SPIDER_MD5_1 = 0x06b1d0daf66f4e24
CAVE_SPIDER_MD5_0 = 0x0b0d2c0f3c3d3d5b
CAVE_SPIDER_MD5_1 = 0xcbd4a636a1c9348d
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

    def next_int(self, min_val, max_val):
        bound = max_val - min_val + 1
        l = self.next_long() & 0xFFFFFFFF
        m = l * bound
        low = m & 0xFFFFFFFF
        if low < bound:
            t = (-bound) % bound
            while low < t:
                l = self.next_long() & 0xFFFFFFFF
                m = l * bound
                low = m & 0xFFFFFFFF
        return ((m >> 32) & 0xFFFFFFFF) + min_val

class SpiderLoot:
    def __init__(self, seed, md5_lo, md5_hi):
        self.rng = LootTableRNG(md5_lo, md5_hi)
        self.rng.set_seed(seed)

    def drop_string(self):
        return self.rng.next_int(0, 2)

    def drop_spider_eye(self):
        return max(0, self.rng.next_int(-1, 1))

def spider_sequence(seed, spider_type="spider", max_seq=MAX_SEQUENCE):
    if spider_type == "spider":
        md5_lo, md5_hi = SPIDER_MD5_0, SPIDER_MD5_1
    elif spider_type == "cave_spider":
        md5_lo, md5_hi = CAVE_SPIDER_MD5_0, CAVE_SPIDER_MD5_1
    else:
        raise ValueError(f"Unknown spider type '{spider_type}'")

    loot = SpiderLoot(seed, md5_lo, md5_hi)
    sequence = []
    for _ in range(max_seq):
        sequence.append({
            "string": loot.drop_string(),
            "spider_eye": loot.drop_spider_eye()
        })
    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spider/Cave Drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-t", type=str, choices=["spider", "cave_spider"], default="spider")
    parser.add_argument("-n", type=str, choices=["string", "spider_eye"], required=True)

    args = parser.parse_args()

    seq = spider_sequence(args.seed, spider_type=args.t, max_seq=args.d)
    sequence = [drops[args.n] for drops in seq]
    print(f"Seed {args.seed} {args.t} {args.n} sequence: {sequence}")
