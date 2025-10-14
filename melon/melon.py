# Initial From (https://gist.github.com/mjtb49/f3e01e3355178d2bb6c814606971c374#file-loottablerng-java) by Matthew for 1.20-pre4
# This is 1.20-pre5

import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF
MAX_SEQUENCE = 20

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
MELON_MD5_0 = 0x45e0c0d79db027a8
MELON_MD5_1 = 0xafe0a44cafba7e37
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb

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
        self.seedLoHash = MELON_MD5_0
        self.seedHiHash = MELON_MD5_1

    def set_seed(self, seed):
        l2 = seed ^ SILVER_RATIO_64
        l3 = l2 + - SUBTRACT_CONSTANT

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
            raise ValueError("Bound must be positive")
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

    def melon_drop(self):
        val = self.next_int(5) + 3  # so picks 0-4, adds 3 = melons
        self.next_long()
        return val

def melon_sequence(seed, max_seq=MAX_SEQUENCE):
    rng = LootTableRNG()
    rng.set_seed(seed)
    sequence = []
    for _ in range(max_seq):
        sequence.append(rng.melon_drop())
    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate melon drops.")
    parser.add_argument("seed", type=int, help="World Seed")
    parser.add_argument("-d", "--depth", type=int, default=MAX_SEQUENCE,
                        help=f"# of melon drops to generate (default {MAX_SEQUENCE})")
    
    args = parser.parse_args()

    seq = melon_sequence(args.seed, args.depth)

    print(f"Seed {args.seed} melon drops: {seq}")
