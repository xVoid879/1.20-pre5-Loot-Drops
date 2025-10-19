import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6A09E667F3BCC909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
PHANTOM_MD5_0 = 0x0583f503414191b5
PHANTOM_MD5_1 = 0x7004291589f34d24
STAFFORD_MIX_1 = 0xBF58476D1CE4E5B9
STAFFORD_MIX_2 = 0x94D049BB133111EB
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
        self.seedLoHash = md5_lo
        self.seedHiHash = md5_hi

    def set_seed(self, world_seed: int):
        l2 = (world_seed ^ SILVER_RATIO_64) & MASK_64
        l3 = (l2 - SUBTRACT_CONSTANT) & MASK_64
        self.seedLo = mix_stafford13(l2 ^ self.seedLoHash)
        self.seedHi = mix_stafford13(l3 ^ self.seedHiHash)
        if (self.seedLo | self.seedHi) == 0:
            self.seedLo = GOLDEN_RATIO_64 & MASK_64
            self.seedHi = SILVER_RATIO_64 & MASK_64

    def next_long(self) -> int:
        l = self.seedLo
        m = self.seedHi
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

def simulate_phantom(world_seed: int, max_seq: int = MAX_SEQUENCE):
    rng = LootTableRNG(PHANTOM_MD5_0, PHANTOM_MD5_1)
    rng.set_seed(world_seed)

    seq = []
    for _ in range(max_seq):
        membrane_count = rng.next_int(2)
        seq.append(membrane_count)
    return seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phantom drop")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    seq = simulate_phantom(args.seed, args.d)
    print(f"Seed {args.seed} phantom membrane sequence: {seq}")
