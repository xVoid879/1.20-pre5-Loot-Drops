import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
DROWNED_MD5_0 = 0x0132bb42c4601c8f
DROWNED_MD5_1 = 0x2aad08fce179e2a7
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def mix_stafford13(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
    return seed ^ (seed >> 31)

class DrownedRNG:
    def __init__(self):
        self.seedLo = 0
        self.seedHi = 0

    def set_seed(self, world_seed):
        l2 = world_seed ^ SILVER_RATIO_64
        l3 = l2 - SUBTRACT_CONSTANT
        self.seedLo = mix_stafford13(l2 ^ DROWNED_MD5_0)
        self.seedHi = mix_stafford13(l3 ^ DROWNED_MD5_1)

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

    def next_float(self):
        return (self.next_long() >> 11) * (1.0 / (1 << 53))

    def drop_rotten_flesh(self):
        return self.next_int(3)

    def drop_copper(self):
        return 1 if self.next_float() < 0.11 else 0

def simulate_drowned(seed, max_kills=MAX_SEQUENCE, target_item="rotten_flesh"):
    rng = DrownedRNG()
    rng.set_seed(seed)
    sequence = []

    for _ in range(max_kills):
        rf = rng.drop_rotten_flesh()
        cu = rng.drop_copper()
        if target_item == "rotten_flesh":
            sequence.append(rf)
        elif target_item == "copper":
            sequence.append(cu)
        else:
            raise ValueError("Wrong Item")
    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drowned drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-i", type=str, required=True, choices=["rotten_flesh", "copper"])
    args = parser.parse_args()

    seq = simulate_drowned(args.seed, args.d, args.i)
    print(f"Seed {args.seed} Drowned {args.i} drops: {seq}")
