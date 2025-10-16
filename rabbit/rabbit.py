import argparse
import sys

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
RABBIT_MD5_0 = 0xeea1524ee8878676
RABBIT_MD5_1 = 0x15d6e2b784f14012
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
        self.seedLoHash = RABBIT_MD5_0
        self.seedHiHash = RABBIT_MD5_1

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

def rabbit_hide_drop(rng):
    return rng.next_int(2)

def rabbit_foot_drop(rng):
    r = (rng.next_long() >> 11) / float(1 << 53)
    return 1 if r < 0.1 else 0

def rabbit_sequence(seed, max_seq=MAX_SEQUENCE):
    rng = LootTableRNG()
    rng.set_seed(seed)
    hide_seq = []
    foot_seq = []
    for _ in range(max_seq):
        hide_seq.append(rabbit_hide_drop(rng))
        foot_seq.append(rabbit_foot_drop(rng))
    return hide_seq, foot_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rabbit Drop Simulation")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-n", choices=["hide", "foot"], default="both")

    args = parser.parse_args()
    hide_seq, foot_seq = rabbit_sequence(args.seed, args.d)

    if args.n == "hide":
        print(f"Seed {args.seed} rabbit hide sequence:\n{hide_seq}")
    elif args.n == "foot":
        print(f"Seed {args.seed} rabbit foot sequence:\n{foot_seq}")
    else:
        print("Error: -n argument must be 'hide' or 'foot'", file=sys.stderr)
        sys.exit(1)
