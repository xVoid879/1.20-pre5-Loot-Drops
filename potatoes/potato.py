import sys
import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
POTATO_MD5_0 = 0xedbc0e0223e5e29e
POTATO_MD5_1 = 0xd1fcda23b4868d23
STAFFORD_MIXING_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIXING_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def mix_stafford13(seed):
    seed ^= (seed >> 30) & MASK_64
    seed = (seed * STAFFORD_MIXING_1) & MASK_64
    seed ^= (seed >> 27) & MASK_64
    seed = (seed * STAFFORD_MIXING_2) & MASK_64
    seed ^= (seed >> 31) & MASK_64
    return seed & MASK_64

class Xoroshiro128Plus: # hash hash hash hash
    def __init__(self, seed0, seed1):
        self.seed0 = seed0 & MASK_64
        self.seed1 = seed1 & MASK_64

    def next_long(self):
        s0, s1 = self.seed0, self.seed1
        result = (s0 + s1) & MASK_64
        result = ((result << 17) | (result >> 47)) & MASK_64
        result = (result + s0) & MASK_64

        s1 ^= s0
        self.seed0 = ((s0 << 49) | (s0 >> 15)) & MASK_64
        self.seed0 ^= s1
        self.seed0 ^= ((s1 << 21) & MASK_64)
        self.seed0 &= MASK_64
        self.seed1 = ((s1 << 28) | (s1 >> 36)) & MASK_64

        return result

    def next_float(self):
        return ((self.next_long() >> 40) & 0xFFFFFF) / 16777216.0

class PotatoDrops:
    def __init__(self, world_seed):
        lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
        hi = (lo - SUBTRACT_CONSTANT) & MASK_64
        mixed_lo = mix_stafford13(lo ^ POTATO_MD5_0)
        mixed_hi = mix_stafford13(hi ^ POTATO_MD5_1)
        self.rng = Xoroshiro128Plus(mixed_lo, mixed_hi)

    def next_drop(self):
        potato_count = 1
        extra_rolls = 3
        probability = 4/7
        for _ in range(extra_rolls):
            if self.rng.next_float() < probability:
                potato_count += 1
        potato_count += 1

        poisonous = self.rng.next_float() < 0.02

        return potato_count, poisonous

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Potato Crop Drop")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-i", choices=["normal", "pos"], required=True)
    args = parser.parse_args()

    sim = PotatoDrops(args.seed)
    sequence = []

    for _ in range(args.d):
        count, poisonous = sim.next_drop()
        if args.i == "normal":
            sequence.append(count)
        else:
            sequence.append(1 if poisonous else 0)

    if args.i == "normal":
        print(f"Seed {args.seed} potato crop drop sequence: {sequence}")
    else:
        print(f"Seed {args.seed} poisonous potato crop drop sequence: {sequence}")
