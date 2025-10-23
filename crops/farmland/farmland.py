import sys
import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
BEETROOT_MD5_0 = 0x583ba9feba08f50d
BEETROOT_MD5_1 = 0x96aa01fe8ac675e3
CARROT_MD5_0   = 0x13bb704d54b53096
CARROT_MD5_1   = 0xfda37b628e9c98c7
POTATO_MD5_0   = 0xedbc0e0223e5e29e
POTATO_MD5_1   = 0xd1fcda23b4868d23
WHEAT_MD5_0    = 0x9fa698f5eb5f5287
WHEAT_MD5_1    = 0xddc12ce50a812942
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

class Xoroshiro128Plus:
    def __init__(self, seed0, seed1):
        self.seed0 = seed0 & MASK_64
        self.seed1 = seed1 & MASK_64

    def next_long(self):
        s0, s1 = self.seed0, self.seed1
        result = (s0 + s1) & MASK_64
        result = ((s0 + s1) & MASK_64)
        result = ((result << 17) | (result >> 47)) & MASK_64
        result = (result + s0) & MASK_64

        s1 ^= s0
        self.seed0 = ((s0 << 49) | (s0 >> 15)) & MASK_64
        self.seed0 ^= s1
        self.seed0 ^= (s1 << 21) & MASK_64
        self.seed1 = ((s1 << 28) | (s1 >> 36)) & MASK_64

        self.seed0 &= MASK_64
        self.seed1 &= MASK_64
        return result

    def next_float(self):
        return ((self.next_long() >> 40) & 0xFFFFFF) / 16777216.0

    def next_int(self, bound):
        return int(self.next_float() * bound)

class CropDrops:
    def __init__(self, world_seed, crop_name):
        if crop_name == "potato":
            md5_0, md5_1 = POTATO_MD5_0, POTATO_MD5_1
        elif crop_name == "carrot":
            md5_0, md5_1 = CARROT_MD5_0, CARROT_MD5_1
        elif crop_name == "beetroot":
            md5_0, md5_1 = BEETROOT_MD5_0, BEETROOT_MD5_1
        elif crop_name == "wheat":
            md5_0, md5_1 = WHEAT_MD5_0, WHEAT_MD5_1
        else:
            raise ValueError("? Farmland Crop Name")

        lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
        hi = (lo - SUBTRACT_CONSTANT) & MASK_64
        mixed_lo = mix_stafford13(lo ^ md5_0)
        mixed_hi = mix_stafford13(hi ^ md5_1)
        self.rng = Xoroshiro128Plus(mixed_lo, mixed_hi)
        self.crop = crop_name

    def next_drop(self):
        if self.crop in ("potato", "carrot"):
            count = 1
            for _ in range(3):
                if self.rng.next_float() < 4 / 7:
                    count += 1
            count += 1
            poisonous = False
            if self.crop == "potato":
                poisonous = self.rng.next_float() < 0.02
            return count, poisonous

        elif self.crop in ("wheat", "beetroot"):
            seeds = 0
            for _ in range(3):
                if self.rng.next_float() < 4 / 7:
                    seeds += 1
            return seeds + 1, False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Farmland Crop Drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-n", choices=["potato", "carrot", "beetroot", "wheat"], required=True)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-i", choices=["normal", "pos"])
    args = parser.parse_args()

    if args.n != "potato" and args.i:
        parser.error("-i only for potatoes")

    if args.n == "potato" and args.i is None:
        parser.error("potatoes need -i (normal, pos)")

    sim = CropDrops(args.seed, args.n)
    sequence = []

    for _ in range(args.d):
        count, poisonous = sim.next_drop()
        if args.n == "potato" and args.i == "pos":
            sequence.append(1 if poisonous else 0)
        else:
            sequence.append(count)

    if args.n == "potato" and args.i == "pos":
        print(f"Seed {args.seed} poisonous potato sequence: {sequence}")
    elif args.n in ("wheat", "beetroot"):
        print(f"Seed {args.seed} {args.n} seed drop sequence: {sequence}")
    else:
        print(f"Seed {args.seed} {args.n} crop drop sequence: {sequence}")
