import sys

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
PANDA_SNEEZE_MD5_0 = 0x72a8f9a37353b657
PANDA_SNEEZE_MD5_1 = 0xbc01ceb398ec0fe6
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
Weight = 700  # 1 slimy balls, 699 not
MAX_SEQUENCE = 20

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def mix_stafford(seed):
    seed ^= (seed >> 30) & MASK_64
    seed = (seed * STAFFORD_MIX_1) & MASK_64
    seed ^= (seed >> 27) & MASK_64
    seed = (seed * STAFFORD_MIX_2) & MASK_64
    seed ^= (seed >> 31) & MASK_64
    return seed & MASK_64

class Xoroshiro128Plus: # defining functions not hashing
    def __init__(self, seed0, seed1):
        self.seed0 = seed0 & MASK_64
        self.seed1 = seed1 & MASK_64

    def next_long(self): # turn l and h after stafford to 64
        s0 = self.seed0
        s1 = self.seed1
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

    def next_int(self, bound):
        if bound <= 0:
            raise ValueError("bound must be positive")
        while True:
            bits = self.next_long() & 0xFFFFFFFF
            val = (bits * bound) >> 32
            if val < bound:
                return val

class PandaSneezeLoot: # hashing
    def __init__(self, world_seed):
        lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
        hi = (lo - SUBTRACT_CONSTANT) & MASK_64
        mixed_lo = mix_stafford(lo ^ PANDA_SNEEZE_MD5_0)
        mixed_hi = mix_stafford(hi ^ PANDA_SNEEZE_MD5_1)
        self.rng = Xoroshiro128Plus(mixed_lo, mixed_hi)

    def next_drop(self):
        roll = self.rng.next_int(Weight) # picks 0-699
        return roll == 0 # if 0, gives slime ball

def slime_sequence(seed, max_slime=MAX_SEQUENCE):
    loot = PandaSneezeLoot(seed)
    sequence = []
    count_since_last = 0
    slimes_found = 0

    while slimes_found < max_slime:
        count_since_last += 1
        if loot.next_drop():
            sequence.append(count_since_last)
            count_since_last = 0
            slimes_found += 1

    return sequence

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sequence = slime_sequence(seed, MAX_SEQUENCE)
    print(f"Seed {seed} slime ball panda sneeze sequence: {sequence}")
