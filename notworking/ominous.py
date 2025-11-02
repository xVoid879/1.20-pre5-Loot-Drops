import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6A09E667F3BCC909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
md5_0 = 0x05a13d5ce5edaab3
md5_1 = 0x1a3950a30a86bc23
STAFFORD_MIXING_1 = 0xBF58476D1CE4E5B9
STAFFORD_MIXING_2 = 0x94D049BB133111EB
MAX_SEQUENCE = 20

def rotate_left(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def stafford_mix(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIXING_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIXING_2 & MASK_64
    return (seed ^ (seed >> 31)) & MASK_64

class Vaultrng:
    # hash hash hash hash
    def __init__(self, hash_lo, hash_hi):
        self.lo = 0
        self.hi = 0
        self.hash_lo = hash_lo
        self.hash_hi = hash_hi

    def set_seed(self, seed):
        temp = (seed ^ SILVER_RATIO_64) & MASK_64
        temp2 = (temp - SUBTRACT_CONSTANT) & MASK_64
        self.lo = stafford_mix(temp ^ self.hash_lo)
        self.hi = stafford_mix(temp2 ^ self.hash_hi)
        if self.lo | self.hi == 0:
            self.lo = GOLDEN_RATIO_64 & MASK_64
            self.hi = SILVER_RATIO_64 & MASK_64

    def next_long(self):
        l, m = self.lo, self.hi
        n = (rotate_left((l + m) & MASK_64, 17) + l) & MASK_64
        m ^= l
        self.lo = (rotate_left(l, 49) ^ m ^ ((m << 21) & MASK_64)) & MASK_64
        self.hi = rotate_left(m, 28) & MASK_64
        return n

    def next_int(self, bound):
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

# loot table
RARE = [
    ("emerald_block", 5),
    ("iron_block", 4),
    ("crossbow", 4),
    ("golden_apple", 3),
    ("diamond_axe", 3),
    ("diamond_chestplate", 3),
    ("enchanted_book_knockback", 2),
    ("enchanted_book_breach", 2),
    ("book_wind_burst", 2),
    ("diamond_block", 1)
]

COMMON = [
    ("emerald", 5, (4,10)),
    ("wind_charge", 4, (8,12)),
    ("tipped_arrow_slowness", 3, (4,12)),
    ("diamond", 2, (2,3)),
    ("ominous_bottle", 1, (1,1))
]

UNIQUE = [
    ("enchanted_golden_apple", 3),
    ("flow_armor_trim_smithing_template", 3),
    ("flow_banner_pattern", 2),
    ("music_disc_creator", 1),
    ("heavy_core", 1)
]

def roll_item(rng, items):
    total_weight = sum(w for _, w, *rest in items)
    roll = rng.next_int(total_weight)
    cum = 0
    for entry in items:
        name, weight = entry[0], entry[1]
        count_range = entry[2] if len(entry) > 2 else (1,1)
        cum += weight
        if roll < cum:
            count = rng.next_int(count_range[1] - count_range[0] + 1) + count_range[0] \
                    if count_range[1] > count_range[0] else count_range[0]
            return name, count
    return items[-1][0], 1

def simulate_vault(seed, max_chests=MAX_SEQUENCE):
    rng = Vaultrng(md5_0, md5_1)
    rng.set_seed(seed)
    results = []

    for _ in range(max_chests):
        chest = []

        if rng.next_int(10) < 8:
            chest.append(roll_item(rng, RARE))
        else:
            chest.append(roll_item(rng, COMMON))
        for _ in range(rng.next_int(3) + 1):
            chest.append(roll_item(rng, COMMON))
        if rng.next_int(4) < 3:
            unique = roll_item(rng, UNIQUE)
            chest.append(("unique: " + unique[0], unique[1]))
        final = {}
        for item, count in chest:
            final[item] = final.get(item, 0) + count

        results.append(list(final.items()))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vault")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    drops = simulate_vault(args.seed, args.d)
    for i, chest in enumerate(drops, 1):
        print(f"chest {i}:")
        for item, count in chest:
            print(f"  {item} x{count}")
