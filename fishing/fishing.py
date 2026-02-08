import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
FISHING_MD5_0 = 0xaa538bfd131e1849
FISHING_MD5_1 = 0x82a057801864c9b2
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

FishT = [("cod", 60), ("salmon", 25), ("tropical_fish", 2), ("pufferfish", 13)]
JunkT = [("lily_pad", 17), ("leather_boots", 10), ("leather", 10), ("bone", 10), ("water_bottle", 10), ("string", 5), ("fishing_rod (junk)", 2), ("bowl", 10), ("stick", 5), ("ink_sac 10x", 1), ("tripwire_hook", 10), ("rotten_flesh", 10)]
TreasureT = [("name_tag", 21), ("saddle", 21), ("bow", 21), ("fishing_rod (treasure)", 21), ("enchanted_book", 21), ("nautilus_shell", 21)]
MaxD = { "leather_boots": 65, "fishing_rod (junk)": 64, "fishing_rod (treasure)": 64, "bow": 384 }

def rotl(x, k):
    return ((x << k) & MASK_64) | (x >> (64 - k))

def get_state(world_seed):
    l = (world_seed ^ SILVER_RATIO_64) & MASK_64
    h = (l + GOLDEN_RATIO_64) & MASK_64
    l ^= FISHING_MD5_0
    h ^= FISHING_MD5_1
    l = ((l ^ (l >> 30)) * STAFFORD_MIX_1) & MASK_64
    h = ((h ^ (h >> 30)) * STAFFORD_MIX_1) & MASK_64
    l = ((l ^ (l >> 27)) * STAFFORD_MIX_2) & MASK_64
    h = ((h ^ (h >> 27)) * STAFFORD_MIX_2) & MASK_64
    l ^= (l >> 31)
    h ^= (h >> 31)
    return [l, h]

def xoro_next(state):
    s0, s1 = state
    result = (rotl((s0 + s1) & MASK_64, 17) + s0) & MASK_64
    s1 ^= s0
    state[0] = (rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK_64)) & MASK_64
    state[1] = rotl(s1, 28) & MASK_64
    return result

def next_int(state, bound):
    while True:
        bits = xoro_next(state) & 0xFFFFFFFF
        val = (bits * bound) >> 32
        if val < bound:
            return val

def next_float(state):
    return (xoro_next(state) >> 40) / float(1 << 24)

def get_pool_roll(state, luck):
    junk = max(10 + luck * -2, 0)
    treasure = max(5 + luck * 2, 0)
    fish = max(85 + luck * -1, 0)

    total = junk + treasure + fish
    first = next_int(state, total)

    if first < junk:
        return "junk"
    elif first < junk + treasure:
        return "treasure"
    return "fish"

def select_item(pool, state):
    table = FishT if pool == "fish" else JunkT if pool == "junk" else TreasureT
    total = sum(w for _, w in table)
    roll = next_int(state, total)
    acc = 0
    for item, w in table:
        acc += w
        if roll < acc:
            return item
    return table[-1][0]

def apply_damage(item, state, durability_cap):
    if item not in MaxD:
        return None
    max_dmg = MaxD[item]
    roll = next_float(state) * durability_cap
    remaining = max_dmg - int((1.0 - roll) * max_dmg)
    return remaining, max_dmg

def roll(state, luck):
    pool = get_pool_roll(state, luck)
    item = select_item(pool, state)

    dmg = None
    if pool == "junk":
        dmg = apply_damage(item, state, 0.9)
    elif pool == "treasure":
        dmg = apply_damage(item, state, 0.25)

    return pool, item, dmg

def simulate(seed, luck, jungle=False):
    if jungle:
        JunkT.append(("bamboo", 10))
        
    state = get_state(seed)

    for i in range(1, MAX_SEQUENCE + 1):
        pool, item, dmg = roll(state, luck)
        if dmg:
            remaining, maxd = dmg
            print(f"Catch {i}: {item} - Durability: {remaining}/{maxd}")
        else:
            print(f"Catch {i}: {item}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("-l", type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument("--jungle", action="store_true")
    args = parser.parse_args()

    simulate(args.seed, args.l, args.jungle)
