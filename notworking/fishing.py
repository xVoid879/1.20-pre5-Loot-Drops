#it does seed 1 correctly no lots didn't test other lots, seed 2 no lots, seed 13 no lots, seed 13 3 lots
#all correct
#but for seed 8 only lots 3 it says first is enchanted book but it's salmon
#and for seed 44, the first two are correct (lots doesn't change first 5), but the next three are not, one of the three being in the same pool 
import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
FISHING_MD5_0 = 0xaa538bfd131e1849
FISHING_MD5_1 = 0x82a057801864c9b2
STAFFORD_MIXING_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIXING_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

FishT = [("cod", 60), ("salmon", 25), ("tropical_fish", 2), ("pufferfish", 13)]
JunkT = [("lily_pad", 17), ("leather_boots", 10), ("leather", 10), ("bone", 10), ("potion", 10), ("string", 5), ("fishing_rod", 2), ("bowl", 10), ("stick", 5), ("ink_sac", 1), ("tripwire_hook", 10), ("rotten_flesh", 10)]
TreasureT = [("name_tag", 21), ("saddle", 21), ("bow", 21), ("fishing_rod", 21), ("enchanted_book", 21), ("nautilus_shell", 21)]

def mix_stafford(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIXING_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIXING_2 & MASK_64
    seed ^= (seed >> 31)
    return seed & MASK_64


def rotl(x, k):
    return ((x << k) & MASK_64) | (x >> (64 - k))


def xoro_next(state):
    s0, s1 = state
    result = (rotl((s0 + s1) & MASK_64, 17) + s0) & MASK_64
    s1 ^= s0
    state[0] = (rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK_64)) & MASK_64
    state[1] = rotl(s1, 28) & MASK_64
    return result


def get_state(world_seed):
    unmixed_lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    unmixed_hi = (unmixed_lo - SUBTRACT_CONSTANT) & MASK_64
    l = mix_stafford(unmixed_lo ^ FISHING_MD5_0)
    h = mix_stafford(unmixed_hi ^ FISHING_MD5_1)
    return [l, h]


def next_int(state, bound):
    while True:
        bits = xoro_next(state) & 0xFFFFFFFF
        val = (bits * bound) >> 32
        if val < bound:
            return val


def get_pool(first_roll, luck_of_the_sea):
    if luck_of_the_sea == 0:
        junk, treasure = 10.0, 0.0
    elif luck_of_the_sea == 1:
        junk, treasure = 8.5, 7.3
    elif luck_of_the_sea == 2:
        junk, treasure = 7.1, 9.5
    elif luck_of_the_sea >= 3:
        junk, treasure = 5.7, 11.7

    if first_roll < junk:
        return "junk"
    elif first_roll < junk + treasure:
        return "treasure"
    else:
        return "fish"


def select_item(pool, second_roll, junk_table):
    table = FishT if pool == "fish" else junk_table if pool == "junk" else TreasureT
    total_weight = sum(w for _, w in table)
    roll_point = second_roll * total_weight // 100
    running_sum = 0
    for item, weight in table:
        running_sum += weight
        if roll_point < running_sum:
            return item
    return table[-1][0]


def roll(state, luck_of_the_sea, junk_table):
    first = next_int(state, 100)
    pool = get_pool(first, luck_of_the_sea)
    second = next_int(state, 100)
    item = select_item(pool, second, junk_table)
    return pool, first, second, item


def simulate(world_seed, luck_of_the_sea, jungle=False):
    junk_table = JunkT.copy()
    if jungle:
        junk_table.append(("bamboo", 10))
    state = get_state(world_seed)
    for i in range(1, MAX_SEQUENCE + 1):
        _, _, _, item = roll(state, luck_of_the_sea, junk_table)
        print(f"Catch {i}: {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("-l", type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument("--jungle", action="store_true")
    args = parser.parse_args()
    simulate(args.seed, args.l, args.jungle)
