# Simulates carrot, iron, and potato drops from zombies or husks
import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
ZOMBIE_MD5_0 = 0xf5549fb67eceeb03
ZOMBIE_MD5_1 = 0x4c45b69e40dba4ce
HUSK_MD5_0 = 0x3c2b70d12dc2f30d
HUSK_MD5_1 = 0xc54f6485a3853c0a
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))


def mix_stafford13(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
    return seed ^ (seed >> 31)


def get_xoro_state(world_seed: int, mob: str):
    unmixed_lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    unmixed_hi = (unmixed_lo - SUBTRACT_CONSTANT) & MASK_64

    if mob == "zombie":
        lo = mix_stafford13(unmixed_lo ^ ZOMBIE_MD5_0)
        hi = mix_stafford13(unmixed_hi ^ ZOMBIE_MD5_1)
    elif mob == "husk":
        lo = mix_stafford13(unmixed_lo ^ HUSK_MD5_0)
        hi = mix_stafford13(unmixed_hi ^ HUSK_MD5_1)
    else:
        raise ValueError("Wrong mob type")

    return [lo, hi]


def xoro_next(state):
    s0, s1 = state
    result = (rotl64((s0 + s1) & MASK_64, 17) + s0) & MASK_64
    s1 ^= s0
    state[0] = (rotl64(s0, 49) ^ s1 ^ ((s1 << 21) & MASK_64)) & MASK_64
    state[1] = rotl64(s1, 28) & MASK_64
    return result


def next_float(state):
    return (xoro_next(state) >> 11) * (1.0 / (1 << 53))


def roll_uniform(state, min_val: float, max_val: float) -> float:
    return min_val + (max_val - min_val) * next_float(state)


def drop_rotten_flesh(state):
    count = roll_uniform(state, 0.0, 2.0)
    return int(count + 0.5)


def next_int(state, bound):
    if bound <= 0:
        raise ValueError("bound must be positive")
    l = xoro_next(state) & 0xFFFFFFFF
    m = l * bound
    low = m & 0xFFFFFFFF
    if low < bound:
        t = (-bound) % bound
        while low < t:
            l = xoro_next(state) & 0xFFFFFFFF
            m = l * bound
            low = m & 0xFFFFFFFF
    return (m >> 32) & 0xFFFFFFFF


def choose_loot(state):
    idx = next_int(state, 3)
    return ["iron", "carrot", "potato"][idx]


def drop_rare(state):
    if next_float(state) < 0.025:
        return choose_loot(state)
    return None


def simulate_mob(world_seed: int, mob: str, max_drops, target_item: str):
    state = get_xoro_state(world_seed, mob)
    intervals = []
    kills = 0
    last_drop = 0

    while len(intervals) < max_drops:
        kills += 1
        drop_rotten_flesh(state)
        rare = drop_rare(state)
        if rare == target_item:
            interval = kills - last_drop
            intervals.append(interval)
            last_drop = kills
    return intervals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zombie/Husk drop simulation")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE, help="max rare drops to simulate")
    parser.add_argument("-n", type=str, required=True, choices=["zombie", "husk"])
    parser.add_argument("-i", type=str, required=True, choices=["iron", "carrot", "potato"])
    args = parser.parse_args()

    seq = simulate_mob(args.seed, args.n, args.d, args.i)
    print(f"Seed {args.seed} ({args.n}) {args.i} drops: {seq}")
