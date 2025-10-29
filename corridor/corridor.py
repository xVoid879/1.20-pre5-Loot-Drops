import sys

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
CORRIDOR_MD5_0 = 0x8973e5e2cbd56a77
CORRIDOR_MD5_1 = 0xc1cd9670021eee91
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

Items = [
    {"name": "minecraft:emerald", "weight": 125, "min": 1, "max": 3},
    {"name": "minecraft:arrow", "weight": 100, "min": 2, "max": 8},
    {"name": "minecraft:iron_ingot", "weight": 100, "min": 1, "max": 2},
    {"name": "minecraft:trial_key", "weight": 10, "min": 1, "max": 1},
    {"name": "minecraft:music_disc_creator_music_box", "weight": 5, "min": 1, "max": 1},
    {"name": "minecraft:diamond", "weight": 5, "min": 1, "max": 2},
    {"name": "minecraft:emerald_block", "weight": 5, "min": 1, "max": 1},
    {"name": "minecraft:diamond_block", "weight": 1, "min": 1, "max": 1},
]

Total_Weight = 351

def mix_stafford(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
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
    l = mix_stafford(unmixed_lo ^ CORRIDOR_MD5_0)
    h = mix_stafford(unmixed_hi ^ CORRIDOR_MD5_1)
    return [l, h]

def next_int(state, bound):
    if bound <= 0:
        raise ValueError("bound must be positive")
    while True:
        bits = xoro_next(state) & 0xFFFFFFFF
        val = (bits * bound) >> 32
        if val < bound:
            return val

def roll(state):
    j = next_int(state, Total_Weight)
    for e in Items:
        j -= e["weight"]
        if j < 0:
            if e["min"] != e["max"]:
                count = next_int(state, e["max"] - e["min"] + 1) + e["min"]
            else:
                count = e["min"]
            return {"name": e["name"], "count": count}

def simulate(world_seed, n=MAX_SEQUENCE):
    state = get_state(world_seed)
    rolls = []
    for _ in range(n):
        rolls.append(roll(state))
    return rolls

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Wrong argument")
        sys.exit(1)

    seed = int(sys.argv[1])
    results = simulate(seed, MAX_SEQUENCE)
    for i, item in enumerate(results, 1):
        print(f"Roll {i}: {item['name']} x{item['count']}")
