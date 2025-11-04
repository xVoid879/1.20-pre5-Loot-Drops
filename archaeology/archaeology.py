# ALL BUT DESERT WELL

import sys

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64   = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
DESERT_PYRAMID_MD5_0 = 0xc915c5ba1b22df8a
DESERT_PYRAMID_MD5_1 = 0x4b1d34289df9e1ea
OCEAN_RUIN_COLD_MD5_0 = 0x2522d5aa4bca1474
OCEAN_RUIN_COLD_MD5_1 = 0x6647be1330619a47
OCEAN_RUIN_WARM_MD5_0 = 0x71a014056afb5255
OCEAN_RUIN_WARM_MD5_1 = 0x55c930c888ddcc5d
TRAIL_RUINS_COMMON_MD5_0 = 0x4de31a80763d3b9d
TRAIL_RUINS_COMMON_MD5_1 = 0x7d42f7caaac32dc8
TRAIL_RUINS_RARE_MD5_0 = 0xd3fe5c1136117cba
TRAIL_RUINS_RARE_MD5_1 = 0x3ffeed7a07ec0ad6
STAFFORD_MIXING_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIXING_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

Tables = {
    "desert_pyramid": {
        "md5_0": DESERT_PYRAMID_MD5_0,
        "md5_1": DESERT_PYRAMID_MD5_1,
        "items": [
            {"name": "Archer Pottery Shard", "weight": 1},
            {"name": "Miner Pottery Shard", "weight": 1},
            {"name": "Prize Pottery Shard", "weight": 1},
            {"name": "Skull Pottery Shard", "weight": 1},
            {"name": "Diamond", "weight": 1},
            {"name": "TnT", "weight": 1},
            {"name": "Gunpowder", "weight": 1},
            {"name": "Emerald", "weight": 1},
        ],
    },

    "ocean_ruin_cold": {
        "md5_0": OCEAN_RUIN_COLD_MD5_0,
        "md5_1": OCEAN_RUIN_COLD_MD5_1,
        "items": [
            {"name": "Blade Pottery Shard", "weight": 1},
            {"name": "Explorer Pottery Shard", "weight": 1},
            {"name": "Mourner Pottery Shard", "weight": 1},
            {"name": "Plenty Pottery Shard", "weight": 1},
            {"name": "Iron Axe", "weight": 1},
            {"name": "Emerald", "weight": 2},
            {"name": "Wheat", "weight": 2},
            {"name": "Wooden Hoe", "weight": 2},
            {"name": "Coal", "weight": 2},
            {"name": "Gold Nugget", "weight": 2},
        ],
    },

    "ocean_ruin_warm": {
        "md5_0": OCEAN_RUIN_WARM_MD5_0,
        "md5_1": OCEAN_RUIN_WARM_MD5_1,
        "items": [
            {"name": "Angler Pottery Shard", "weight": 1},
            {"name": "Shelter Pottery Shard", "weight": 1},
            {"name": "Snort Pottery Shard", "weight": 1},
            {"name": "Sniffer Egg", "weight": 1},
            {"name": "Iron Axe", "weight": 1},
            {"name": "Emerald", "weight": 2},
            {"name": "Wheat", "weight": 2},
            {"name": "Wooden Hoe", "weight": 2},
            {"name": "Coal", "weight": 2},
            {"name": "Gold Nugget", "weight": 2},
        ],
    },

    "trail_ruins_common": {
        "md5_0": TRAIL_RUINS_COMMON_MD5_0,
        "md5_1": TRAIL_RUINS_COMMON_MD5_1,
        "items": [
            {"name": "Emerald", "weight": 2},
            {"name": "Wheat", "weight": 2},
            {"name": "Wooden Hoe", "weight": 2},
            {"name": "Clay", "weight": 2},
            {"name": "Brick", "weight": 2},
            {"name": "Yellow Dye", "weight": 2},
            {"name": "Blue Dye", "weight": 2},
            {"name": "Light Blue Dye", "weight": 2},
            {"name": "White Dye", "weight": 2},
            {"name": "Orange Dye", "weight": 2},
            {"name": "Red Candle", "weight": 2},
            {"name": "Green Candle", "weight": 2},
            {"name": "Purple Candle", "weight": 2},
            {"name": "Brown Candle", "weight": 2},
            {"name": "Magenta Stained Glass Pane", "weight": 1},
            {"name": "Pink Stained Glass Pane", "weight": 1},
            {"name": "Blue Stained Glass Pane", "weight": 1},
            {"name": "Light Blue Stained Glass Pane", "weight": 1},
            {"name": "Red Stained Glass Pane", "weight": 1},
            {"name": "Yellow Stained Glass Pane", "weight": 1},
            {"name": "Purple Stained Glass Pane", "weight": 1},
            {"name": "Spruce Hanging Sign", "weight": 1},
            {"name": "Oak Hanging Sign", "weight": 1},
            {"name": "Gold Nugget", "weight": 1},
            {"name": "Coal", "weight": 1},
            {"name": "Wheat Seeds", "weight": 1},
            {"name": "Beetroot Seeds", "weight": 1},
            {"name": "Dead Bush", "weight": 1},
            {"name": "Flower Pot", "weight": 1},
            {"name": "String", "weight": 1},
            {"name": "Lead", "weight": 1},
        ],
    },

    "trail_ruins_rare": {
        "md5_0": TRAIL_RUINS_RARE_MD5_0,
        "md5_1": TRAIL_RUINS_RARE_MD5_1,
        "items": [
            {"name": "Burn Pottery Shard", "weight": 1},
            {"name": "Danger Pottery Shard", "weight": 1},
            {"name": "Friend Pottery Shard", "weight": 1},
            {"name": "Heart Pottery Shard", "weight": 1},
            {"name": "Heartbreak Pottery Shard", "weight": 1},
            {"name": "Howl Pottery Shard", "weight": 1},
            {"name": "Sheaf Pottery Shard", "weight": 1},
            {"name": "Wayfinder Armor Trim Smithing Template", "weight": 1},
            {"name": "Raiser Armor Trim Smithing Template", "weight": 1},
            {"name": "Shaper Armor Trim Smithing Template", "weight": 1},
            {"name": "Host Armor Trim Smithing Template", "weight": 1},
            {"name": "Music Disc Relic", "weight": 1},
        ],
    },
}

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

def get_state(world_seed, md5_0, md5_1):
    unmixed_lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    unmixed_hi = (unmixed_lo - SUBTRACT_CONSTANT) & MASK_64
    l = mix_stafford(unmixed_lo ^ md5_0)
    h = mix_stafford(unmixed_hi ^ md5_1)
    return [l, h]

def next_int(state, bound):
    while True:
        bits = xoro_next(state) & 0xFFFFFFFF
        val = (bits * bound) >> 32
        if val < bound:
            return val

def roll(state, items):
    total_weight = sum(e["weight"] for e in items)
    j = next_int(state, total_weight)
    for e in items:
        j -= e["weight"]
        if j < 0:
            return e["name"]

def simulate(world_seed, archaeology_site, n=MAX_SEQUENCE):
    if archaeology_site not in Tables:
        raise ValueError(f"?: {archaeology_site}")

    v = Tables[archaeology_site]
    state = get_state(world_seed, v["md5_0"], v["md5_1"])
    gifts = [roll(state, v["items"]) for _ in range(n)]
    return gifts

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python FILE.py seed archaeologysite")
        sys.exit(1)

    seed = int(sys.argv[1])
    archaeology_site = sys.argv[2].lower()

    seq = simulate(seed, archaeology_site, MAX_SEQUENCE)
    for i, item in enumerate(seq, 1):
        print(f"{archaeology_site.capitalize()} Dig {i}: {item}")
