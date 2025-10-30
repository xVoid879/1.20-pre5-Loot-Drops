# ALL BUT FLETCHERS

import sys

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64   = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
ARMORER_MD5_0 = 0x91462ee6df2038e6
ARMORER_MD5_1 = 0x793a326d3b10ccdd
BUTCHER_MD5_0 = 0xe595d235951351b4
BUTCHER_MD5_1 = 0x9a004375849863aa
CARTOGRAPHER_MD5_0 = 0xfe8b41b51c76c5d9
CARTOGRAPHER_MD5_1 = 0x46c8db7e472882c5
CLERIC_GIFT_MD5_0 = 0x91c68e52e84cf575
CLERIC_GIFT_MD5_1 = 0x4c5693fcf5431c4e
FARMER_GIFT_MD5_0 = 0x4d1111d356a5d80f
FARMER_GIFT_MD5_1 = 0xf8d105389dd1d819
FISHERMAN_GIFT_MD5_0 = 0xb58d8dab542b4f21
FISHERMAN_GIFT_MD5_1 = 0xa068c1809ba2e05f
SHEPHERD_GIFT_MD5_0 = 0x00001d26f823d60a
SHEPHERD_GIFT_MD5_1 = 0xc2a617d2391a50c2
TOOLSMITH_GIFT_MD5_0 = 0xb8f6b20075f58e57
TOOLSMITH_GIFT_MD5_1 = 0xb3fbd3f74ee89c52
WEAPONSMITH_GIFT_MD5_0 = 0x9ba236536deb330d
WEAPONSMITH_GIFT_MD5_1 = 0x7e627441a8f8c1e3
STAFFORD_MIXING_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIXING_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

Tables = {
    "armorer": {
        "md5_0": ARMORER_MD5_0,
        "md5_1": ARMORER_MD5_1,
        "items": [
            {"name": "Chainmail Helmet", "weight": 1},
            {"name": "Chainmail Chestplate", "weight": 1},
            {"name": "Chainmail Leggings", "weight": 1},
            {"name": "Chainmail Boots", "weight": 1},
        ]
    },
    "butcher": {
        "md5_0": BUTCHER_MD5_0,
        "md5_1": BUTCHER_MD5_1,
        "items": [
            {"name": "Cooked Rabbit", "weight": 1},
            {"name": "Cooked Chicken", "weight": 1},
            {"name": "Cooked Porkchop", "weight": 1},
            {"name": "Cooked Beef", "weight": 1},
            {"name": "Cooked Mutton", "weight": 1},
        ]
    },            
    "cleric": {
        "md5_0": CLERIC_GIFT_MD5_0,
        "md5_1": CLERIC_GIFT_MD5_1,
        "items": [
            {"name": "Redstone", "weight": 1},
            {"name": "Lapis", "weight": 1},
        ]
    },
    "cartographer": {
        "md5_0": CARTOGRAPHER_MD5_0,
        "md5_1": CARTOGRAPHER_MD5_1,
        "items": [
            {"name": "Map", "weight": 1},
            {"name": "Paper", "weight": 1},
        ]
    },
    "farmer": {
        "md5_0": FARMER_GIFT_MD5_0,
        "md5_1": FARMER_GIFT_MD5_1,
        "items": [
            {"name": "Bread", "weight": 1},
            {"name": "Pumpkin Pie", "weight": 1},
            {"name": "Cookie", "weight": 1},
        ]
    },
    "fisherman": {
        "md5_0": FISHERMAN_GIFT_MD5_0,
        "md5_1": FISHERMAN_GIFT_MD5_1,
        "items": [
            {"name": "Cod", "weight": 1},
            {"name": "Salmon", "weight": 1},
        ]
    },
    "shepherd": {
        "md5_0": SHEPHERD_GIFT_MD5_0,
        "md5_1": SHEPHERD_GIFT_MD5_1,
        "items": [
            {"name": "White Wool", "weight": 1},
            {"name": "Orange Wool", "weight": 1},
            {"name": "Magenta Wool", "weight": 1},
            {"name": "Light Blue Wool", "weight": 1},
            {"name": "Yellow Wool", "weight": 1},
            {"name": "Lime Wool", "weight": 1},
            {"name": "Pink Wool", "weight": 1},
            {"name": "Gray Wool", "weight": 1},
            {"name": "Light Gray Wool", "weight": 1},
            {"name": "Cyan Wool", "weight": 1},
            {"name": "Purple Wool", "weight": 1},
            {"name": "Blue Wool", "weight": 1},
            {"name": "Brown Wool", "weight": 1},
            {"name": "Green Wool", "weight": 1},
            {"name": "Red Wool", "weight": 1},
            {"name": "Black Wool", "weight": 1},
        ]
    },
    "toolsmith": {
        "md5_0": TOOLSMITH_GIFT_MD5_0,
        "md5_1": TOOLSMITH_GIFT_MD5_1,
        "items": [
            {"name": "Stone Pickaxe", "weight": 1},
            {"name": "Stone Axe", "weight": 1},
            {"name": "Stone Hoe", "weight": 1},
            {"name": "Stone Shovel", "weight": 1},
        ]
    },
    "weaponsmith": {
        "md5_0": WEAPONSMITH_GIFT_MD5_0,
        "md5_1": WEAPONSMITH_GIFT_MD5_1,
        "items": [
            {"name": "Stone Axe", "weight": 1},
            {"name": "Golden Axe", "weight": 1},
            {"name": "Iron Axe", "weight": 1},
        ]
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

def simulate(world_seed, villager_type, n=MAX_SEQUENCE):
    if villager_type not in Tables:
        raise ValueError(f"?: {villager_type}")

    v = Tables[villager_type]
    state = get_state(world_seed, v["md5_0"], v["md5_1"])
    gifts = [roll(state, v["items"]) for _ in range(n)]
    return gifts

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python FILE.py seed villagertype")
        sys.exit(1)

    seed = int(sys.argv[1])
    villager_type = sys.argv[2].lower()

    seq = simulate(seed, villager_type, MAX_SEQUENCE)
    for i, item in enumerate(seq, 1):
        print(f"{villager_type.capitalize()} Gift {i}: {item}")
