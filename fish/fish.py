import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB

TROPICAL_FISH_MD5_0 = 0x545d53b3c344ac34
TROPICAL_FISH_MD5_1 = 0x01ffbae0fc122efa
PUFFERFISH_MD5_0 = 0xe5cf4bfff35643ed
PUFFERFISH_MD5_1 = 0xd50b88ce7e872929
SALMON_MD5_0 = 0x0e8a296f8b93cd8f
SALMON_MD5_1 = 0xa96cefee966f31ac
COD_MD5_0 = 0xcc10ec546d1f676a
COD_MD5_1 = 0xf70fe3d70293298b

STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def xNext64(state):
    l, h = state
    n = (rotl64((l + h) & MASK_64, 17) + l) & MASK_64
    h ^= l
    l_new = (rotl64(l, 49) ^ h ^ ((h << 21) & MASK_64)) & MASK_64
    h_new = rotl64(h, 28) & MASK_64
    state[0], state[1] = l_new, h_new
    return n

def next_float(state):
    return (xNext64(state) >> 11) * (1.0 / (1 << 53))

def bone_meal_kills(seed, entity, num_drops):
    if entity == "tropical_fish":
        md5_0, md5_1 = TROPICAL_FISH_MD5_0, TROPICAL_FISH_MD5_1
    elif entity == "pufferfish":
        md5_0, md5_1 = PUFFERFISH_MD5_0, PUFFERFISH_MD5_1
    elif entity == "salmon":
        md5_0, md5_1 = SALMON_MD5_0, SALMON_MD5_1
    elif entity == "cod":
        md5_0, md5_1 = COD_MD5_0, COD_MD5_1
    else:
        raise ValueError(f"'{entity}' does not match")
    
    # hash hash hash hash
    l = (seed ^ SILVER_RATIO_64) & MASK_64
    h = (l - SUBTRACT_CONSTANT) & MASK_64
    l ^= md5_0
    h ^= md5_1
    l = ((l ^ (l >> 30)) * STAFFORD_MIX_1) & MASK_64
    h = ((h ^ (h >> 30)) * STAFFORD_MIX_1) & MASK_64
    l = ((l ^ (l >> 27)) * STAFFORD_MIX_2) & MASK_64
    h = ((h ^ (h >> 27)) * STAFFORD_MIX_2) & MASK_64
    l ^= (l >> 31)
    h ^= (h >> 31)

    state = [l, h]
    kills_between = []
    count = 0
    drops_collected = 0

    while drops_collected < num_drops:
        count += 1
        if next_float(state) < 0.05:
            kills_between.append(count)
            count = 0
            drops_collected += 1

    return kills_between

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bone Meal from fish")
    parser.add_argument("seed", type=int)
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    kills_between = bone_meal_kills(args.seed, args.n, args.d)
    print(f"Seed {args.seed} {args.n} kills until next bone meal: {kills_between}")
