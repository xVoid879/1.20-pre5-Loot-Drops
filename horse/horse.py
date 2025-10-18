import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
DONKEY_MD5_0 = 0xbc25a06cb050f6bf
DONKEY_MD5_1 = 0x146b9eada6a045a6
HORSE_MD5_0 = 0x763f1e5c2907462e
HORSE_MD5_1 = 0x88961b1db84c2cc3
LLAMA_MD5_0 = 0x27d1d5ca32afd0e1
LLAMA_MD5_1 = 0xfc4082104d0abab9
MULE_MD5_0 = 0x4c679d200080eb9e
MULE_MD5_1 = 0xa3371c4b2e6aa4a6
TRADER_LLAMA_MD5_0 = 0x483fa57e3fd07828
TRADER_LLAMA_MD5_1 = 0x3678e8e5f105e31b
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

def java_next_int(state, bound):
    while True:
        i = xNext64(state) & 0xFFFFFFFF
        j = (i * bound) & MASK_64
        k = j & 0xFFFFFFFF
        if k < bound:
            l = (-bound) % bound
            while k < l:
                i = xNext64(state) & 0xFFFFFFFF
                j = (i * bound) & MASK_64
                k = j & 0xFFFFFFFF
        return (j >> 32) & 0xFFFFFFFF

def leather_sequence(seed, entity, num_kills):
    if entity == "horse":
        md5_0, md5_1 = HORSE_MD5_0, HORSE_MD5_1
    elif entity == "donkey":
        md5_0, md5_1 = DONKEY_MD5_0, DONKEY_MD5_1
    elif entity == "llama":
        md5_0, md5_1 = LLAMA_MD5_0, LLAMA_MD5_1
    elif entity == "mule":
        md5_0, md5_1 = MULE_MD5_0, MULE_MD5_1
    elif entity == "trader_llama":
        md5_0, md5_1 = TRADER_LLAMA_MD5_0, TRADER_LLAMA_MD5_1
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
    results = []

    for _ in range(num_kills):
        count = java_next_int(state, 3)
        results.append(count)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horse looking leather drops")
    parser.add_argument("seed", type=int)
    parser.add_argument("-n", type=str, required=True, choices=["horse", "donkey", "llama", "mule", "trader_llama"])
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    seq = leather_sequence(args.seed, args.n, args.d)
    print(f"Seed {args.seed} {args.n} leather drops: {seq}")
