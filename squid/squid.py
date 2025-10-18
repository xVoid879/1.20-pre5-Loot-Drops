import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
SQUID_MD5_0 = 0xd1debecea9b76c48
SQUID_MD5_1 = 0xa17a5280a0ad83be
GLOW_SQUID_MD5_0 = 0xe5e13ed3665c1396
GLOW_SQUID_MD5_1 = 0xb0576dbfaa16cf2e
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

def squid_sequence(seed, squid_type="normal", max_seq=MAX_SEQUENCE):
    if squid_type == "normal":
        md5_lo, md5_hi = SQUID_MD5_0, SQUID_MD5_1
    elif squid_type == "glow":
        md5_lo, md5_hi = GLOW_SQUID_MD5_0, GLOW_SQUID_MD5_1
    else:
        raise ValueError(f"Wrong type: {squid_type}. Must be normal or glow.")
    
    l = (seed ^ SILVER_RATIO_64) & MASK_64
    h = (l + GOLDEN_RATIO_64) & MASK_64
    l ^= md5_lo
    h ^= md5_hi
    l = ((l ^ (l >> 30)) * STAFFORD_MIX_1) & MASK_64
    h = ((h ^ (h >> 30)) * STAFFORD_MIX_1) & MASK_64
    l = ((l ^ (l >> 27)) * STAFFORD_MIX_2) & MASK_64
    h = ((h ^ (h >> 27)) * STAFFORD_MIX_2) & MASK_64
    l ^= (l >> 31)
    h ^= (h >> 31)

    state = [l, h]
    sequence = []

    for _ in range(max_seq):
        drops = java_next_int(state, 3) + 1
        sequence.append(drops)

    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Squid and Glow Squid Ink Sacs")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    parser.add_argument("-n", type=str, default="normal",
                        choices=["normal", "glow"])
    
    args = parser.parse_args()

    seq = squid_sequence(args.seed, squid_type=args.n, max_seq=args.d)
    print(f"Seed {args.seed} {args.n} squid ink sac sequence: {seq}")
