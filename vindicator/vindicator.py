import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
VINDICATOR_MD5_0 = 0x225629b4261b6cff
VINDICATOR_MD5_1 = 0xb9106462b85adcd1
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def rotl64(x, r):
    return ((x << r) & MASK_64) | (x >> (64 - r))

def mix_stafford13(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
    return seed ^ (seed >> 31)

def get_xoro_state(world_seed: int):
    unmixed_lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    unmixed_hi = (unmixed_lo - SUBTRACT_CONSTANT) & MASK_64
    lo = mix_stafford13(unmixed_lo ^ VINDICATOR_MD5_0)
    hi = mix_stafford13(unmixed_hi ^ VINDICATOR_MD5_1)
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

def drop_emerald(state):
    return next_int(state, 2)

def simulate_vindicator(seed, max_drops):
    state = get_xoro_state(seed)
    seq = []
    for _ in range(max_drops):
        seq.append(drop_emerald(state))
    return seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vindicator emerald drop")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    seq = simulate_vindicator(args.seed, args.d)
    print(f"Seed {args.seed} vindicator emerald sequence: {seq}")
