import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
ENDERMAN_MD5_0 = 0xa6dd84b6981b3414
ENDERMAN_MD5_1 = 0x4bec48c8de2830b3
STAFFORD_MIXING_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIXING_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def mix_stafford13(seed: int) -> int:
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIXING_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIXING_2 & MASK_64
    seed ^= (seed >> 31)
    return seed & MASK_64

def rotl(x: int, k: int) -> int:
    return ((x << k) & MASK_64) | (x >> (64 - k))

def get_xoro_state(world_seed: int):
    unmixed_lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    unmixed_hi = (unmixed_lo - SUBTRACT_CONSTANT) & MASK_64
    lo = mix_stafford13(unmixed_lo ^ ENDERMAN_MD5_0)
    hi = mix_stafford13(unmixed_hi ^ ENDERMAN_MD5_1)
    return [lo, hi]

def xoro_next(state):
    s0, s1 = state
    result = (rotl((s0 + s1) & MASK_64, 17) + s0) & MASK_64
    s1 ^= s0
    state[0] = (rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK_64)) & MASK_64
    state[1] = rotl(s1, 28) & MASK_64
    return result

def next_int(state, min_val, max_val):
    bound = max_val - min_val + 1
    while True:
        bits = xoro_next(state) >> 31
        val = bits % bound
        if bits - val + (bound - 1) >= 0:
            return val + min_val

def next_ender_pearl(state):
    return next_int(state, 0, 1)

def simulate_ender(world_seed: int, dts: int = MAX_SEQUENCE): # dts is short for drops to sim
    state = get_xoro_state(world_seed)
    sequence = []
    kills_since_last_pearl = 0

    while len(sequence) < dts:
        kills_since_last_pearl += 1
        if next_ender_pearl(state):
            sequence.append(kills_since_last_pearl)
            kills_since_last_pearl = 0

    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ender pearl simulation")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    sequence = simulate_ender(args.seed, args.d)

    print(f"Seed {args.seed} ender pearl drops: {sequence}")
