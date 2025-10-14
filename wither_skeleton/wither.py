import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF
MAX_SEQUENCE = 20  # max sequence it outputs

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
SKULL_MD5_0 = 0x5a9fe82fe5b3dea9
SKULL_MD5_1 = 0x44ac0b3acb5ebf34
STAFFORD_MIXING_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIXING_2 = 0x94d049bb133111eb

def mix_stafford13(seed: int) -> int:
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIXING_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIXING_2 & MASK_64
    seed ^= (seed >> 31)
    return seed & MASK_64

def get_random_sequence_xoro(world_seed: int, SKULL_MD5_0: int, SKULL_MD5_1: int):
    unmixed_lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    unmixed_hi = (unmixed_lo - SUBTRACT_CONSTANT) & MASK_64

    split_lo = unmixed_lo ^ SKULL_MD5_0
    split_hi = unmixed_hi ^ SKULL_MD5_1

    lo = mix_stafford13(split_lo)
    hi = mix_stafford13(split_hi)

    return [lo, hi]

def rotl(x: int, k: int) -> int:
    return ((x << k) & MASK_64) | (x >> (64 - k))

def xoro_next(state):
    s0, s1 = state
    result = (rotl((s0 + s1) & MASK_64, 17) + s0) & MASK_64
    s1 ^= s0
    state[0] = (rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK_64)) & MASK_64
    state[1] = rotl(s1, 28) & MASK_64
    return result

def next_bits(state, bits: int) -> int:
    return xoro_next(state) >> (64 - bits)

def next_float(state) -> float:
    return next_bits(state, 24) / float(1 << 24)

def next_int(state) -> int:
    return xoro_next(state) & 0xFFFFFFFF  # low 32 bits

def next_has_skull(state) -> bool:
    for _ in range(2):
        next_int(state)
    return next_float(state) < 0.025

def simulate(world_seed: int, MAX_SEQUENCE: int):
    state = get_random_sequence_xoro(world_seed, SKULL_MD5_0, SKULL_MD5_1)
    sequence = []
    kills_since = 0
    while len(sequence) < MAX_SEQUENCE:
        kills_since += 1
        if next_has_skull(state):
            sequence.append(kills_since)
            kills_since = 0
    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wither Skeleton Skull Sequence")
    parser.add_argument("seed", type=int, help="World seed")
    args = parser.parse_args()

    seq = simulate(args.seed, MAX_SEQUENCE)
    print(f"Seed {args.seed} wither skeleton sequence {seq}")