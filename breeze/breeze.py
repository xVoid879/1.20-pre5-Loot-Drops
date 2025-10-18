import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
BREEZE_MD5_0 = 0x89ecfb4dca170763
BREEZE_MD5_1 = 0xb8cbc5ab6d52c6da
STAFFORD_MIXING_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIXING_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def mix_stafford13(seed: int) -> int:
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIXING_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIXING_2 & MASK_64
    seed ^= (seed >> 31)
    return seed & MASK_64


def get_random_sequence_xoro(world_seed: int, md5_0: int, md5_1: int):
    unmixed_lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    unmixed_hi = (unmixed_lo - SUBTRACT_CONSTANT) & MASK_64

    split_lo = unmixed_lo ^ md5_0
    split_hi = unmixed_hi ^ md5_1

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


def next_int(state, bound: int) -> int:
    while True:
        bits = xoro_next(state) >> 31
        val = bits % bound
        if bits - val + (bound - 1) >= 0:
            return val


def next_breeze_rod_count(state) -> int:
    return 1 + next_int(state, 2)


def breeze_sequence(world_seed: int, total_elements: int = MAX_SEQUENCE):
    state = get_random_sequence_xoro(world_seed, BREEZE_MD5_0, BREEZE_MD5_1)
    return [next_breeze_rod_count(state) for _ in range(total_elements)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breeze Rod Drop")
    parser.add_argument("seed", type=int)
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE)
    args = parser.parse_args()

    seq = breeze_sequence(args.seed, args.d)
    print(f"Seed {args.seed} breeze rod drops: {seq}")
