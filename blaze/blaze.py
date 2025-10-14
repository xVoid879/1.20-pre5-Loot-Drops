import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF
MAX_SEQUENCE = 20  # max and default sequence

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
BLAZE_MD5_0 = 0xa9ec152f9c889472
BLAZE_MD5_1 = 0xcb9b0580c2b91a9e
STAFFORD_MIXING_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIXING_2 = 0x94d049bb133111eb


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


def next_bits(state, bits: int) -> int:
    return xoro_next(state) >> (64 - bits)


def next_int(state, bound: int) -> int:
    while True:
        bits = xoro_next(state) >> 31
        val = bits % bound
        if bits - val + (bound - 1) >= 0:
            return val


def next_blaze_rod(state) -> bool:
    set_count = next_int(state, 2)
    return set_count == 1


def simulate(world_seed: int, total_elements: int = MAX_SEQUENCE):
    state = get_random_sequence_xoro(world_seed, BLAZE_MD5_0, BLAZE_MD5_1)
    
    rod_counts = []
    counter = 0

    while len(rod_counts) < total_elements:
        counter += 1
        if next_blaze_rod(state):
            rod_counts.append(counter)
            counter = 0 

    if len(rod_counts) > total_elements:
        rod_counts = rod_counts[:total_elements]

    return rod_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blaze Rod Drop Sequence")
    parser.add_argument("seed", type=int, help="World seed")
    parser.add_argument("-d", type=int, default=MAX_SEQUENCE, help="# to output")
    args = parser.parse_args()

    seq = simulate(args.seed, args.d)
    print(f"Seed {args.seed} blaze rod drop sequence: {seq}")