import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF
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

def get_random_sequence_xoro(world_seed: int):
    unmixed_lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    unmixed_hi = (unmixed_lo - SUBTRACT_CONSTANT) & MASK_64
    lo = mix_stafford13(unmixed_lo ^ BLAZE_MD5_0)
    hi = mix_stafford13(unmixed_hi ^ BLAZE_MD5_1)
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

def next_float(state) -> float:
    return ((xoro_next(state) >> 40) & 0xFFFFFF) / float(1 << 24)

def next_base_drop(state) -> int:
    return (xoro_next(state) >> 31) & 1

def next_blaze_rod(state) -> int:
    base = next_base_drop(state)
    extra = round(next_float(state) * 3) # 3 looting
    return base + extra

def simulate_streak(world_seed: int) -> int:
    state = get_random_sequence_xoro(world_seed)
    streak = 0
    while True:
        if next_blaze_rod(state) == 4:
            streak += 1
        else:
            break
    return streak

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Back-to-Back 4's with Looting")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("-b", type=int, dest="back", required=True)
    args = parser.parse_args()

    for seed in range(args.start, args.end):
        streak_length = simulate_streak(seed)
        if streak_length >= args.back:
            print(f"Seed {seed} with {streak_length}")