import sys

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
GLOWSTONE_MD5_0 = 0x83196c8056477ab2
GLOWSTONE_MD5_1 = 0x41f5812fc482f9ac
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

def mix_stafford(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIX_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIX_2 & MASK_64
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

def get_state(world_seed):
    lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    hi = (lo - SUBTRACT_CONSTANT) & MASK_64
    l = mix_stafford(lo ^ GLOWSTONE_MD5_0)
    h = mix_stafford(hi ^ GLOWSTONE_MD5_1)
    return [l, h]

def next_int(state, bound):
    while True:
        bits = xoro_next(state) & 0xFFFFFFFF
        val = (bits * bound) >> 32
        if val < bound:
            return val

def simulate(seed, n=MAX_SEQUENCE):
    state = get_state(seed)
    drops = []
    for _ in range(n):
        drop_count = next_int(state, 3) + 2
        drops.append(drop_count)
        xoro_next(state)
    return drops

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    drops = simulate(seed, MAX_SEQUENCE)
    print(f"Seed {seed} glowstone dust sequence:", drops)
