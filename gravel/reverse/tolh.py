import argparse

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
GRAVEL_MD5_0 = 0x2fedfb509401412f
GRAVEL_MD5_1 = 0x6b4882392a3638a0
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb

def stafford_mix(x):
    x = ((x ^ (x >> 30)) * STAFFORD_MIX_1) & MASK_64
    x = ((x ^ (x >> 27)) * STAFFORD_MIX_2) & MASK_64
    x = x ^ (x >> 31)
    return x & MASK_64

def state(seed):
    l = (seed ^ SILVER_RATIO_64) & MASK_64
    h = (l + GOLDEN_RATIO_64) & MASK_64

    l ^= GRAVEL_MD5_0
    h ^= GRAVEL_MD5_1

    l = stafford_mix(l)
    h = stafford_mix(h)

    return l, h

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate l and h gravel state")
    parser.add_argument("seed", type=int, help="World seed")
    args = parser.parse_args()

    l, h = state(args.seed)
    print(f"Seed: {args.seed}")
    print(f"l = 0x{l:016x}")
    print(f"h = 0x{h:016x}")