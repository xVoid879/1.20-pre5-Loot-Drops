import sys
from z3 import *

MASK_64 = 0xFFFFFFFFFFFFFFFF

GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15
SILVER_RATIO_64 = 0x6a09e667f3bcc909
GRAVEL_MD5_0 = 0x2fedfb509401412f
GRAVEL_MD5_1 = 0x6b4882392a3638a0
STAFFORD_MIX_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIX_2 = 0x94d049bb133111eb

def parse_arg_int(s):
    s = s.strip()
    if s.startswith("0x") or s.startswith("0X"):
        return int(s, 16)
    return int(s, 10)

def to_signed(u64):
    if u64 >= (1 << 63):
        return u64 - (1 << 64)
    return u64

def main():
    if len(sys.argv) < 3:
        print("python reverse.py l h [solutions]")
        sys.exit(1)

    observed_l = parse_arg_int(sys.argv[1]) & MASK_64
    observed_h = parse_arg_int(sys.argv[2]) & MASK_64
    max_solutions = int(sys.argv[3]) if len(sys.argv) >= 4 else 5

    solver = Solver()

    l_pre = BitVec('l_pre', 64)
    h_pre = BitVec('h_pre', 64)

    def stafford_z3(x):
        x = ((x ^ LShR(x, 30)) * STAFFORD_MIX_1) & MASK_64
        x = ((x ^ LShR(x, 27)) * STAFFORD_MIX_2) & MASK_64
        x = x ^ LShR(x, 31)
        return x

    solver.add(stafford_z3(l_pre) == observed_l)
    solver.add(stafford_z3(h_pre) == observed_h)

    seed = BitVec('seed', 64)
    l_bv = (seed ^ BitVecVal(SILVER_RATIO_64, 64)) ^ BitVecVal(MD5_0, 64)
    h_bv = ((seed ^ BitVecVal(SILVER_RATIO_64, 64)) + BitVecVal(GOLDEN_RATIO_64, 64)) & BitVecVal(MASK_64, 64)
    h_bv = h_bv ^ BitVecVal(MD5_1, 64)

    solver.add(l_pre == l_bv)
    solver.add(h_pre == h_bv)

    found = 0
    while found < max_solutions:
        check = solver.check()
        if check == sat:
            m = solver.model()
            seed_val_unsigned = m[seed].as_long() & MASK_64
            seed_val_signed = to_signed(seed_val_unsigned)
            print(f"Found seed #{found+1}: {seed_val_signed} (0x{seed_val_unsigned:016x})")
            solver.add(seed != BitVecVal(seed_val_unsigned, 64))
            found += 1
        elif check == unknown:
            print("No Solutions")
            break
        else:
            print("No or more solutions.")
            break

    if found == 0:
        print("No seeds found")

if __name__ == "__main__":
    main()
