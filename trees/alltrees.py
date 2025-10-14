# HUGE THANKS to Fragrant Result (https://github.com/FragrantResult186)

import sys

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6A09E667F3BCC909
SUBTRACT_CONSTANT = 0x61C8864680B583EB

# from https://gist.github.com/xVoid879/996dad365721789e53855c59fcf2fd99
ACACIA_LEAVES_MD5_0 = 0xff546e958442acf1
ACACIA_LEAVES_MD5_1 = 0xf95e1c4b073ec699
AZALEA_LEAVES_MD5_0 = 0xf019377ea52c964e
AZALEA_LEAVES_MD5_1 = 0x641a1b8f5889c4ed
BIRCH_LEAVES_MD5_0 = 0xf28deff6960b5a60
BIRCH_LEAVES_MD5_1 = 0x894e85666ba780ed
CHERRY_LEAVES_MD5_0 = 0xb502b9fae0076878
CHERRY_LEAVES_MD5_1 = 0xdcd0fa37558121c6
DARK_OAK_LEAVES_MD5_0 = 0x591e4bdfa0e1abc9
DARK_OAK_LEAVES_MD5_1 = 0x977be5db1d8ffbc1
FLOWERING_AZALEA_LEAVES_MD5_0 = 0x3a64023d256a6749
FLOWERING_AZALEA_LEAVES_MD5_1 = 0xe3984839c1bab885
JUNGLE_LEAVES_MD5_0 = 0x298d58c3634bed3c
JUNGLE_LEAVES_MD5_1 = 0x5fd281aa02a06aa4
MANGROVE_LEAVES_MD5_0 = 0x9888b2e0ebbe7d53
MANGROVE_LEAVES_MD5_1 = 0x55df737fab70fe71
OAK_LEAVES_MD5_0 = 0xef6489bec2529e35
OAK_LEAVES_MD5_1 = 0x1f1ab2c703aa2b5d
PALE_OAK_LEAVES_MD5_0 = 0x49c7de25b6924070
PALE_OAK_LEAVES_MD5_1 = 0xde4af3b3a2c3f957
SPRUCE_LEAVES_MD5_0 = 0x77905201c4056a90
SPRUCE_LEAVES_MD5_1 = 0x4d25b9140095c5b5

STAFFORD_MIXING_1 = 0xBF58476D1CE4E5B9
STAFFORD_MIXING_2 = 0x94D049BB133111EB
MAX_SEQUENCE = 20  # max and default sequence to print

class Item:
    def __init__(self, name, count):
        self.name = name
        self.count = count
    def __eq__(self, other):
        return isinstance(other, Item) and self.name == other.name and self.count == other.count
    def __hash__(self):
        return hash((self.name, self.count))
    def __repr__(self):
        return f"{self.name}:{self.count}"

class Drop:
    def __init__(self, item_func, chances):
        self.item_func = item_func
        self.chances = chances
    def create_item(self):
        return self.item_func()

class Xoroshiro128Plus:
    def __init__(self, seed0, seed1):
        self.seed0 = seed0 & MASK_64
        self.seed1 = seed1 & MASK_64

    def next_long(self):
        s0 = self.seed0
        s1 = self.seed1
        result = (s0 + s1) & MASK_64
        result = ((result << 17) | (result >> 47)) & MASK_64
        result = (result + s0) & MASK_64

        s1 ^= s0
        self.seed0 = ((s0 << 49) | (s0 >> 15)) & MASK_64
        self.seed0 ^= s1
        self.seed0 ^= ((s1 << 21) & MASK_64)
        self.seed0 &= MASK_64
        self.seed1 = ((s1 << 28) | (s1 >> 36)) & MASK_64

        return result

    def next_float(self):
        return ((self.next_long() >> 40) & 0xFFFFFF) / 16777216.0

    def next_int(self, min_val, max_val):
        val = (self.next_long() >> 33) & 0x7FFFFFFF
        return min_val + (val * (max_val - min_val) // 2147483648)

class RandomSequence:
    def __init__(self, world_seed, md5_lo, md5_hi):
        lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
        hi = (lo - SUBTRACT_CONSTANT) & MASK_64
        mixed_lo = self.mix_stafford13(lo ^ md5_lo)
        mixed_hi = self.mix_stafford13(hi ^ md5_hi)
        self.random = Xoroshiro128Plus(mixed_lo, mixed_hi)

    @staticmethod
    def mix_stafford13(seed):
        seed ^= (seed >> 30) & MASK_64
        seed = (seed * STAFFORD_MIXING_1) & MASK_64
        seed ^= (seed >> 27) & MASK_64
        seed = (seed * STAFFORD_MIXING_2) & MASK_64
        seed ^= (seed >> 31) & MASK_64
        return seed & MASK_64

    def roll_chance(self, chances, level):
        level = max(0, min(level, len(chances) - 1))
        return self.random.next_float() < chances[level]

    def set_count(self, min_val, max_val):
        return self.random.next_int(min_val, max_val + 1)

class BlockRandomSequence(RandomSequence):
    def get_loot_table(self):
        raise NotImplementedError
    def next_drops(self):
        drops = []
        fortune = 0
        for drop in self.get_loot_table():
            if self.roll_chance(drop.chances, fortune):
                drops.append(drop.create_item())
        return drops

class AcaciaLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, ACACIA_LEAVES_MD5_0, ACACIA_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:acacia_sapling",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
        ]
    def get_loot_table(self): return self.loot_table

class AzaleaLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, AZALEA_LEAVES_MD5_0, AZALEA_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:azalea_itself",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
        ]
    def get_loot_table(self): return self.loot_table

class BirchLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, BIRCH_LEAVES_MD5_0, BIRCH_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:birch_sapling",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
        ]
    def get_loot_table(self): return self.loot_table

class CherryLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, CHERRY_LEAVES_MD5_0, CHERRY_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:cherry_sapling",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
        ]
    def get_loot_table(self): return self.loot_table

class DarkOakLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, DARK_OAK_LEAVES_MD5_0, DARK_OAK_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:dark_oak_sapling",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
            Drop(lambda: Item("minecraft:dark_oak_apple",1), [0.005,0.0055555557,0.00625,0.008333334,0.025])
        ]
    def get_loot_table(self): return self.loot_table

class FloweringAzaleaLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, FLOWERING_AZALEA_LEAVES_MD5_0, FLOWERING_AZALEA_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:flowering_azalea",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
        ]
    def get_loot_table(self): return self.loot_table

class JungleLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, JUNGLE_LEAVES_MD5_0, JUNGLE_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:jungle_sapling",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
        ]
    def get_loot_table(self): return self.loot_table

class MangroveLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, MANGROVE_LEAVES_MD5_0, MANGROVE_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
        ]
    def get_loot_table(self): return self.loot_table

class OakLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, OAK_LEAVES_MD5_0, OAK_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:oak_sapling",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick", self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
            Drop(lambda: Item("minecraft:oak_apple",1), [0.005,0.0055555557,0.00625,0.008333334,0.025])
        ]
    def get_loot_table(self): return self.loot_table

class PaleOakLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, PALE_OAK_LEAVES_MD5_0, PALE_OAK_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:pale_oak_sapling",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
        ]
    def get_loot_table(self): return self.loot_table

class SpruceLeaves(BlockRandomSequence):
    def __init__(self, world_seed):
        super().__init__(world_seed, SPRUCE_LEAVES_MD5_0, SPRUCE_LEAVES_MD5_1)
        self.loot_table = self.set_loot_table()
    def set_loot_table(self):
        return [
            Drop(lambda: Item("minecraft:spruce_sapling",1), [0.05,0.0625,0.083333336,0.1]),
            Drop(lambda: Item("minecraft:stick",self.set_count(1,2)), [0.02,0.022222223,0.025,0.033333335,0.1]),
        ]
    def get_loot_table(self): return self.loot_table

LEAF_CLASSES = {
    "oak": OakLeaves,
    "acacia": AcaciaLeaves,
    "azalea": AzaleaLeaves,
    "flowering": FloweringAzaleaLeaves,
    "birch": BirchLeaves,
    "cherry": CherryLeaves,
    "dark": DarkOakLeaves,
    "jungle": JungleLeaves,
    "mangrove": MangroveLeaves,
    "pale": PaleOakLeaves,
    "spruce": SpruceLeaves
}

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python alltrees.py <seed> [-d <depth>] -n <leaf>_<item>")
        sys.exit(1)

    world_seed = int(sys.argv[1])
    desired = MAX_SEQUENCE
    target_name = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i].lower() == "-d" and i + 1 < len(sys.argv):
            try:
                desired = min(MAX_SEQUENCE, int(sys.argv[i + 1]))
                i += 2
            except ValueError:
                i += 1
        elif sys.argv[i].lower() == "-n" and i + 1 < len(sys.argv):
            target_name = sys.argv[i + 1].lower()
            i += 2
        else:
            i += 1

    if not target_name or "_" not in target_name:
        print("Invalid -n <leaf>_<item>")
        sys.exit(1)

    leaf_name, item_name = target_name.split("_", 1)
    if leaf_name not in LEAF_CLASSES:
        print(f"Unknown leaf type: {leaf_name}")
        sys.exit(1)

    LeafClass = LEAF_CLASSES[leaf_name]
    leaf_block = LeafClass(world_seed)

    if item_name == "stick":
        target = "stick"
    else:
        target = Item(f"minecraft:{leaf_name}_{item_name}", 1)

    dfound = 0
    since_last = 0
    sequence = []

    while dfound < desired:
        drops = leaf_block.next_drops()
        found = False
        for drop in drops:
            if isinstance(target, Item):
                if drop == target:
                    found = True
                    break
            elif target == "stick" and drop.name == "minecraft:stick":
                found = True
                break
        if found:
            sequence.append(since_last + 1)
            since_last = 0
            dfound += 1
        else:
            since_last += 1

    print(f"Seed {world_seed} {target_name} sequence: {sequence}")