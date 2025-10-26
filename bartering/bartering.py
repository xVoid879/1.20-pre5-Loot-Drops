# Initial by DuncanRuns https://github.com/DuncanRuns/MiLTSU/blob/main/src/main/java/me/duncanruns/miltsu/randomsequence/BarteringRandomSequence.java

import sys

MASK_64 = 0xFFFFFFFFFFFFFFFF

SILVER_RATIO_64 = 0x6a09e667f3bcc909
SUBTRACT_CONSTANT = 0x61C8864680B583EB
BARTERING_MD5_0 = 0xf79b444cdb83b923
BARTERING_MD5_1 = 0xe09fad0dcb68166a
STAFFORD_MIXING_1 = 0xbf58476d1ce4e5b9
STAFFORD_MIXING_2 = 0x94d049bb133111eb
MAX_SEQUENCE = 20

Items = [ # loot table
    {"name":"book","min":1,"max":1,"weight":5,"roll_amount":False,"roll_ss":True},
    {"name":"iron_boots","min":1,"max":1,"weight":8,"roll_amount":False,"roll_ss":True},
    {"name":"potion","min":1,"max":1,"weight":8,"roll_amount":False,"roll_ss":False},
    {"name":"splash_potion","min":1,"max":1,"weight":8,"roll_amount":False,"roll_ss":False},
    {"name":"potion","min":1,"max":1,"weight":10,"roll_amount":False,"roll_ss":False},
    {"name":"iron_nugget","min":10,"max":36,"weight":10,"roll_amount":True,"roll_ss":False},
    {"name":"ender_pearl","min":2,"max":4,"weight":10,"roll_amount":True,"roll_ss":False},
    {"name":"string","min":3,"max":9,"weight":20,"roll_amount":True,"roll_ss":False},
    {"name":"quartz","min":5,"max":12,"weight":20,"roll_amount":True,"roll_ss":False},
    {"name":"obsidian","min":1,"max":1,"weight":40,"roll_amount":False,"roll_ss":False},
    {"name":"crying_obsidian","min":1,"max":3,"weight":40,"roll_amount":True,"roll_ss":False},
    {"name":"fire_charge","min":1,"max":1,"weight":40,"roll_amount":False,"roll_ss":False},
    {"name":"leather","min":2,"max":4,"weight":40,"roll_amount":True,"roll_ss":False},
    {"name":"soul_sand","min":2,"max":8,"weight":40,"roll_amount":True,"roll_ss":False},
    {"name":"nether_brick","min":2,"max":8,"weight":40,"roll_amount":True,"roll_ss":False},
    {"name":"spectral_arrow","min":6,"max":12,"weight":40,"roll_amount":True,"roll_ss":False},
    {"name":"gravel","min":8,"max":16,"weight":40,"roll_amount":True,"roll_ss":False},
    {"name":"blackstone","min":8,"max":16,"weight":40,"roll_amount":True,"roll_ss":False},
]

Total_Weight = 459 # 5 + 8 + 8 + 8 + 10 + 10 + 10 + 20 + 20 + 40 + 40 + 40 + 40 + 40 + 40 + 40 + 40 + 40

def mix_stafford(seed):
    seed = (seed ^ (seed >> 30)) * STAFFORD_MIXING_1 & MASK_64
    seed = (seed ^ (seed >> 27)) * STAFFORD_MIXING_2 & MASK_64
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
    unmixed_lo = (world_seed ^ SILVER_RATIO_64) & MASK_64
    unmixed_hi = (unmixed_lo - SUBTRACT_CONSTANT) & MASK_64
    l = mix_stafford(unmixed_lo ^ BARTERING_MD5_0)
    h = mix_stafford(unmixed_hi ^ BARTERING_MD5_1)
    return [l, h]

def next_int(state, bound):
    if bound <= 0:
        raise ValueError("bound must be positive")
    while True:
        bits = xoro_next(state) & 0xFFFFFFFF
        val = (bits * bound) >> 32
        if val < bound:
            return val

def roll(state):
    j = next_int(state, Total_Weight)
    for e in Items:
        if (j := j - e["weight"]) < 0:
            amount = 1
            if e["roll_amount"]:
                amount = e["min"] + next_int(state, e["max"] - e["min"] + 1)
            if e["roll_ss"]:
                next_int(state, 1)
                amount = next_int(state, 3) + 1
            return (e["name"], amount)
    return None

def simulate(world_seed, n=MAX_SEQUENCE):
    state = get_state(world_seed)
    trades = []
    for _ in range(n):
        trades.append(roll(state))
    return trades

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seq = simulate(seed, MAX_SEQUENCE)
    for i, (name, count) in enumerate(seq, 1):
        print(f"Trade {i}: {name} x{count}")
