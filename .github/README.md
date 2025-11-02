<div align="center">
  <h1 style="margin-top: 0">1.20-pre5+ Loot Drops</h1>
  <span>Research/Simulation of Minecraft Loot Drops (1.20-pre5+)</span>
</div>

---

## What is this?

This repository contains simulations of **Minecraft 1.20-pre5+ loot drops**. This is what Minecraft still uses at the time of writing this (Will edit this README if they change).

Some CUDA kernels are included for finding seeds with cool loot drops. No records are stored in this repository.

The main goal is to simulate as many/all the loot drops in the game. (And possibly create a website for easier use when done with all)

---

## Before You Start...

You should watch this video by Matthew Bolan, which covers 1.20-pre1 through pre4 loot drop logic. This is also a nice introduction to loot drops:

<a href="https://www.youtube.com/watch?v=a5-dISWtkDs"><img src="https://img.shields.io/badge/Watch-Matthew%20Bolan-red?logo=youtube"></a>

You can should also see:
- **Block MD5 Values** (leads to entity and gameplay values as well):  
  [gist.github.com/xVoid879/996dad365721789e53855c59fcf2fd99](https://gist.github.com/xVoid879/996dad365721789e53855c59fcf2fd99)
- **Simplified Gravel Drop Explanation (1.20-pre5+)**:  
  [gist.github.com/xVoid879/8c5665c2c1ef3926a3e66cf4f052c888](https://gist.github.com/xVoid879/8c5665c2c1ef3926a3e66cf4f052c888)

---

## Changes in 1.20-pre5

In **Minecraft 1.20-pre5**, Mojang made some changes to the loot drop logic:

| Change | What it did |
| :------ | :----------- |
| **Stafford Mixing after Xoroshiro + MD5** | Stafford mixing happens after Xoroshiro and MD5 hashing. |
| **Burn Call Removed** | The burn call (`nextLong()`) that was added in 1.20-pre3 was removed. |
| **Drop Correlation Fixed** | Due to Stafford mixing happening after other hashing steps, the unintended correlation between loot drops was fixed. (e.g., if you got 4 melons on the 7th mine, you were guaranteed to get a blaze rod on the 13th kill). |

---


## Credits

*(In no particular order)*

- [**Tremeschin**](https://github.com/tremeschin) — Explored dead bush stick logic and created a python script to obtain two MD5 values from a specific resource location. (Statistical Research on finding cool loot drops seeds)
- [**Matthew Bolan**](https://github.com/mjtb49) — Researched 1.20-pre1 to 1.20-pre4 loot drop logic, creating a video on it, and [providing melon 1.20-pre4 loot drop logic](https://gist.github.com/mjtb49/f3e01e3355178d2bb6c814606971c374).
- [**Xpple**](https://github.com/xpple) — Provided the 1.20-pre5+ gravel loot drop logic, which, in my opinion, started this.
- [**Fragrant Result**](https://github.com/FragrantResult186) — For providing 1.20-pre5+ tree leaf loot drop logic.
- [**PseudoGravity**](https://github.com/pseudogravity) — Finding a solution to a dumb mistake I made in my CUDA kernel searching for back-to-back flints. Also, for exploring a potential optimization for seedfinding lootdrops, involving seeds with similar xoroshiro initializations. (Statistical Research on finding cool loot drop seeds)
- [**DuncanRuns**](https://github.com/DuncanRuns) — Created a [Java library](https://github.com/duncanruns/miltsu), simulating bartering, blazes, and wither skeleton 1.20-pre5+ loot drops.
- [**InventivetalentDev**](https://github.com/InventivetalentDev) — Created [mcasset.cloud ](https://mcasset.cloud) where I got the loot tables more easily and used it to create a script to obtain all MD5 values.

*If I missed anyone, please tell me*

---

If you have questions, you can either join the Minecraft@Home Discord or DM me on Discord
- Discord: **[Minecraft@Home](https://discord.gg/mch)**
- DM: **xvoid879**

---
