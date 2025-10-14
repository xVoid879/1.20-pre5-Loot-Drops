# 1.20-pre5+ Loot Drops 
Research and simulation of Minecraft 1.20-pre5+ loot drops, with some having CUDA kernels for discovering seeds with cool loot drops (No records in this repo).

The main goal is to just simulate all the loot drops.

## Before you start...

You should definitely watch this video by Matthew Bolan: https://www.youtube.com/watch?v=a5-dISWtkDs

Block's MD5 Values (Also leads to Entities and Gameplay) here: https://gist.github.com/xVoid879/996dad365721789e53855c59fcf2fd99

Simplified Explanation on how gravel drops work, talking a bit about how the seed is hashed in 1.20-pre5: https://gist.github.com/xVoid879/8c5665c2c1ef3926a3e66cf4f052c888

## Credits (In no particular order) for the research:
- [Tremeschin](https://github.com/tremeschin/) for exploring dead bush stick logic and how to obtain the two MD5 Values. (Statistical Research)
- [Matthew Bolan](https://github.com/mjtb49/) on creating a video about loot drops and exploring the previous version's loot drops. Also [providing](https://gist.github.com/mjtb49/f3e01e3355178d2bb6c814606971c374) melon logic for 1.20-pre4.
- [Xpple](https://github.com/xpple/) for providing gravel drop logic, which in my opinion, started this.
- [Fragrant Result](https://github.com/FragrantResult186) for providing tree logic.
- [PseudoGravity](https://github.com/pseudogravity) for finding a solution to a dumb mistake I made in my CUDA kernel searching for back-to-back flints. Also for exploring a potential optimization involving seeds with similar xoroshiro initializations. (Statistical Research)
- [DuncanRuns](https://github.com/DuncanRuns) for creating a [library](https://github.com/duncanruns/miltsu) in Java simulating many things, including blaze and wither loot drops.
- Please tell me if I missed anyone
