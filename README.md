# 1.20-pre5+ Loot Drops
Research, Simulation on 1.20-pre5+ Loot Drops. Also contains CUDA Kernels for finding cool loot drops.

The goal is just to simulate all the loot drops and, optionally, find seeds with cool loot drops (Records won't be here).

You should definitely watch this video by Matthew Bolan: https://www.youtube.com/watch?v=a5-dISWtkDs

Block's MD5 Values (Also leads to Entities and Gameplay) here: https://gist.github.com/xVoid879/996dad365721789e53855c59fcf2fd99

Simplified Explanation on how gravel drops work: https://gist.github.com/xVoid879/8c5665c2c1ef3926a3e66cf4f052c888

**Credits (In no particular order) for the research:**
- [Tremeschin](https://github.com/tremeschin/) for exploring dead bush stick logic and how to obtain the two MD5 Values. (Statistical Research)
- [Matthew Bolan](https://github.com/mjtb49/) on creating a video about loot drops and exploring the previous version's loot drops. Also providing melon logic for 1.20-pre4.
- [Xpple](https://github.com/xpple/) for providing gravel drop logic, which in my opinion, started this.
- [Fragrant Result](https://github.com/FragrantResult186) for providing tree logic.
- [PseudoGravity](https://github.com/pseudogravity) for finding a solution to a dumb mistake I made in my CUDA kernel searching for back-to-back flints. Also for exploring a potential optimization involving seeds with similar xoroshiro initializations. (Statistical Research)
- [DuncanRuns](https://github.com/DuncanRuns) for creating a [library](https://github.com/duncanruns/miltsu) in Java simulating many things, including blaze and wither loot drops.
