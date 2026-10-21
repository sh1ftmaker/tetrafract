# Undermine: Fort War

A single-player build of the Fort War level from [WorldParty](https://github.com/zalo/WorldParty) (the `fortwar` branch) by Johnathon Selstad. The game is used under the MIT License, see [`LICENSE-WorldParty`](LICENSE-WorldParty).

In this version, the game's mesh-cutting destruction is replaced with Tetrafract's GPU tetrahedral simulation:
- Untouched terrain stays static voxels.
- Damaged areas become tets.
- Calm areas go to sleep.
- Small debris turns into dust.

Networking, teams and scoring are not included.

Textures are CC0 from Poly Haven (see `assets/textures/README.md`).

**Play:** https://sh1ftmaker.github.io/tetrafract/fortwar/
