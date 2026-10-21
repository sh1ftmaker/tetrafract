# Tetrafract

Real-time destructible materials in the browser: wood that bends and splinters along the grain, glass that shatters, stone that crumbles into dust, and steel that bends and stays bent. All of the physics runs on the GPU in WebGPU compute shaders.

**Live demo:** https://sh1ftmaker.github.io/tetrafract/

It needs a WebGPU browser: desktop Chrome or Edge 113+, or Safari 18+.

## How it works

Every object is a tetrahedral mesh built from voxels. Each tet is shape-matched to its rest shape, and overlapping 4×4×4-cell clusters (lattice shape matching) add bending stiffness. A Chebyshev-accelerated Jacobi solver runs everything, entirely in three.js TSL compute.

Cracks run along tet faces. When a face is overstressed, the GPU splits the mesh there by duplicating vertices with a union-find pass, so the crack opens in the same substep. Broken faces spawn dust, splinters and glass slivers, and single loose tets retire into particles once they come to rest.

**[Undermine: Fort War](fortwar/)** ([play](https://sh1ftmaker.github.io/tetrafract/fortwar/)) applies the same simulation to a whole destructible game level from [WorldParty](https://github.com/zalo/WorldParty) (MIT). Intact terrain stays static, and only damaged areas are simulated.

An earlier experiment with rigid pieces joined by breakable XPBD welds is kept in [`archive/v2-rigid-pieces/`](archive/v2-rigid-pieces/) ([live](https://sh1ftmaker.github.io/tetrafract/archive/v2-rigid-pieces/)).

## Controls

Drag to orbit, scroll or pinch to zoom, and click or tap to fire. The panel picks the scene, tool, toughness, projectile settings and quality.

## Running locally

It is static files only, and three.js loads from a CDN:

```sh
python3 -m http.server 8173
# open http://localhost:8173/
```

## Credits

- Per-tet rotation extraction and Jacobi vertex averaging follow [Tetrament](https://github.com/zalo/Tetrament) by Johnathon Selstad (MIT). Rotation extraction itself is from Müller et al., "A Robust Method to Extract the Rotational Part of Deformations" (2016).
- Lattice shape matching: Rivers & James, "FastLSM: Fast Lattice Shape Matching for Robust Real-Time Deformation" (2007).
- Chebyshev acceleration: Wang, "A Chebyshev Semi-Iterative Approach for Accelerating Projective and Position-based Dynamics" (2015).
- XPBD rigid bodies (archived v2): Müller et al., "Detailed Rigid Body Simulation with Extended Position Based Dynamics" (2020).
- Built with [three.js](https://threejs.org/).
