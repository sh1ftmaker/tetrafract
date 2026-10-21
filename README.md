# Tetrafract

Real-time destructible materials in the browser: wood that bends and splinters along the grain, glass that shatters, stone that crumbles into dust, and steel that bends and stays bent. All of the physics runs on the GPU in WebGPU compute shaders.

**Live demo:** https://sh1ftmaker.github.io/tetrafract/

It needs a WebGPU browser: desktop Chrome or Edge 113+, or Safari 18+.

## Two approaches

- **v1 · Tetrahedra** (`index.html`, `src/`).
  - **Model:** every object is a tetrahedral mesh built from voxels. Each tet is shape-matched to its rest shape, and overlapping 4×4×4-cell clusters (lattice shape matching) add bending stiffness. A Chebyshev-accelerated Jacobi solver runs everything, entirely in three.js TSL compute.
  - **Fracture:** cracks run along tet faces. When a face is overstressed, the GPU splits the mesh there by duplicating vertices with a union-find pass, so the crack opens in the same substep.
  - **Debris:** broken faces spawn dust, splinters and glass slivers. Single loose tets retire into particles once they come to rest.
- **v2 · Rigid pieces** (`v2/`). Objects are pre-split into rigid pieces held together by breakable 6-DOF welds:
  - wood is split into fibres along the grain;
  - glass is split into shards, re-cut around the impact point;
  - stone is split into chunks.

  The pieces are solved with XPBD in raw WGSL, with probe-sphere and convex-hull contacts.

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
- XPBD rigid bodies (v2): Müller et al., "Detailed Rigid Body Simulation with Extended Position Based Dynamics" (2020).
- Built with [three.js](https://threejs.org/).
