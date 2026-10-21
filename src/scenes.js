// Demo scenes, loosely after classic destruction demo reels: a wooden trestle that
// collapses, a hut flattened by a boulder, glass shattered by a falling pole,
// and a lab of materials (wood, steel, concrete, jelly) under a wrecking ball.

import * as THREE from 'three/webgpu';
import { WorldBuilder, MAT } from './builder.js';

const X = 0, Y = 1, Z = 2;

function boulder(w, { center, radius, cell = 0.11, velocity = [0, 0, 0], spin = [0, 0, 0] }) {
  const n = Math.ceil((radius * 2) / cell);
  const min = center.map((c) => c - (n * cell) / 2);
  const bumps = [Math.random() * 6, Math.random() * 6, Math.random() * 6];
  const wobble = (p) => 0.12 * Math.sin(p[0] * 9 + bumps[0]) * Math.sin(p[1] * 7 + bumps[1]) * Math.sin(p[2] * 8 + bumps[2]);
  w.addVoxels({
    material: MAT.ROCK, min, dims: [n, n, n], cell,
    cellFn: (i, j, k) => {
      const p = [(i + 0.5) / n - 0.5, (j + 0.5) / n - 0.5, (k + 0.5) / n - 0.5];
      return Math.hypot(...p) * 2 < 1 + wobble(p) ? {} : null;
    },
    // pull the voxel staircase onto the lumpy rock surface (radially, most of the way)
    warp: (q, surface) => {
      if (!surface) return q;
      const d = [q[0] - center[0], q[1] - center[1], q[2] - center[2]];
      const l = Math.hypot(...d) || 1;
      const u = d.map((x) => x / l);
      const R = radius * (1 + wobble(u.map((x) => x * 0.5)));
      const t = 0.85 * (R - l);
      return [q[0] + u[0] * t, q[1] + u[1] * t, q[2] + u[2] * t];
    },
    velocity: (x, y, z) => {
      const r = [x - center[0], y - center[1], z - center[2]];
      return [
        velocity[0] + spin[1] * r[2] - spin[2] * r[1],
        velocity[1] + spin[2] * r[0] - spin[0] * r[2],
        velocity[2] + spin[0] * r[1] - spin[1] * r[0],
      ];
    },
  });
}

export const SCENES = {
  trestle: {
    label: 'Trestle bridge',
    camera: [7.5, 4.2, 7.5], target: [3.2, 1.3, 0.5],
    build(w) {
      const nx = 64, ny = 27, nz = 10, h = 0.1;
      const bents = [1, 16, 31, 46, 61];
      w.addVoxels({
        material: MAT.WOOD, min: [0, 0, 0], dims: [nx, ny, nz], cell: h, jointStrength: 0.3,
        cellFn: (i, j, k) => {
          for (let b = 0; b < bents.length; b++) {
            const x0 = bents[b];
            if (i < x0 || i > x0 + 1) continue;
            const part = b * 10;
            if (j <= 23 && (k <= 1 || k >= 8)) return { grain: Y, part: part + (k <= 1 ? 1 : 2) };
            if ((j === 22 || j === 23) || (j === 11 || j === 12)) return { grain: Z, part: part + (j > 15 ? 3 : 4) };
            // X brace
            const t = (k - 1) / 7, yb = 1 + t * 20, yb2 = 21 - t * 20;
            if (i === x0 && j < 22 && (Math.abs(j - yb) < 1.1 || Math.abs(j - yb2) < 1.1)) return { grain: Y, part: part + 5 };
          }
          if ((j === 24 || j === 25) && ((k >= 1 && k <= 2) || (k >= 7 && k <= 8))) return { grain: X, part: 100 + (k < 5 ? 0 : 1) };
          if (j === 26 && i % 3 === 0) return { grain: Z, part: 200 + i };
          return null;
        },
      });
    },
  },

  hut: {
    label: 'Hut & boulder',
    camera: [-4.5, 3.2, 6.5], target: [1.0, 1.1, 0],
    boulders: [{ center: [7.5, 1.6, 0.1], radius: 0.75, velocity: [-11, 2.5, 0] }],
    build(w) {
      const nx = 30, ny = 25, nz = 24, h = 0.1;
      w.addVoxels({
        material: MAT.WOOD, min: [-1.5, 0, -1.2], dims: [nx, ny, nz], cell: h, jointStrength: 0.4,
        cellFn: (i, j, k) => {
          const wallX = i === 0 || i === nx - 1, wallZ = k === 0 || k === nz - 1;
          const corner = (i <= 1 || i >= nx - 2) && (k <= 1 || k >= nz - 2);
          if (j <= 21 && corner) return { grain: Y, part: 1000 + (i > 5 ? 1 : 0) + (k > 5 ? 2 : 0) };
          if (j > 21 && j < 24) return (wallX || wallZ || i % 6 === 0) ? { grain: wallX ? Z : X, part: 900 + (wallX ? i : k) } : null;
          if (j === 24) return { grain: Z, part: 800 + (i >> 2) }; // roof boards
          if (j > 21) return null;
          // door in the -z wall, window in the +x wall
          if (k === 0 && i >= 11 && i <= 18 && j < 18) return null;
          if (i === nx - 1 && k >= 8 && k <= 15 && j >= 9 && j <= 15) return null;
          if (wallZ) return { grain: X, part: (k ? 100 : 0) + (j >> 1) };
          if (wallX) return { grain: Z, part: (i ? 300 : 200) + (j >> 1) };
          return null;
        },
      });
    },
  },

  glass: {
    label: 'Glass tank',
    camera: [3.6, 2.6, 4.4], target: [0, 1.1, 0],
    build(w) {
      const s = 0.8, H = 2.2, th = 0.05, h = 0.1;
      const n = Math.round((2 * s) / h), ny = Math.round(H / h);
      // four panes of a tank: two facing x, two facing z
      w.addVoxels({ material: MAT.GLASS, min: [-s, 0, -s - th], dims: [n, ny, 1], cell: [h, h, th] });
      w.addVoxels({ material: MAT.GLASS, min: [-s, 0, s], dims: [n, ny, 1], cell: [h, h, th] });
      w.addVoxels({ material: MAT.GLASS, min: [-s - th, 0, -s], dims: [1, ny, n], cell: [th, h, h] });
      w.addVoxels({ material: MAT.GLASS, min: [s, 0, -s], dims: [1, ny, n], cell: [th, h, h] });
      // a tall pole standing inside, already tipping toward the +x pane
      const ph = 3.2;
      w.addVoxels({
        material: MAT.WOOD, min: [-0.1, 0, -0.1], dims: [2, 16, 2], cell: [0.1, ph / 16, 0.1], grain: Y,
        velocity: (x, y) => [y * 0.9, 0, y * 0.25],
      });
      // free-standing panes to shoot at
      for (let i = 0; i < 3; i++) {
        w.addVoxels({ material: MAT.GLASS, min: [-2.2 + i * 1.6, 0, -2.6], dims: [12, 16, 1], cell: [0.1, 0.1, 0.04], pin: (x, y) => y < 0.01 });
      }
    },
  },

  lab: {
    label: 'Materials lab',
    camera: [7.5, 4.5, 6.5], target: [0, 1.0, 0],
    wreckingBall: { pivot: [0, 6.6, 0], length: 5.0, angle: -1.1, radius: 0.6 },
    // a load dropped onto the middle of each beam
    boulders: [{ center: [-0.05, 3.6, -3.6], radius: 0.4, velocity: [0, -4, 0] }, { center: [-0.05, 3.6, 3.2], radius: 0.4, velocity: [0, -4, 0] }],
    build(w) {
      // concrete wall anchored to its foundation, in the wrecking ball's path
      w.addVoxels({ material: MAT.STONE, min: [-0.15, 0, -1.5], dims: [3, 24, 30], cell: 0.1, pin: (x, y) => y < 0.01 });
      // wood beam and steel beam, each on a pair of stone pedestals
      for (const [z, mat] of [[-3.6, MAT.WOOD], [3.2, MAT.STEEL]]) {
        for (const x of [-2.2, 1.6]) w.addVoxels({ material: MAT.STONE, min: [x, 0, z - 0.3], dims: [6, 9, 6], cell: 0.1 });
        w.addVoxels({ material: mat, min: [-2.4, 0.9, z - 0.15], dims: [mat === MAT.STEEL ? 44 : 22, mat === MAT.STEEL ? 2 : 3, 3], cell: mat === MAT.STEEL ? 0.1 : [0.2, 0.1, 0.1], grain: X });
      }
      // jelly cube
      w.addVoxels({ material: MAT.JELLY, min: [3.0, 0.0, -0.6], dims: [7, 7, 7], cell: 0.12 });
      // steel lamp post bolted to the ground
      w.addVoxels({ material: MAT.STEEL, min: [2.8, 0, 1.4], dims: [2, 30, 2], cell: 0.1, grain: Y, pin: (x, y) => y < 0.01 });
    },
  },
};

// Small scene for headless testing (not listed in the UI).
export const TEST_SCENES = {
  tiny: {
    label: 'Tiny', camera: [2.5, 1.8, 2.5], target: [0, 0.4, 0],
    build(w) {
      w.addVoxels({ material: MAT.WOOD, min: [-0.6, 0.3, -0.1], dims: [12, 2, 2], cell: 0.1, grain: X });
      w.addVoxels({ material: MAT.GLASS, min: [-0.4, 0, 0.4], dims: [8, 6, 1], cell: [0.1, 0.1, 0.04] });
    },
  },
};

TEST_SCENES.contact = {
  label: 'Contact', camera: [2.5, 2, 3], target: [0, 0.3, 0],
  build(w) {
    w.addVoxels({ material: MAT.GLASS, min: [-0.6, 0, -0.6], dims: [12, 1, 12], cell: [0.1, 0.04, 0.1] });
    w.addVoxels({ material: MAT.WOOD, min: [-0.5, 0.8, -0.1], dims: [10, 2, 2], cell: 0.1, grain: X, velocity: [0, -3, 0] });
    w.addVoxels({ material: MAT.WOOD, min: [1.0, 0, -0.5], dims: [3, 2, 10], cell: 0.1, grain: Z });
    boulder(w, { center: [1.15, 1.2, 0], radius: 0.25, cell: 0.1, velocity: [0, -3, 0] });
  },
};

TEST_SCENES.beam = {
  label: 'Beam', camera: [0, 1.5, 5], target: [0, 0.8, 0],
  build(w) {
    for (const x of [-2.1, 1.7]) w.addVoxels({ material: MAT.STONE, min: [x, 0, -0.3], dims: [4, 8, 6], cell: 0.1 });
    w.addVoxels({ material: MAT.WOOD, min: [-2.0, 0.8, -0.1], dims: [40, 2, 2], cell: 0.1, grain: X });
  },
};

export function buildScene(key, seed = 7) {
  const w = new WorldBuilder(seed);
  (SCENES[key] ?? TEST_SCENES[key]).build(w);
  return w.build();
}

/** Kinematic pendulum ball for the lab scene. */
export function makeWreckingBall(def) {
  const g = 9.81;
  let theta = def.angle, omega = 0;
  const pivot = new THREE.Vector3(...def.pivot);
  const ball = {
    kinematic: true, radius: def.radius, mass: 1e9, pivot,
    pos: new THREE.Vector3(), vel: new THREE.Vector3(),
    update(dt) {
      omega += (-g / def.length) * Math.sin(theta) * dt;
      theta += omega * dt;
      this.pos.set(pivot.x + Math.sin(theta) * def.length, pivot.y - Math.cos(theta) * def.length, pivot.z);
      this.vel.set(Math.cos(theta) * def.length * omega, Math.sin(theta) * def.length * omega, 0);
    },
  };
  ball.update(0);
  return ball;
}

TEST_SCENES.steel = {
  label: 'Steel', camera: [0, 1.5, 5], target: [0, 0.8, 0],
  build(w) {
    for (const x of [-1.6, 1.2]) w.addVoxels({ material: MAT.STONE, min: [x, 0, -0.3], dims: [4, 8, 6], cell: 0.1 });
    w.addVoxels({ material: MAT.STEEL, min: [-1.5, 0.8, -0.1], dims: [30, 2, 2], cell: 0.1, grain: X });
  },
};
