// GPU tetrahedral FEM with tetrahedral fracture, written as WebGPU compute
// shaders in three.js TSL. Everything, fracture included, runs on the GPU.
//
// One substep:
//   1. tetPass      per tet: best-fit rotation of the rest shape onto the current
//                   corners (polar decomposition by quaternion iteration), emit a goal
//                   position per corner, plastic flow for metals, insert into hash grid
//   2. collidePass  per tet: sphere-sphere pushes against nearby tets (other bodies
//                   and detached fragments of the same body)
//   3. facePass     per shared face: how strongly do the two tets disagree about
//                   where their shared corners belong? Past the face's toughness the
//                   face breaks and its three nodes are marked for splitting
//   4. splitPass    per marked node: connected components of the tets around it
//                   (through unbroken faces); every extra component gets a fresh
//                   copy of the vertex, so the crack opens in the very same substep
//   5. projectPass  per vertex: mass-weighted average of the goals of the tet corners
//                   that use it, then ground / projectile / anchor constraints
//
// A substep is predictPass (Verlet prediction) followed by several Jacobi
// iterations of 1 (+2 on the first) and 5, with 3-4 on the last. The Jacobi
// iterations are accelerated with Chebyshev semi-iteration (Wang 2015, "A
// Chebyshev Semi-Iterative Approach for Accelerating Projective and
// Position-based Dynamics") so stiffness travels through long beams quickly.
//
// Per-tet rotation extraction follows Tetrament (github.com/zalo/Tetrament, MIT)
// and Müller et al. "A Robust Method to Extract the Rotational Part of Deformations".

import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Return, instanceIndex, instancedArray, uniform, uniformArray,
  float, int, uint, vec3, vec4, ivec3, ivec4, cross, dot, length, max, min, mix, sin, cos, abs,
  atomicAdd, atomicLoad, atomicStore, atomicFunc, select, intBitsToFloat,
  workgroupArray, workgroupBarrier, workgroupId, localId, hash, exp, sqrt,
} from 'three/tsl';
import { MATERIALS, TET_FACES } from './builder.js';

export const MAX_BALLS = 8;
// Ring buffer of cosmetic debris particles (dust, chips, glints, sparks); no physics cost.
export const MAX_PARTICLES = 8192;
// Particle kinds (= material ids): 0 wood chip, 1 glass glint, 2 steel spark, 3 stone dust, 5 rock dust
const DUST = (mat) => mat.equal(3).or(mat.equal(5));
// A tet that came loose from everything ("lone") is retired once it settles or after LONE_MAX s:
// it shrinks away over DYING s, leaves a particle behind and stops costing solver time.
const LONE_REST = 2, LONE_MAX = 6, DYING = 0.35;
const CLUSTER_YIELD = 3;

// ---- quaternion helpers (inlined TSL) ----
const rotate = (v, q) => v.add(cross(q.xyz, cross(q.xyz, v).add(v.mul(q.w))).mul(2.0));
const conj = (q) => vec4(q.xyz.negate(), q.w);
const qmul = (a, b) => vec4(
  a.w.mul(b.x).add(a.x.mul(b.w)).add(a.y.mul(b.z)).sub(a.z.mul(b.y)),
  a.w.mul(b.y).sub(a.x.mul(b.z)).add(a.y.mul(b.w)).add(a.z.mul(b.x)),
  a.w.mul(b.z).add(a.x.mul(b.y)).sub(a.y.mul(b.x)).add(a.z.mul(b.w)),
  a.w.mul(b.w).sub(a.x.mul(b.x)).sub(a.y.mul(b.y)).sub(a.z.mul(b.z)),
);

/** q <- rotation maximising alignment with covariance columns A0..A2 (warm-started). */
function extractRotation(A0, A1, A2, q, iterations) {
  for (let i = 0; i < iterations; i++) {
    const X = rotate(vec3(1, 0, 0), q), Y = rotate(vec3(0, 1, 0), q), Z = rotate(vec3(0, 0, 1), q);
    const omega = cross(X, A0).add(cross(Y, A1)).add(cross(Z, A2))
      .div(abs(dot(X, A0).add(dot(Y, A1)).add(dot(Z, A2))).add(1e-9)).toVar();
    const w = length(omega).toVar();
    const half = w.mul(0.5);
    const dq = vec4(omega.div(max(w, 1e-9)).mul(sin(half)), cos(half));
    q.assign(qmul(dq, q).normalize());
  }
  return q;
}

export class Simulation {
  constructor(renderer, mesh, opts = {}) {
    this.renderer = renderer;
    this.mesh = mesh;
    this.substeps = opts.substeps ?? 6;
    this.iterations = opts.iterations ?? 8;
    this.rho = opts.rho ?? null; // Chebyshev spectral radius estimate (null: derive from iterations)
    this.clusterWeight = opts.clusterWeight ?? 4;
    this.frameDt = 1 / 60;
    this.capacity = mesh.T * 4; // worst case: every tet corner its own vertex
    this.vertexCount = mesh.N;
    this.brokenCount = 0;
    this.simTime = 0;
    this.pendingRead = false;
    this.backlog = 0;
    this._createBuffers();
    this._createKernels();
  }

  get dt() { return this.frameDt / this.substeps; }

  _createBuffers() {
    const m = this.mesh;
    const T = m.T, F = Math.max(m.F, 1), V = this.capacity;

    // Vertex records, 4 x vec4: (pos, node id) (prev, pinned) (rest, object id) (previous iterate, mass)
    const verts = new Float32Array(V * 16);
    const dt = this.dt;
    for (let n = 0; n < m.N; n++) {
      const o = n * 16;
      for (let a = 0; a < 3; a++) {
        verts[o + a] = verts[o + 8 + a] = verts[o + 12 + a] = m.nodePos[n * 3 + a];
        verts[o + 4 + a] = m.nodePos[n * 3 + a] - m.nodeVel[n * 3 + a] * dt;
      }
      verts[o + 3] = n;
      verts[o + 7] = m.nodePinned[n];
      verts[o + 11] = m.nodeObj[n];
    }
    this.verts = instancedArray(verts, 'vec4');
    this.cornerVertex = instancedArray(new Int32Array(m.tetNodes), 'int'); // vertex used by tet corner t*4+k
    this.nodeRange = instancedArray(m.nodeRange, 'ivec4');
    this.nodeData = instancedArray(m.nodeData, 'int');
    this.nodeDirty = instancedArray(m.N, 'int');

    const restLocal = new Float32Array(T * 16), info = new Float32Array(T * 4), restC = new Float32Array(T * 4);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < 4; k++) {
        const n = m.tetNodes[t * 4 + k];
        for (let a = 0; a < 3; a++) restLocal[(t * 4 + k) * 4 + a] = m.nodePos[n * 3 + a] - m.tetRestC[t * 3 + a];
      }
      restLocal[(t * 4 + 3) * 4 + 3] = 1; // the w column holds the tet's rotation quaternion
      info.set([m.tetMass[t], m.tetRadius[t], m.tetMat[t], m.tetObj[t]], t * 4);
      restC.set([m.tetRestC[t * 3], m.tetRestC[t * 3 + 1], m.tetRestC[t * 3 + 2], m.tetExcl[t]], t * 4);
    }
    this.restLocal = instancedArray(restLocal, 'vec4');
    this.tetInfo = instancedArray(info, 'vec4');   // mass, radius, material, object
    this.tetRestC = instancedArray(restC, 'vec4'); // rest centroid, collision exclusion
    this.tetCent = instancedArray(T, 'vec4');
    // Three goal sets per tet corner: [tet fit | cluster layer A fit | cluster layer B fit].
    // Tet goals' w doubles as a scratch label in splitPass; cluster goals' w is their weight.
    this.goals = instancedArray(T * 4 * 3, 'vec4');
    this.clusterBuf = instancedArray(m.clusterBuf, 'vec4');
    this.clusterData = instancedArray(m.clusterData, 'int');
    this.clusterRest = instancedArray(m.clusterRest, 'vec4'); // per cluster entry: rest offset (yields for metals)
    // >= 0 seconds spent lone, >= 100 dying (100 + t), < 0 retired
    this.tetAge = instancedArray(T, 'float');
    const parts = new Float32Array(MAX_PARTICLES * 8);
    for (let i = 0; i < MAX_PARTICLES; i++) parts.set([0, -1000, 0, 99], i * 8);
    this.particles = instancedArray(parts, 'vec4'); // (pos, age) (vel, kind + seed fraction)
    this.tetNext = instancedArray(T, 'int');
    // hash grid sized to ~2 cells per tet (power of two, 4k..256k)
    this.gridCells = Math.min(1 << 18, Math.max(4096, 1 << Math.ceil(Math.log2(T * 2))));
    this.grid = instancedArray(this.gridCells, 'int').toAtomic();

    this.faceData = instancedArray(m.faceData, 'ivec4');
    this.faceState = instancedArray(F, 'int');
    this.slotFace = instancedArray(m.slotFace, 'int');
    // [broken faces, vertices, particles spawned, -, then 4 ints per ball: momentum pushed onto it]
    const counters = new Int32Array(4 + MAX_BALLS * 4);
    counters[1] = m.N;
    this.counters = instancedArray(counters, 'int').toAtomic();
    // Projectiles live on the GPU too: 3 x vec4 per slot (pos, radius) (prev, 1/mass) (0, 0, 0, kinematic)
    this.balls = instancedArray(MAX_BALLS * 3, 'vec4');
    this.ballSlots = new Array(MAX_BALLS).fill(null);

    let maxR = 0;
    for (let t = 0; t < T; t++) maxR = Math.max(maxR, m.tetRadius[t]);
    this.u = {
      dt: uniform(this.dt),
      frameDt: uniform(1 / 60),
      gravity: uniform(new THREE.Vector3(0, -9.81, 0)),
      damping: uniform(0.9995),
      omega: uniform(1.0),
      insertGrid: uniform(1, 'int'),
      clusterWeight: uniform(this.clusterWeight),
      toughness: uniform(1.0),
      kinPos: uniform(new THREE.Vector3()), // driven position of a kinematic ball (wrecking ball)
      kinVel: uniform(new THREE.Vector3()),
      mat: uniformArray(MATERIALS.map((x) => new THREE.Vector4(x.stiffness, x.yield, x.plastic, x.friction)), 'vec4'),
      pushOrigin: uniform(new THREE.Vector3()),
      pushDir: uniform(new THREE.Vector3()),
      pushStrength: uniform(0),
      cellSize: uniform(maxR * 2),
    };
  }

  _createKernels() {
    const { T, F, N } = this.mesh;
    const u = this.u, CAP = this.capacity;
    const { verts, cornerVertex, nodeRange, nodeData, nodeDirty, restLocal, tetInfo, tetRestC, tetCent,
      goals, tetNext, grid, faceData, faceState, counters, balls, clusterBuf, clusterData, clusterRest } = this;
    const C = this.mesh.C;
    const ballAcc = { element: (i) => counters.element(i.add(4)) };
    const pos = (v) => verts.element(v.mul(4));
    const prev = (v) => verts.element(v.mul(4).add(1));
    const rest = (v) => verts.element(v.mul(4).add(2));
    const older = (v) => verts.element(v.mul(4).add(3));

    const cellOf = (p) => ivec3(p.div(u.cellSize).floor());
    const hashCell = (c) => uint(c.x).mul(uint(73856093)).bitXor(uint(c.y).mul(uint(19349663)))
      .bitXor(uint(c.z).mul(uint(83492791))).bitAnd(uint(this.gridCells - 1));

    this.clearGrid = Fn(() => {
      atomicStore(grid.element(instanceIndex), int(-1));
    })().compute(this.gridCells);

    this.tetPass = Fn(() => {
      const t = int(instanceIndex);
      If(t.greaterThanEqual(T), () => { Return(); });
      const info = tetInfo.element(t).toVar();
      If(info.x.equal(0), () => { Return(); }); // retired
      const b = t.mul(4);
      const p0 = pos(cornerVertex.element(b)).xyz.toVar(), p1 = pos(cornerVertex.element(b.add(1))).xyz.toVar();
      const p2 = pos(cornerVertex.element(b.add(2))).xyz.toVar(), p3 = pos(cornerVertex.element(b.add(3))).xyz.toVar();
      const c = p0.add(p1).add(p2).add(p3).mul(0.25).toVar();
      const l0 = restLocal.element(b).toVar(), l1 = restLocal.element(b.add(1)).toVar();
      const l2 = restLocal.element(b.add(2)).toVar(), l3 = restLocal.element(b.add(3)).toVar();
      const r0 = l0.xyz, r1 = l1.xyz, r2 = l2.xyz, r3 = l3.xyz;
      const q0 = p0.sub(c).toVar(), q1 = p1.sub(c).toVar(), q2 = p2.sub(c).toVar(), q3 = p3.sub(c).toVar();
      const A0 = q0.mul(r0.x).add(q1.mul(r1.x)).add(q2.mul(r2.x)).add(q3.mul(r3.x)).toVar();
      const A1 = q0.mul(r0.y).add(q1.mul(r1.y)).add(q2.mul(r2.y)).add(q3.mul(r3.y)).toVar();
      const A2 = q0.mul(r0.z).add(q1.mul(r1.z)).add(q2.mul(r2.z)).add(q3.mul(r3.z)).toVar();
      const q = vec4(l0.w, l1.w, l2.w, l3.w).toVar();
      extractRotation(A0, A1, A2, q, 1); // warm-started every iteration, so one step is enough

      const mp = u.mat.element(int(info.z)).toVar(); // stiffness, yield, plastic rate, friction
      goals.element(b).assign(vec4(mix(p0, rotate(r0, q).add(c), mp.x), 0));
      goals.element(b.add(1)).assign(vec4(mix(p1, rotate(r1, q).add(c), mp.x), 0));
      goals.element(b.add(2)).assign(vec4(mix(p2, rotate(r2, q).add(c), mp.x), 0));
      goals.element(b.add(3)).assign(vec4(mix(p3, rotate(r3, q).add(c), mp.x), 0));

      // Plasticity: once strained past yield, part of the deformation becomes the
      // new rest shape (steel bends and stays bent).
      If(mp.y.lessThan(1.0), () => {
        const qi = conj(q);
        const e0 = rotate(q0, qi).sub(r0).toVar(), e1 = rotate(q1, qi).sub(r1).toVar();
        const e2 = rotate(q2, qi).sub(r2).toVar(), e3 = rotate(q3, qi).sub(r3).toVar();
        const strain = max(max(length(e0), length(e1)), max(length(e2), length(e3))).div(info.y.mul(2.0)).toVar();
        If(strain.greaterThan(mp.y), () => {
          const k = mp.z.mul(float(1).sub(mp.y.div(strain)));
          const n0 = r0.add(e0.mul(k)).toVar(), n1 = r1.add(e1.mul(k)).toVar();
          const n2 = r2.add(e2.mul(k)).toVar(), n3 = r3.add(e3.mul(k)).toVar();
          const mean = n0.add(n1).add(n2).add(n3).mul(0.25).toVar();
          l0.xyz.assign(n0.sub(mean));
          l1.xyz.assign(n1.sub(mean));
          l2.xyz.assign(n2.sub(mean));
          l3.xyz.assign(n3.sub(mean));
        });
      });
      restLocal.element(b).assign(vec4(l0.xyz, q.x));
      restLocal.element(b.add(1)).assign(vec4(l1.xyz, q.y));
      restLocal.element(b.add(2)).assign(vec4(l2.xyz, q.z));
      restLocal.element(b.add(3)).assign(vec4(l3.xyz, q.w));

      tetCent.element(t).assign(vec4(c, info.y));
      If(u.insertGrid.equal(1), () => {
        tetNext.element(t).assign(atomicFunc('atomicExchange', grid.element(hashCell(cellOf(c))), t));
      });
    })().compute(T);

    this.collidePass = Fn(() => {
      const t = int(instanceIndex);
      If(t.greaterThanEqual(T), () => { Return(); });
      const info = tetInfo.element(t).toVar();
      If(info.x.equal(0), () => { Return(); });
      const cent = tetCent.element(t).toVar();
      const restC = tetRestC.element(t).toVar();
      const base = cellOf(cent.xyz).sub(1).toVar();
      const push = vec3(0).toVar();
      Loop({ start: 0, end: 3, type: 'int', name: 'gx' }, ({ gx }) => {
        Loop({ start: 0, end: 3, type: 'int', name: 'gy' }, ({ gy }) => {
          Loop({ start: 0, end: 3, type: 'int', name: 'gz' }, ({ gz }) => {
            const o = atomicLoad(grid.element(hashCell(base.add(ivec3(gx, gy, gz))))).toVar();
            const guard = int(0).toVar();
            Loop(o.greaterThanEqual(0).and(guard.lessThan(256)), () => {
              guard.addAssign(1);
              const oInfo = tetInfo.element(o).toVar();
              const oRest = tetRestC.element(o).toVar();
              // Tets of the same body only collide when they were far apart at rest,
              // i.e. once fragments have come loose and swing back into each other.
              const restD = length(oRest.xyz.sub(restC.xyz));
              const ok = o.notEqual(t).and(oInfo.w.notEqual(info.w).or(restD.greaterThan(max(restC.w, oRest.w))));
              If(ok, () => {
                const d = cent.xyz.sub(tetCent.element(o).xyz).toVar();
                const dist = length(d).toVar();
                const pen = info.y.add(oInfo.y).sub(dist).toVar();
                If(pen.greaterThan(0).and(dist.greaterThan(1e-6)), () => {
                  push.addAssign(d.div(dist).mul(pen).mul(oInfo.x.div(info.x.add(oInfo.x))));
                });
              });
              o.assign(tetNext.element(o));
            });
          });
        });
      });
      If(dot(push, push).greaterThan(0), () => {
        const b = t.mul(4);
        const p = vec4(push, 0);
        goals.element(b).addAssign(p);
        goals.element(b.add(1)).addAssign(p);
        goals.element(b.add(2)).addAssign(p);
        goals.element(b.add(3)).addAssign(p);
      });
    })().compute(T);

    const goalOf = (tet, packed, i) => goals.element(tet.mul(4).add(packed.div(1 << (i * 2)).mod(4))).xyz;
    this.facePass = Fn(() => {
      const f = int(instanceIndex);
      If(f.greaterThanEqual(F), () => { Return(); });
      If(faceState.element(f).notEqual(0), () => { Return(); });
      const fi = faceData.element(f.mul(2)).toVar();
      const n = tetCent.element(fi.y).xyz.sub(tetCent.element(fi.x).xyz).normalize().toVar();
      const worst = float(0).toVar();
      for (let i = 0; i < 3; i++) {
        // A's fit pulls the shared corner toward A, B's toward B -> the face opens
        const d = goalOf(fi.x, fi.z, i).sub(goalOf(fi.y, fi.w, i)).toVar();
        const dn = dot(d, n);
        const tension = max(dn.negate(), 0);
        const shear = length(d.sub(n.mul(dn)));
        worst.assign(max(worst, tension.add(shear.mul(0.6))));
      }
      const size = tetInfo.element(fi.x).y.add(tetInfo.element(fi.y).y);
      const fn = faceData.element(f.mul(2).add(1)).toVar();
      If(worst.div(size).greaterThan(intBitsToFloat(fn.w).mul(u.toughness)), () => {
        faceState.element(f).assign(1);
        nodeDirty.element(fn.x).assign(1);
        nodeDirty.element(fn.y).assign(1);
        nodeDirty.element(fn.z).assign(1);
        atomicAdd(counters.element(0), 1);
      });
    })().compute(Math.max(F, 1));

    // Union-find by label propagation over the tets around one node, using
    // goals[corner].w as scratch (each corner belongs to exactly one node).
    this.splitPass = Fn(() => {
      const n = int(instanceIndex);
      If(n.greaterThanEqual(N), () => { Return(); });
      If(nodeDirty.element(n).equal(0), () => { Return(); });
      nodeDirty.element(n).assign(0);
      const r = nodeRange.element(n).toVar(); // corner start, corner count, pair start, pair count
      const corner = (i) => nodeData.element(r.x.add(i));
      const label = (c) => goals.element(c).w;
      Loop({ start: 0, end: r.y, type: 'int', name: 'i' }, ({ i }) => {
        label(corner(i)).assign(float(i));
      });
      const changed = int(1).toVar();
      const iter = int(0).toVar();
      Loop(changed.equal(1).and(iter.lessThan(40)), () => {
        changed.assign(0);
        iter.addAssign(1);
        Loop({ start: 0, end: r.w, type: 'int', name: 'p' }, ({ p }) => {
          const o = r.z.add(p.mul(3));
          const ca = nodeData.element(o).toVar(), cb = nodeData.element(o.add(1)).toVar();
          If(faceState.element(nodeData.element(o.add(2))).equal(0), () => {
            const la = label(ca).toVar(), lb = label(cb).toVar();
            If(la.notEqual(lb), () => {
              const m = min(la, lb);
              label(ca).assign(m);
              label(cb).assign(m);
              changed.assign(1);
            });
          });
        });
      });
      // The first component keeps the vertex; each further one gets a copy.
      Loop({ start: 0, end: r.y, type: 'int', name: 'i' }, ({ i }) => {
        const ci = corner(i).toVar();
        If(label(ci).equal(float(i)), () => {
          const vold = cornerVertex.element(ci).toVar();
          const taken = int(0).toVar();
          Loop({ start: 0, end: i, type: 'int', name: 'j' }, ({ j }) => {
            const cj = corner(j).toVar();
            If(label(cj).equal(float(j)).and(cornerVertex.element(cj).equal(vold)), () => { taken.assign(1); });
          });
          const v = vold.toVar();
          If(taken.equal(1), () => {
            const nv = atomicAdd(counters.element(1), 1).toVar();
            If(nv.lessThan(CAP), () => {
              pos(nv).assign(pos(vold));
              prev(nv).assign(prev(vold));
              rest(nv).assign(rest(vold));
              older(nv).assign(older(vold));
              v.assign(nv);
            });
          });
          Loop({ start: i, end: r.y, type: 'int', name: 'j' }, ({ j }) => {
            const cj = corner(j).toVar();
            If(label(cj).equal(float(i)), () => { cornerVertex.element(cj).assign(v); });
          });
        });
      });
    })().compute(N);

    const activeVertex = () => {
      const v = int(instanceIndex);
      If(v.greaterThanEqual(min(atomicLoad(counters.element(1)), CAP)), () => { Return(); });
      return v;
    };

    // Verlet prediction: prev <- pos, pos <- pos + velocity * dt (+ gravity)
    this.predictPass = Fn(() => {
      const v = activeVertex();
      const pv = prev(v).toVar();
      const p = pos(v).xyz.toVar();
      If(pv.w.greaterThan(0.5), () => { // pinned
        const rp = rest(v).xyz.toVar();
        pos(v).xyz.assign(rp);
        prev(v).xyz.assign(rp);
        older(v).xyz.assign(rp);
        Return();
      });
      const vel = p.sub(pv.xyz).mul(u.damping).add(u.gravity.mul(u.dt.mul(u.dt)));
      const np = p.add(vel).toVar();
      prev(v).xyz.assign(p);
      pos(v).xyz.assign(np);
      older(v).xyz.assign(np);
    })().compute(CAP);

    // Jacobi projection: average the tet goals, Chebyshev-accelerate, then contacts.
    this.projectPass = Fn(() => {
      const v = activeVertex();
      If(prev(v).w.greaterThan(0.5), () => { Return(); });
      const cur = pos(v).toVar();
      const r = nodeRange.element(int(cur.w)).toVar();
      const sum = vec3(0).toVar();
      const wsum = float(0).toVar();
      const mass = float(0).toVar();
      const friction = float(0).toVar();
      Loop({ start: 0, end: r.y, type: 'int', name: 'i' }, ({ i }) => {
        const c = nodeData.element(r.x.add(i)).toVar();
        If(cornerVertex.element(c).equal(v), () => {
          const ti = tetInfo.element(c.div(4)).toVar();
          const ga = goals.element(c.add(T * 4)).toVar(), gb = goals.element(c.add(T * 8)).toVar();
          sum.addAssign(goals.element(c).xyz.add(ga.xyz.mul(ga.w)).add(gb.xyz.mul(gb.w)).mul(ti.x));
          wsum.addAssign(ti.x.mul(ga.w.add(gb.w).add(1)));
          mass.addAssign(ti.x.mul(0.25));
          friction.assign(u.mat.element(int(ti.z)).w);
        });
      });
      If(wsum.lessThanEqual(0), () => { Return(); });
      const old = older(v).xyz.toVar();
      const np = mix(old, sum.div(wsum), u.omega).toVar();
      older(v).assign(vec4(cur.xyz, mass));

      // ground plane with positional friction
      If(np.y.lessThan(0), () => {
        np.y.assign(0);
        const pp = prev(v).xyz;
        np.x.assign(mix(np.x, pp.x, friction.mul(0.5)));
        np.z.assign(mix(np.z, pp.z, friction.mul(0.5)));
      });

      // projectiles: split the overlap by inverse mass, hand the ball its share of momentum
      Loop({ start: 0, end: MAX_BALLS, type: 'int', name: 'bi' }, ({ bi }) => {
        const bp = balls.element(bi.mul(3)).toVar();
        const d = np.sub(bp.xyz).toVar();
        const dist = length(d).toVar();
        If(dist.lessThan(bp.w).and(dist.greaterThan(1e-6)), () => {
          const wb = balls.element(bi.mul(3).add(1)).w;
          const wv = float(1).div(mass);
          const dx = d.div(dist).mul(bp.w.sub(dist)).mul(wv.div(wv.add(wb))).toVar();
          np.addAssign(dx);
          // debris pinned between ball and ground must not launch the ball
          If(wb.greaterThan(0).and(np.y.greaterThan(0.002)), () => {
            const mom = dx.mul(mass).mul(1e6).toVar();
            atomicAdd(ballAcc.element(bi.mul(4)), int(mom.x));
            atomicAdd(ballAcc.element(bi.mul(4).add(1)), int(mom.y));
            atomicAdd(ballAcc.element(bi.mul(4).add(2)), int(mom.z));
          });
        });
      });
      pos(v).xyz.assign(np);
    })().compute(CAP);

    // Volumetric contact: a vertex found inside a tet of another body (or of a distant,
    // already detached part of its own body) is pushed out through that tet's nearest
    // face. The push is shared by mass and the tangential slip relative to the tet is
    // damped (friction). Runs once per substep after the last projection.
    this.contactPass = Fn(() => {
      const v = activeVertex();
      If(prev(v).w.greaterThan(0.5), () => { Return(); });
      const p = pos(v).xyz.toVar();
      const rv = rest(v).toVar(); // rest position, object
      const mv = older(v).w.toVar();
      const best = vec4(0).toVar(); // push direction * depth, other tet's mass
      const bestTet = int(-1).toVar();
      const base = cellOf(p).sub(1).toVar();
      Loop({ start: 0, end: 3, type: 'int', name: 'gx' }, ({ gx }) => {
        Loop({ start: 0, end: 3, type: 'int', name: 'gy' }, ({ gy }) => {
          Loop({ start: 0, end: 3, type: 'int', name: 'gz' }, ({ gz }) => {
            const t = atomicLoad(grid.element(hashCell(base.add(ivec3(gx, gy, gz))))).toVar();
            const guard = int(0).toVar();
            Loop(t.greaterThanEqual(0).and(guard.lessThan(256)), () => {
              guard.addAssign(1);
              const info = tetInfo.element(t).toVar();
              const rc = tetRestC.element(t).toVar();
              const b = t.mul(4);
              const i0 = cornerVertex.element(b).toVar(), i1 = cornerVertex.element(b.add(1)).toVar();
              const i2 = cornerVertex.element(b.add(2)).toVar(), i3 = cornerVertex.element(b.add(3)).toVar();
              const own = i0.equal(v).or(i1.equal(v)).or(i2.equal(v)).or(i3.equal(v));
              const other = info.w.notEqual(rv.w).or(length(rv.xyz.sub(rc.xyz)).greaterThan(rc.w));
              If(own.not().and(other), () => {
                const P = [pos(i0).xyz.toVar(), pos(i1).xyz.toVar(), pos(i2).xyz.toVar(), pos(i3).xyz.toVar()];
                const inside = int(1).toVar();
                const depth = float(-1e9).toVar();
                const nrm = vec3(0).toVar();
                for (const [a, bb, cc] of TET_FACES) {
                  const n = cross(P[bb].sub(P[a]), P[cc].sub(P[a])).normalize().toVar();
                  const d = dot(p.sub(P[a]), n).toVar();
                  If(d.greaterThanEqual(0), () => { inside.assign(0); });
                  If(d.greaterThan(depth), () => { depth.assign(d); nrm.assign(n); });
                }
                If(inside.equal(1).and(depth.negate().greaterThan(length(best.xyz))), () => {
                  best.assign(vec4(nrm.mul(depth.negate()), info.x));
                  bestTet.assign(t);
                });
              });
              t.assign(tetNext.element(t));
            });
          });
        });
      });
      If(bestTet.greaterThanEqual(0), () => {
        const b = bestTet.mul(4);
        const tv = vec3(0).toVar(); // the tet's motion this substep
        for (let k = 0; k < 4; k++) {
          const iv = cornerVertex.element(b.add(k));
          tv.addAssign(pos(iv).xyz.sub(prev(iv).xyz).mul(0.25));
        }
        const share = best.w.div(best.w.add(mv));
        const np = p.add(best.xyz.mul(share)).toVar();
        const n = best.xyz.normalize();
        const rel = np.sub(prev(v).xyz).sub(tv).toVar();
        const tang = rel.sub(n.mul(dot(rel, n)));
        np.subAssign(tang.mul(0.4));
        pos(v).xyz.assign(np);
      });
    })().compute(CAP);

    // Lattice shape matching: one 64-thread workgroup per cluster reduces the mass-weighted
    // covariance of its corners, extracts the best-fit rotation and writes a goal for
    // every corner. Since the rest offsets are mass-centred, sum(m p r^T) is already the
    // covariance about the current centroid, so a single reduction pass suffices.
    const WG = 64;
    const s0 = workgroupArray('vec4', WG), s1 = workgroupArray('vec4', WG);
    const s2 = workgroupArray('vec4', WG), s3 = workgroupArray('vec4', WG);
    this.clusterPass = Fn(() => {
      const ci = int(workgroupId.x), lane = int(localId.x);
      const base = ci.mul(4);
      const rc = clusterBuf.element(base).toVar();       // rest centroid, weight (0 once broken, -1 retired)
      const q = clusterBuf.element(base.add(1)).toVar(); // rotation, warm start
      const layer = clusterBuf.element(base.add(2)).w;
      const range = ivec4(clusterBuf.element(base.add(3))).toVar(); // corners start/count, faces start/count
      const seg = select(layer.greaterThan(0.5), int(T * 8), int(T * 4));
      If(rc.w.equal(0), () => {
        // cracked: zero its goal weights once, then skip it for good
        Loop({ start: lane, end: range.y, type: 'int', update: WG, name: 'i' }, ({ i }) => {
          goals.element(seg.add(clusterData.element(range.x.add(i)))).w.assign(0);
        });
        If(lane.equal(0), () => { clusterBuf.element(base).w.assign(-1); });
      });
      // No early return: barriers below must stay in uniform control flow, so dead
      // clusters just run their loops zero times.
      const n = select(rc.w.greaterThan(0), range.y, int(0)).toVar();
      const a0 = vec4(0).toVar(), a1 = vec3(0).toVar(), a2 = vec3(0).toVar(), a3 = vec3(0).toVar();
      Loop({ start: lane, end: n, type: 'int', update: WG, name: 'i' }, ({ i }) => {
        const c = clusterData.element(range.x.add(i)).toVar();
        const m = tetInfo.element(c.div(4)).x.mul(0.25).toVar();
        const p = pos(cornerVertex.element(c)).xyz.mul(m).toVar();
        const r = clusterRest.element(range.x.add(i)).xyz.toVar();
        a0.addAssign(vec4(p, m));
        a1.addAssign(p.mul(r.x));
        a2.addAssign(p.mul(r.y));
        a3.addAssign(p.mul(r.z));
      });
      s0.element(lane).assign(a0);
      s1.element(lane).assign(vec4(a1, 0));
      s2.element(lane).assign(vec4(a2, 0));
      s3.element(lane).assign(vec4(a3, 0));
      workgroupBarrier();
      for (let st = WG / 2; st >= 1; st /= 2) {
        If(lane.lessThan(st), () => {
          s0.element(lane).addAssign(s0.element(lane.add(st)));
          s1.element(lane).addAssign(s1.element(lane.add(st)));
          s2.element(lane).addAssign(s2.element(lane.add(st)));
          s3.element(lane).addAssign(s3.element(lane.add(st)));
        });
        workgroupBarrier();
      }
      const tot = s0.element(int(0)).toVar();
      const cm = tot.xyz.div(max(tot.w, 1e-9)).toVar();
      extractRotation(s1.element(int(0)).xyz.toVar(), s2.element(int(0)).xyz.toVar(), s3.element(int(0)).xyz.toVar(), q, 2);
      Loop({ start: lane, end: n, type: 'int', update: WG, name: 'i' }, ({ i }) => {
        const c = clusterData.element(range.x.add(i)).toVar();
        const r = clusterRest.element(range.x.add(i)).xyz;
        goals.element(seg.add(c)).assign(vec4(rotate(r, q).add(cm), rc.w.mul(u.clusterWeight)));
      });
      // Plasticity (metals): if the cluster as a whole is strained past yield, part of the
      // deformation becomes its new rest shape, so a bent bar stays bent. One factor for
      // the whole cluster keeps the rest offsets mass-centred.
      const mp = u.mat.element(int(tetInfo.element(clusterData.element(range.x).div(4)).z)).toVar();
      const np = select(mp.y.lessThan(1.0), n, int(0)).toVar();
      // a cluster spans several cells, so it yields at a higher strain than one tet: below
      // that, its deformation is the solver's own elastic give and must spring back
      const yieldC = mp.y.mul(CLUSTER_YIELD);
      {
        workgroupBarrier(); // everyone has read the reduction results
        const qi = conj(q);
        const acc = vec4(0).toVar();
        Loop({ start: lane, end: np, type: 'int', update: WG, name: 'i' }, ({ i }) => {
          const c = clusterData.element(range.x.add(i)).toVar();
          const m = tetInfo.element(c.div(4)).x;
          const r = clusterRest.element(range.x.add(i)).xyz;
          const e = rotate(pos(cornerVertex.element(c)).xyz.sub(cm), qi).sub(r).toVar();
          acc.addAssign(vec4(dot(e, e).mul(m), dot(r, r).mul(m), 0, 0));
        });
        s0.element(lane).assign(acc);
        workgroupBarrier();
        for (let st = WG / 2; st >= 1; st /= 2) {
          If(lane.lessThan(st), () => { s0.element(lane).addAssign(s0.element(lane.add(st))); });
          workgroupBarrier();
        }
        const strain = sqrt(s0.element(int(0)).x.div(max(s0.element(int(0)).y, 1e-12))).toVar();
        If(strain.greaterThan(yieldC), () => {
          const k = mp.z.mul(float(1).sub(yieldC.div(strain))).toVar();
          Loop({ start: lane, end: np, type: 'int', update: WG, name: 'i' }, ({ i }) => {
            const c = clusterData.element(range.x.add(i)).toVar();
            const r = clusterRest.element(range.x.add(i)).xyz.toVar();
            const e = rotate(pos(cornerVertex.element(c)).xyz.sub(cm), qi).sub(r);
            clusterRest.element(range.x.add(i)).xyz.assign(r.add(e.mul(k)));
          });
        });
      }
      // a crack through the cluster switches it off from the next iteration on
      Loop({ start: lane, end: select(n.greaterThan(0), range.w, int(0)), type: 'int', update: WG, name: 'i' }, ({ i }) => {
        If(faceState.element(clusterData.element(range.z.add(i))).notEqual(0), () => {
          clusterBuf.element(base).w.assign(0);
        });
      });
      If(lane.equal(0).and(n.greaterThan(0)), () => {
        clusterBuf.element(base.add(1)).assign(q);
        clusterBuf.element(base.add(2)).xyz.assign(cm);
      });
    })().compute(Math.max(C, 1) * WG, [WG]);

    // ---- cosmetic debris ----
    const { particles, tetAge, slotFace } = this;
    const PM = MAX_PARTICLES - 1;
    const spawn = (p, v, kind, seed) => {
      const idx = atomicAdd(counters.element(2), 1).bitAnd(PM).toVar();
      const h1 = hash(uint(idx).add(uint(seed))), h2 = hash(uint(idx).mul(3).add(7)), h3 = hash(uint(idx).mul(5).add(11));
      const jitter = vec3(h1, h2, h3).sub(0.5).toVar();
      const dust = DUST(kind);
      const vel = v.add(jitter.mul(select(dust, float(1.2), float(2.5)))).add(vec3(0, select(dust, float(0.4), float(1.0)), 0));
      particles.element(idx.mul(2)).assign(vec4(p.add(jitter.mul(0.05)), 0));
      particles.element(idx.mul(2).add(1)).assign(vec4(vel, float(kind).add(h1.mul(0.98))));
    };

    // Newly broken faces (state 1 -> 2) throw out a dust puff / chip / glint. Once per frame.
    this.spawnPass = Fn(() => {
      const f = int(instanceIndex);
      If(f.greaterThanEqual(F), () => { Return(); });
      If(faceState.element(f).notEqual(1), () => { Return(); });
      faceState.element(f).assign(2);
      const fi = faceData.element(f.mul(2)).toVar();
      const fn = faceData.element(f.mul(2).add(1)).toVar();
      const kind = int(tetInfo.element(fi.x).z).toVar();
      If(kind.equal(4), () => { Return(); }); // jelly
      If(kind.equal(1).and(hash(uint(f)).greaterThan(0.3)), () => { Return(); }); // glass: sparse slivers, the tets are the shards
      const c = pos(fn.x).xyz.add(pos(fn.y).xyz).add(pos(fn.z).xyz).div(3);
      const v = pos(fn.x).xyz.sub(prev(fn.x).xyz).div(u.dt);
      spawn(c, v, kind, f);
    })().compute(Math.max(F, 1));

    this.particlePass = Fn(() => {
      const i = int(instanceIndex);
      const a = particles.element(i.mul(2)).toVar();
      If(a.w.greaterThan(8), () => { Return(); });
      const b = particles.element(i.mul(2).add(1)).toVar();
      const dt = u.frameDt;
      const dust = DUST(int(b.w));
      const v = b.xyz.toVar();
      If(dust, () => {
        v.mulAssign(exp(dt.mul(-2.5)));
        v.y.addAssign(dt.mul(0.25)); // warm dust drifts up a little
      }).Else(() => {
        v.mulAssign(exp(dt.mul(-0.3)));
        v.y.subAssign(dt.mul(9.81));
      });
      const p = a.xyz.add(v.mul(dt)).toVar();
      If(p.y.lessThan(0.008), () => {
        p.y.assign(0.008);
        v.assign(vec3(v.x.mul(0.5), abs(v.y).mul(0.25), v.z.mul(0.5)));
      });
      particles.element(i.mul(2)).assign(vec4(p, a.w.add(dt)));
      particles.element(i.mul(2).add(1)).xyz.assign(v);
    })().compute(MAX_PARTICLES);

    // Retire lone tets (no unbroken face left): once at rest or after a while they shrink
    // away, leave a particle and drop out of the solver (mass 0, vertices pinned in place).
    this.despawnPass = Fn(() => {
      const t = int(instanceIndex);
      If(t.greaterThanEqual(T), () => { Return(); });
      const age = tetAge.element(t).toVar();
      If(age.lessThan(0), () => { Return(); });
      const b = t.mul(4);
      If(age.greaterThanEqual(100), () => {
        age.addAssign(u.frameDt);
        If(age.greaterThanEqual(100 + DYING), () => {
          tetInfo.element(t).x.assign(0);
          age.assign(-1);
          for (let k = 0; k < 4; k++) {
            const v = cornerVertex.element(b.add(k)).toVar();
            rest(v).xyz.assign(pos(v).xyz);
            prev(v).w.assign(1);
          }
        });
        tetAge.element(t).assign(age);
        Return();
      });
      const lone = int(1).toVar();
      for (let k = 0; k < 4; k++) {
        const f = slotFace.element(b.add(k)).toVar();
        If(f.greaterThanEqual(0), () => { If(faceState.element(f).equal(0), () => { lone.assign(0); }); });
      }
      If(lone.equal(0), () => { Return(); });
      const c = vec3(0).toVar(), v = vec3(0).toVar();
      for (let k = 0; k < 4; k++) {
        const iv = cornerVertex.element(b.add(k));
        c.addAssign(pos(iv).xyz.mul(0.25));
        v.addAssign(pos(iv).xyz.sub(prev(iv).xyz).mul(0.25));
      }
      v.divAssign(u.dt);
      age.addAssign(u.frameDt);
      If(age.greaterThan(LONE_MAX).or(age.greaterThan(LONE_REST).and(length(v).lessThan(0.3))), () => {
        const kind = int(tetInfo.element(t).z);
        If(kind.notEqual(4), () => { spawn(c, v.mul(0.3), kind, t); });
        age.assign(100);
      });
      tetAge.element(t).assign(age);
    })().compute(T);

    // Ball integration (one thread per slot), mirrors predictPass.
    this.ballPredict = Fn(() => {
      const i = int(instanceIndex);
      const rec = balls.element(i.mul(3)).toVar();
      If(rec.w.lessThanEqual(0), () => { Return(); });
      If(balls.element(i.mul(3).add(2)).w.greaterThan(0.5), () => {
        balls.element(i.mul(3)).xyz.assign(u.kinPos);
        balls.element(i.mul(3).add(1)).xyz.assign(u.kinPos.sub(u.kinVel.mul(u.dt)));
        Return();
      });
      const pv = balls.element(i.mul(3).add(1)).xyz.toVar();
      const vel = rec.xyz.sub(pv).mul(0.9999).add(u.gravity.mul(u.dt.mul(u.dt))).toVar();
      const np = rec.xyz.add(vel).toVar();
      const nprev = rec.xyz.toVar();
      If(np.y.lessThan(rec.w), () => { // ground: no bounce, rolling friction
        np.y.assign(rec.w);
        nprev.y.assign(rec.w);
        nprev.x.assign(mix(nprev.x, np.x, 0.01));
        nprev.z.assign(mix(nprev.z, np.z, 0.01));
      });
      balls.element(i.mul(3)).xyz.assign(np);
      balls.element(i.mul(3).add(1)).xyz.assign(nprev);
    })().compute(MAX_BALLS);

    // Apply the momentum vertices pushed onto each ball during the last projection.
    this.ballApply = Fn(() => {
      const i = int(instanceIndex);
      const acc = vec3(
        float(atomicLoad(ballAcc.element(i.mul(4)))),
        float(atomicLoad(ballAcc.element(i.mul(4).add(1)))),
        float(atomicLoad(ballAcc.element(i.mul(4).add(2)))),
      ).toVar();
      atomicStore(ballAcc.element(i.mul(4)), int(0));
      atomicStore(ballAcc.element(i.mul(4).add(1)), int(0));
      atomicStore(ballAcc.element(i.mul(4).add(2)), int(0));
      const w = balls.element(i.mul(3).add(1)).w;
      balls.element(i.mul(3)).xyz.subAssign(acc.mul(w).mul(1e-6));
    })().compute(MAX_BALLS);

    // Force push: shove everything near a ray (velocity change via prev position).
    this.pushPass = Fn(() => {
      const v = activeVertex();
      const rel = pos(v).xyz.sub(u.pushOrigin).toVar();
      const along = dot(rel, u.pushDir).toVar();
      const off = length(rel.sub(u.pushDir.mul(along)));
      const radius = along.mul(0.15).add(0.35);
      const fall = float(1).sub(off.div(radius)).clamp(0, 1).mul(select(along.greaterThan(0), float(1), float(0)));
      prev(v).xyz.subAssign(u.pushDir.mul(fall.mul(u.pushStrength).mul(u.dt)));
    })().compute(CAP);

    this.kernels = [this.spawnPass, this.particlePass, this.despawnPass, this.contactPass, this.clusterPass, this.ballPredict, this.ballApply, this.clearGrid, this.predictPass, this.tetPass, this.collidePass, this.facePass, this.splitPass, this.projectPass, this.pushPass];
  }

  /** Advances `steps` substeps. Projectiles are moved on the CPU between substeps. */
  step(steps) {
    // Backpressure: if last frame's readback hasn't come back the GPU is behind;
    // skip stepping instead of queueing ever more work.
    if (this.pendingRead && ++this.backlog > 2) return false;
    const dt = this.dt;
    this.u.dt.value = dt;
    const K = this.iterations;
    const rho = this.rho ?? Math.min(0.97, Math.max(0.6, 1 - 1.2 / K));
    const rho2 = rho * rho;
    for (let s = 0; s < steps; s++) {
      this._updateBalls(dt);
      this.renderer.compute([this.ballPredict, this.clearGrid, this.predictPass]);
      let omega = 1;
      for (let k = 0; k < K; k++) {
        // Chebyshev weights; plain Jacobi for the first two iterations
        omega = k < 2 ? 1 : k === 2 ? 2 / (2 - rho2) : 4 / (4 - rho2 * omega);
        this.u.omega.value = omega;
        this.u.insertGrid.value = k === 0 ? 1 : 0;
        const pass = this.mesh.C ? [this.clusterPass, this.tetPass] : [this.tetPass];
        if (k === 0) pass.push(this.collidePass);
        if (k === K - 1) pass.push(this.facePass, this.splitPass);
        pass.push(this.projectPass, this.ballApply);
        if (k === K - 1) pass.push(this.contactPass);
        this.renderer.compute(pass);
      }
      this.simTime += dt;
    }
    this.u.frameDt.value = steps * dt;
    if (steps > 0) this.renderer.compute([this.spawnPass, this.particlePass, this.despawnPass]);
    this._poll();
    return true;
  }

  push(origin, dir, strength) {
    this.u.pushOrigin.value.copy(origin);
    this.u.pushDir.value.copy(dir).normalize();
    this.u.pushStrength.value = strength;
    this.renderer.compute(this.pushPass);
  }

  // ---- projectiles ----
  /** Spawns a ball in a GPU slot. kinematic balls follow `update(dt)` (wrecking ball). */
  addBall(opts) {
    const { pos, vel, radius, mass, kinematic = false, life = Infinity } = opts;
    let slot = this.ballSlots.findIndex((b) => !b);
    if (slot < 0) { // recycle the oldest dynamic ball
      slot = this.ballSlots.reduce((best, b, i) => (!b.kinematic && (best < 0 || b.born < this.ballSlots[best].born) ? i : best), -1);
    }
    const ball = { ...opts, slot, kinematic, life, born: performance.now() };
    this.ballSlots[slot] = ball;
    this._writeBall(slot, pos, vel, radius, kinematic ? 0 : 1 / mass, kinematic);
    return ball;
  }

  removeBall(ball) {
    if (this.ballSlots[ball.slot] !== ball) return;
    this.ballSlots[ball.slot] = null;
    this._writeBall(ball.slot, ball.pos, ball.vel, 0, 0, false);
  }

  _writeBall(slot, pos, vel, radius, invMass, kinematic) {
    const dt = this.dt, a = this.balls.value.array, o = slot * 12;
    a.set([pos.x, pos.y, pos.z, radius, pos.x - vel.x * dt, pos.y - vel.y * dt, pos.z - vel.z * dt, invMass, 0, 0, 0, kinematic ? 1 : 0], o);
    this.balls.value.addUpdateRange(o, 12);
    this.balls.value.needsUpdate = true;
  }

  _updateBalls(dt) {
    const now = performance.now();
    for (const b of this.ballSlots) {
      if (!b) continue;
      if (b.kinematic) {
        b.update(dt);
        this.u.kinPos.value.copy(b.pos);
        this.u.kinVel.value.copy(b.vel);
      } else if (now - b.born > b.life) this.removeBall(b);
    }
  }

  // ---- async GPU -> CPU: counters for the HUD (and backpressure) ----
  async _poll() {
    if (this.pendingRead || this.disposed) return;
    this.pendingRead = true;
    try {
      const cnt = await this.renderer.getArrayBufferAsync(this.counters.value);
      if (this.disposed) return;
      const c = new Int32Array(cnt);
      this.brokenCount = c[0];
      this.vertexCount = Math.min(c[1], this.capacity);
    } finally {
      this.pendingRead = false;
      this.backlog = 0;
    }
  }

  dispose() {
    this.disposed = true;
    for (const k of this.kernels) k.dispose?.();
  }
}
