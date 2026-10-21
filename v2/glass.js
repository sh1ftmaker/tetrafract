// Glass panes with an impact-centred fracture pattern.
//
// A pane is pre-scored into triangular shards following a radial + concentric
// pattern (rays from the focus to fixed points on the border, rings that get
// coarser away from the focus). Shards are welded by brittle joints.
//
// The pattern topology (which shard touches which) never changes, only the
// geometry does. So when the CPU predicts at fire time that a projectile will hit
// an intact pane, it re-cuts that pane around the predicted impact point and
// patches the GPU buffers in place (bodies, probes, hull planes, joint frames,
// render vertices) before the projectile is launched. Real glass cracks radiate
// from wherever it was hit; so does this.

import { v3, prismPoly, MAT, FLAG, packBody, packParticles, packHull, packJoint } from './builder.js';
import { BODY_FLOATS, PARTICLE_FLOATS, JOINT_FLOATS, SHAPE_VEC4, ADJ } from './physics.js';

const { add, sub, mul, dot, cross, norm } = v3;
export const GLASS = { density: 2500, tensile: 7e6, bending: 7e6 };

export class GlassPane {
  constructor(W, center, u, v, width, height, thick, o = {}) {
    const rand = o.rand ?? W.rand;
    Object.assign(this, { W, center, u, v, n: norm(cross(u, v)), width, height, thick, strength: o.strength ?? 1 });
    const hw = width / 2, hh = height / 2;
    // fixed border points (CCW from bottom-left corner), corners included
    const S = o.sectors ?? 16;
    const per = 2 * (width + height);
    const edges = [[[-hw, -hh], [hw, -hh]], [[hw, -hh], [hw, hh]], [[hw, hh], [-hw, hh]], [[-hw, hh], [-hw, -hh]]];
    this.border = [];
    for (const [a, b] of edges) {
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const n = Math.max(2, Math.round((S * L) / per));
      for (let i = 0; i < n; i++) {
        const t = i === 0 ? 0 : (i + (rand() - 0.5) * 0.5) / n;
        this.border.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
      }
    }
    this.S = this.border.length;
    this.rings = o.rings ?? 7;
    this.fr = [0];
    for (let j = 1; j <= this.rings; j++) this.fr.push(Math.pow(j / this.rings, 1.65));
    this.jit = [], this.diag = [];
    for (let j = 0; j <= this.rings; j++) {
      this.jit.push([...Array(this.S)].map(() => rand() * 2 - 1));
      this.diag.push([...Array(this.S)].map(() => rand() < 0.5));
    }
    this.focus = this.clampFocus(o.focus ?? [0, 0]);
    this.tex = (p) => { const d = sub(p, center); return [dot(d, u), dot(d, v), dot(d, this.n)]; };
    this.build();
  }

  clampFocus([x, y]) {
    const mx = this.width * 0.36, my = this.height * 0.36;
    return [Math.max(-mx, Math.min(mx, x)), Math.max(-my, Math.min(my, y))];
  }

  /** Triangular cells for the current focus. Vertex ids: -1 = focus, j*S+k = ring j, ray k. */
  cells() {
    const { S, rings, fr, jit, diag, border, focus: f } = this;
    const P = (j, k) => {
      if (j === 0) return f;
      const b = border[k % S];
      if (j === rings) return b;
      const t = fr[j] * (1 + jit[j][k % S] * 0.18 * (fr[j] - fr[j - 1]) / fr[j]);
      return [f[0] + (b[0] - f[0]) * t, f[1] + (b[1] - f[1]) * t];
    };
    const id = (j, k) => (j === 0 ? -1 : j * S + (k % S));
    const out = [];
    for (let j = 0; j < rings; j++) for (let k = 0; k < S; k++) {
      const k1 = k + 1;
      if (j === 0) { out.push([[P(1, k), id(1, k)], [P(1, k1), id(1, k1)], [P(0, 0), -1]]); continue; }
      const q = [[P(j, k), id(j, k)], [P(j + 1, k), id(j + 1, k)], [P(j + 1, k1), id(j + 1, k1)], [P(j, k1), id(j, k1)]];
      if (diag[j][k]) out.push([q[0], q[1], q[2]], [q[0], q[2], q[3]]);
      else out.push([q[0], q[1], q[3]], [q[1], q[2], q[3]]);
    }
    // CCW orientation
    return out.map((c) => {
      const [a, b, d] = c.map((x) => x[0]);
      const area = (b[0] - a[0]) * (d[1] - a[1]) - (d[0] - a[0]) * (b[1] - a[1]);
      return area < 0 ? [c[0], c[2], c[1]] : c;
    });
  }

  isBorderEdge(ia, ib) {
    const R = this.rings * this.S;
    return ia >= R && ib >= R;
  }

  bodyOpts() { return { density: GLASS.density, material: MAT.GLASS, tex: this.tex, sampleFrame: [this.u, this.v, this.n] }; }

  polyOf(cell) {
    const pts = cell.map((x) => x[0]);
    const flags = cell.map((x, i) => (this.isBorderEdge(x[1], cell[(i + 1) % 3][1]) ? FLAG.RIM : FLAG.RIM | FLAG.INTERIOR));
    return prismPoly(this.center, this.u, this.v, this.n, pts, this.thick, flags);
  }

  /** Internal shared edges → joint geometry (anchor, in-plane edge normal, strengths). */
  edgeJoint(pa, pb) {
    const el = Math.hypot(pb[0] - pa[0], pb[1] - pa[1]);
    const anchor = v3.frame(this.center, this.u, this.v, this.n, (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, 0);
    const axis = norm(v3.frame([0, 0, 0], this.u, this.v, this.n, pb[1] - pa[1], -(pb[0] - pa[0]), 0));
    // tiny shards near the focus still get a sensible minimum bond (noise robustness)
    const le = Math.max(el, 0.06);
    const A = le * this.thick, s = this.strength;
    const M = GLASS.bending * le * this.thick * this.thick / 6 * s * 1.5;
    return { anchor, axis, s: { tension: GLASS.tensile * A * s, compression: GLASS.tensile * A * 20 * s, shear: GLASS.tensile * A * s, bend: M, torsion: M, brittle: true } };
  }

  build() {
    const W = this.W;
    const cells = this.cells();
    this.ids = cells.map((c) => W.addBody(this.polyOf(c), this.bodyOpts()));
    // spare probe slots for re-cutting: a parked dummy body owns them
    let np = 0;
    for (const id of this.ids) np += W.bodies[id].particles.length;
    this.dummy = W.addBody(prismPoly([0, -100, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [[0, 0], [0.01, 0], [0, 0.01]], 0.01, 0), { static: true, material: MAT.GLASS });
    const D = W.bodies[this.dummy];
    D.particles = [...Array(Math.ceil(np * 0.7) + 24)].map(() => [0, 0, 0, 0.001]);
    D.verts = [];
    // internal joints (fixed topology)
    const owner = new Map();
    this.internal = [];
    cells.forEach((c, ci) => {
      for (let i = 0; i < 3; i++) {
        const a = c[i], b = c[(i + 1) % 3];
        const key = Math.min(a[1], b[1]) + ',' + Math.max(a[1], b[1]);
        const other = owner.get(key);
        if (other === undefined) { owner.set(key, ci); continue; }
        const g = this.edgeJoint(a[0], b[0]);
        const j = W.addJoint(this.ids[other], this.ids[ci], g.anchor, g.axis, g.s);
        this.internal.push({ j, ca: other, cb: ci, key });
      }
    });
    // shards on each side of the pane (for seals)
    const hw = this.width / 2, hh = this.height / 2, e = 1e-6;
    this.sides = { left: [], right: [], bottom: [], top: [] };
    cells.forEach((c, ci) => {
      for (let i = 0; i < 3; i++) {
        const a = c[i][0], b = c[(i + 1) % 3][0];
        if (!this.isBorderEdge(c[i][1], c[(i + 1) % 3][1])) continue;
        if (Math.abs(a[0] + hw) < e && Math.abs(b[0] + hw) < e) this.sides.left.push(this.ids[ci]);
        if (Math.abs(a[0] - hw) < e && Math.abs(b[0] - hw) < e) this.sides.right.push(this.ids[ci]);
        if (Math.abs(a[1] + hh) < e && Math.abs(b[1] + hh) < e) this.sides.bottom.push(this.ids[ci]);
        if (Math.abs(a[1] - hh) < e && Math.abs(b[1] - hh) < e) this.sides.top.push(this.ids[ci]);
      }
    });
    this.bodies = this.ids;
    this.intact = true;
  }

  /** After the world is finished: remember joints that attach the pane to other things. */
  collectExternal() {
    const mine = new Set(this.ids);
    const internal = new Set(this.internal.map((x) => x.j));
    this.external = this.W.joints.filter((j) => !internal.has(j) && (mine.has(j.a) || mine.has(j.b)));
  }

  /** Predicts where a ballistic projectile path crosses this pane (pane coords) or null. */
  hitPoint(origin, vel, g = 9.81) {
    const n = this.n;
    let p0 = origin, d0 = dot(sub(p0, this.center), n);
    for (let t = 0.01; t < 3; t += 0.01) {
      const p1 = [origin[0] + vel[0] * t, origin[1] + vel[1] * t - 0.5 * g * t * t, origin[2] + vel[2] * t];
      const d1 = dot(sub(p1, this.center), n);
      if ((d0 > 0) !== (d1 > 0)) {
        const s = d0 / (d0 - d1);
        const p = v3.lerp(p0, p1, s);
        const x = dot(sub(p, this.center), this.u), y = dot(sub(p, this.center), this.v);
        if (Math.abs(x) <= this.width / 2 && Math.abs(y) <= this.height / 2) return { x, y, t };
        return null;
      }
      p0 = p1; d0 = d1;
    }
    return null;
  }

  /**
   * Re-cuts the pane around `focus` and patches all GPU buffers in place.
   * Only valid while the pane is untouched (caller verifies with a readback).
   */
  refracture(focus, sim, data, meshes) {
    const W = this.W;
    this.focus = this.clampFocus(focus);
    const cells = this.cells();
    const dev = sim.device;
    // 1. new shard bodies
    const first = W.bodies[this.ids[0]].firstParticle;
    const D = W.bodies[this.dummy];
    let cap = D.firstParticle + D.particles.length - first;
    cells.forEach((c, ci) => {
      const id = this.ids[ci], old = W.bodies[id];
      const rec = W.computeBody(this.polyOf(c), this.bodyOpts());
      Object.assign(old, rec);
    });
    // 2. probe allocation inside the pane's contiguous range (dummy takes the rest)
    let need = 0;
    for (const id of this.ids) need += W.bodies[id].particles.length;
    while (need > cap - 1) { // trim the most densely sampled shard
      let best = null;
      for (const id of this.ids) { const b = W.bodies[id]; if (!best || b.particles.length > best.particles.length) best = b; }
      best.particles.pop(); need--;
    }
    const nb = this.dummy - this.ids[0] + 1; // ids are contiguous: shards then dummy
    const bf = new Float32Array(nb * BODY_FLOATS), bu = new Uint32Array(bf.buffer);
    const pf = new Float32Array(cap * PARTICLE_FLOATS), pu = new Uint32Array(pf.buffer);
    let p = first;
    for (let k = 0; k < nb; k++) {
      const b = W.bodies[this.ids[0] + k];
      if (b === D) D.particles = [...Array(cap - (p - first))].map(() => [0, 0, 0, 0.001]);
      b.firstParticle = p;
      packBody(b, bf, bu, k * BODY_FLOATS, p, b.particles.length);
      packParticles(b, pf, pu, (p - first) * PARTICLE_FLOATS);
      p += b.particles.length;
    }
    dev.queue.writeBuffer(sim.bodyBuf, this.ids[0] * BODY_FLOATS * 4, bf);
    dev.queue.writeBuffer(sim.particleBuf, first * PARTICLE_FLOATS * 4, pf);
    // 3. hull planes (adjacency part of the shape record is unchanged)
    const SV = SHAPE_VEC4 * 4;
    const su = data.shapes, sf = new Float32Array(su.buffer);
    for (const id of this.ids) packHull(W.bodies[id], su, sf, id * SV + ADJ);
    dev.queue.writeBuffer(sim.shapeBuf, this.ids[0] * SV * 4, su.buffer, su.byteOffset + this.ids[0] * SV * 4, nb * SV * 4);
    // 4. joint frames
    const jf = new Float32Array(JOINT_FLOATS), ju = new Uint32Array(jf.buffer);
    const upd = (j, anchor, axis, s) => {
      const A = W.bodies[j.a], B = W.bodies[j.b];
      j.la = A.toLocal(anchor); j.lb = B.toLocal(anchor);
      j.q0 = qmulc(A.quat, B.quat);
      j.axis = norm(qrotc(A.quat, axis));
      if (s) Object.assign(j, { tension: s.tension, compression: s.compression, shear: s.shear, bend: s.bend, torsion: s.torsion });
      packJoint(j, jf, ju, 0);
      dev.queue.writeBuffer(sim.jointBuf, j.index * JOINT_FLOATS * 4, jf);
    };
    const byKey = new Map();
    cells.forEach((c, ci) => { for (let i = 0; i < 3; i++) { const a = c[i], b = c[(i + 1) % 3]; byKey.set(Math.min(a[1], b[1]) + ',' + Math.max(a[1], b[1]), [a[0], b[0]]); } });
    for (const { j, key } of this.internal) {
      const [pa, pb] = byKey.get(key);
      const g = this.edgeJoint(pa, pb);
      upd(j, g.anchor, g.axis, g.s);
    }
    for (const j of this.external) upd(j, j.anchor, j.axisW);
    // 5. render vertices
    const mesh = meshes.get(MAT.GLASS);
    if (mesh) {
      const g = mesh.geometry;
      const P = g.attributes.position, N = g.attributes.normal, T = g.attributes.tex;
      for (const id of this.ids) {
        const b = W.bodies[id];
        b.verts.forEach((vx, i) => {
          const k = b.vstart + i;
          P.array.set(vx.p, k * 3); N.array.set(vx.n, k * 3); T.array.set([...vx.t, vx.flags], k * 4);
        });
      }
      P.needsUpdate = N.needsUpdate = T.needsUpdate = true;
    }
  }
}

// small quaternion helpers (avoid import cycles)
function qrotc(q, v) { // rotate v by conj(q)
  const c = [-q[0], -q[1], -q[2], q[3]];
  const t = mul(cross(c, v), 2);
  return add(add(v, mul(t, c[3])), cross(c, t));
}
function qmulc(a, b) { // conj(a) * b
  const x = -a[0], y = -a[1], z = -a[2], w = a[3];
  return [w * b[0] + b[3] * x + y * b[2] - z * b[1], w * b[1] + b[3] * y + z * b[0] - x * b[2], w * b[2] + b[3] * z + x * b[1] - y * b[0], w * b[3] - x * b[0] - y * b[1] - z * b[2]];
}
