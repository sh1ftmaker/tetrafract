// CPU-side scene construction: rigid pieces (convex polyhedra), their collision
// particles, render triangles, and the breakable weld joints between them.
// Everything is packed into typed arrays for physics.js / render.js.

import { BODY_FLOATS, JOINT_FLOATS, PARTICLE_FLOATS, ADJ, MAX_PLANES, SHAPE_VEC4 } from './physics.js';

// ----------------------------------------------------------------- tiny vector math
export const v3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
  // a + u*x + v*y + w*z
  frame: (o, u, v, w, x, y, z) => [o[0] + u[0] * x + v[0] * y + w[0] * z, o[1] + u[1] * x + v[1] * y + w[1] * z, o[2] + u[2] * x + v[2] * y + w[2] * z],
};
const { add, sub, mul, dot, cross, len, norm } = v3;

/** Orthonormal frame (u along axis, v, w) with v as close as possible to `up`. */
export function frameFrom(axis, up = [0, 1, 0]) {
  const u = norm(axis);
  let v = sub(up, mul(u, dot(up, u)));
  if (len(v) < 1e-4) v = Math.abs(u[0]) < 0.9 ? sub([1, 0, 0], mul(u, u[0])) : sub([0, 0, 1], mul(u, u[2]));
  v = norm(v);
  return [u, v, cross(u, v)];
}

// rotate vector by quaternion [x,y,z,w]
export function qrot(q, v) {
  const t = mul(cross(q, v), 2);
  return add(add(v, mul(t, q[3])), cross(q, t));
}
const qconj = (q) => [-q[0], -q[1], -q[2], q[3]];
export function qmul(a, b) {
  return [
    a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - a[0] * b[2],
    a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - a[1] * b[0],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}
// quaternion from rotation matrix given as columns c0,c1,c2
export function quatFromCols(c0, c1, c2) {
  const m00 = c0[0], m10 = c0[1], m20 = c0[2], m01 = c1[0], m11 = c1[1], m21 = c1[2], m02 = c2[0], m12 = c2[1], m22 = c2[2];
  const tr = m00 + m11 + m22;
  let x, y, z, w;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s; }
  else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s; }
  else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s; }
  const l = Math.hypot(x, y, z, w);
  return [x / l, y / l, z / l, w / l];
}

// symmetric 3x3 eigen decomposition (cyclic Jacobi). Returns { values, vectors: [c0,c1,c2] }
function eigenSym(A) {
  const a = A.map((r) => r.slice());
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-14 * (Math.abs(a[0][0]) + Math.abs(a[1][1]) + Math.abs(a[2][2]) + 1e-30)) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 3; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 3; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const cols = [0, 1, 2].map((j) => [V[0][j], V[1][j], V[2][j]]);
  if (dot(cross(cols[0], cols[1]), cols[2]) < 0) cols[2] = mul(cols[2], -1);
  return { values: [a[0][0], a[1][1], a[2][2]], vectors: cols };
}

/** Mass properties of a closed, outward-oriented triangle mesh (tetrahedra from a reference point). */
function massProps(verts, tris) {
  let o = [0, 0, 0];
  for (const v of verts) o = add(o, v);
  o = mul(o, 1 / verts.length);
  let vol = 0, c = [0, 0, 0];
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const [i, j, k] of tris) {
    const a = sub(verts[i], o), b = sub(verts[j], o), d = sub(verts[k], o);
    const det = dot(a, cross(b, d));
    vol += det / 6;
    c = add(c, mul(add(add(a, b), d), det / 24));
    // covariance of tetra (0,a,b,d): det/120 * (sum_i x_i x_i^T + (sum x)(sum x)^T)
    const s = add(add(a, b), d);
    for (let r = 0; r < 3; r++) for (let q = 0; q < 3; q++) {
      C[r][q] += (det / 120) * (a[r] * a[q] + b[r] * b[q] + d[r] * d[q] + s[r] * s[q]);
    }
  }
  const com = mul(c, 1 / vol);
  // shift covariance to centre of mass
  for (let r = 0; r < 3; r++) for (let q = 0; q < 3; q++) C[r][q] -= vol * com[r] * com[q];
  const tr = C[0][0] + C[1][1] + C[2][2];
  const I = [0, 1, 2].map((r) => [0, 1, 2].map((q) => (r === q ? tr : 0) - C[r][q]));
  return { volume: vol, com: add(o, com), I };
}

// ----------------------------------------------------------------- polyhedra
/** Convex polyhedron description: { verts: [[x,y,z]...], faces: [{ idx:[...ccw outward], flags }] } */
export function boxPoly(center, frame, half, faceFlags = [0, 0, 0, 0, 0, 0]) {
  const [u, v, w] = frame;
  const verts = [];
  for (let k = 0; k < 8; k++) {
    verts.push(v3.frame(center, u, v, w, (k & 1 ? 1 : -1) * half[0], (k & 2 ? 1 : -1) * half[1], (k & 4 ? 1 : -1) * half[2]));
  }
  // faces: -u, +u, -v, +v, -w, +w
  const faces = [
    { idx: [0, 4, 6, 2], flags: faceFlags[0] }, { idx: [1, 3, 7, 5], flags: faceFlags[1] },
    { idx: [0, 1, 5, 4], flags: faceFlags[2] }, { idx: [2, 6, 7, 3], flags: faceFlags[3] },
    { idx: [0, 2, 3, 1], flags: faceFlags[4] }, { idx: [4, 5, 7, 6], flags: faceFlags[5] },
  ];
  return { verts, faces };
}

/** Prism from a convex CCW polygon (2D coords in plane frame u,v), thickness along w. */
export function prismPoly(origin, u, v, w, poly, thick, sideFlags, capFlags = 0) {
  const n = poly.length;
  const verts = [];
  for (const [x, y] of poly) verts.push(v3.frame(origin, u, v, w, x, y, -thick / 2));
  for (const [x, y] of poly) verts.push(v3.frame(origin, u, v, w, x, y, thick / 2));
  const faces = [
    { idx: [...Array(n).keys()].reverse(), flags: capFlags },
    { idx: [...Array(n).keys()].map((i) => i + n), flags: capFlags },
  ];
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    faces.push({ idx: [i, j, j + n, i + n], flags: Array.isArray(sideFlags) ? sideFlags[i] : sideFlags });
  }
  return { verts, faces };
}

/** Hexahedron from 8 corners (index bits: x=1,y=2,z=4), possibly jittered. */
export function hexPoly(c, faceFlags = [0, 0, 0, 0, 0, 0]) {
  const b = boxPoly([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 1, 1], faceFlags);
  return { verts: c.slice(), faces: b.faces };
}

export function spherePoly(center, radius, detail = 2) {
  // icosphere
  const t = (1 + Math.sqrt(5)) / 2;
  let verts = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map(norm);
  let faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  for (let d = 0; d < detail; d++) {
    const cache = new Map(), nf = [];
    const mid = (a, b) => { const k = a < b ? a + '_' + b : b + '_' + a; if (!cache.has(k)) { cache.set(k, verts.length); verts.push(norm(mul(add(verts[a], verts[b]), 0.5))); } return cache.get(k); };
    for (const [a, b, c] of faces) { const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a); nf.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]); }
    faces = nf;
  }
  return { verts: verts.map((p) => add(center, mul(p, radius))), faces: faces.map((idx) => ({ idx, flags: 0, smooth: true })), smoothCenter: center };
}

function triangulate(poly) {
  const tris = [];
  for (const f of poly.faces) for (let i = 1; i + 1 < f.idx.length; i++) tris.push([f.idx[0], f.idx[i], f.idx[i + 1], f]);
  return tris;
}

// ----------------------------------------------------------------- builder
export const MAT = { WOOD: 0, GLASS: 1, STONE: 2, METAL: 3 };
export const FLAG = { INTERIOR: 1, RIM: 2 };

export class WorldBuilder {
  constructor({ radius = 0.05, groundY = 0 } = {}) {
    this.r = radius;
    this.groundY = groundY;
    this.bodies = [];
    this.joints = [];
    this.statics = [];
    this.rand = mulberry32(1234);
  }

  /**
   * Adds a rigid piece.
   * poly: convex polyhedron in world space. opts: { density, material, static, tex: fn(worldPos)->[x,y,z],
   *  sampleFrame: [u,v,w] lattice orientation for collision particles, tint }
   */
  addBody(poly, opts = {}) {
    const rec = this.computeBody(poly, opts);
    rec.id = this.bodies.length;
    rec.tint = opts.tint ?? this.rand();
    this.bodies.push(rec);
    return rec.id;
  }

  /** Mass properties, principal frame, probes, hull planes and render triangles of a piece. */
  computeBody(poly, opts = {}) {
    const tris = triangulate(poly);
    const mp = massProps(poly.verts, tris);
    const density = opts.density ?? 500;
    const mass = mp.volume * density;
    const { values, vectors } = eigenSym(mp.I);
    const q = quatFromCols(vectors[0], vectors[1], vectors[2]);
    const qi = qconj(q);
    const toLocal = (p) => qrot(qi, sub(p, mp.com));
    const isStatic = !!opts.static;

    // collision particles: lattice (spacing 2r) in the sample frame, kept if inside the shape
    const r = this.r;
    const sf = opts.sampleFrame ?? vectors;
    const planes = poly.faces.map((f) => {
      const a = poly.verts[f.idx[0]], b = poly.verts[f.idx[1]], c = poly.verts[f.idx[2]];
      let n = norm(cross(sub(b, a), sub(c, a)));
      return [n, dot(n, a)];
    });
    const sd = (p) => Math.max(...planes.map(([n, d]) => dot(n, p) - d));
    const ext = sf.map((ax) => { let lo = Infinity, hi = -Infinity; for (const p of poly.verts) { const d = dot(ax, p); lo = Math.min(lo, d); hi = Math.max(hi, d); } return [lo, hi]; });
    const minExt = Math.min(...ext.map(([lo, hi]) => hi - lo));
    const inset = Math.min(0.5 * r, 0.45 * minExt);
    const n = ext.map(([lo, hi]) => Math.max(1, Math.round((hi - lo) / (2 * r))));
    const parts = [];
    for (let i = 0; i < n[0]; i++) for (let j = 0; j < n[1]; j++) for (let k = 0; k < n[2]; k++) {
      const c = [i, j, k].map((t, ax) => ext[ax][0] + (t + 0.5) * (ext[ax][1] - ext[ax][0]) / n[ax]);
      const p = add(add(mul(sf[0], c[0]), mul(sf[1], c[1])), mul(sf[2], c[2]));
      // probes are only needed near the surface (interior is covered by the exact hull test)
      if (sd(p) < -(opts.shell ?? 2.2 * r)) continue;
      if (sd(p) <= -inset + 1e-6) parts.push([...toLocal(p), Math.min(r, -sd(p))]);
    }
    if (!parts.length) parts.push([0, 0, 0, Math.max(0.002, Math.min(r, -sd(mp.com)))]);

    // render triangles (local frame) with flat normals unless the face is smooth
    const tex = opts.tex ?? ((p) => p);
    const rv = [];
    if (opts.renderTris) {
      // custom render mesh (e.g. splintered cut faces); collision still uses the hull
      for (const [pa, pb, pc, flags] of opts.renderTris) {
        const fn = norm(cross(sub(pb, pa), sub(pc, pa)));
        for (const p of [pa, pb, pc]) rv.push({ p: toLocal(p), n: qrot(qi, fn), t: tex(p), flags });
      }
    } else for (const [a, b, c, f] of tris) {
      const pa = poly.verts[a], pb = poly.verts[b], pc = poly.verts[c];
      const fn = norm(cross(sub(pb, pa), sub(pc, pa)));
      for (const p of [pa, pb, pc]) {
        const nrm = f.smooth ? norm(sub(p, poly.smoothCenter)) : fn;
        rv.push({ p: toLocal(p), n: qrot(qi, nrm), t: tex(p), flags: f.flags });
      }
    }
    // exact collision shape in the body frame: sphere or convex planes (n·x <= d)
    let shape;
    if (poly.smoothCenter) shape = { type: 1, radius: len(sub(poly.verts[0], poly.smoothCenter)), planes: [] };
    else {
      const pl = [];
      const addPlane = (a, b, c) => {
        const la = toLocal(a), lb = toLocal(b), lc = toLocal(c);
        const n = norm(cross(sub(lb, la), sub(lc, la)));
        if (!pl.some((q) => dot(q, n) > 0.99999 && Math.abs(q[3] - dot(n, la)) < 1e-5)) pl.push([...n, dot(n, la)]);
      };
      for (const f of poly.faces) {
        const v = f.idx.map((i) => poly.verts[i]);
        const n0 = norm(cross(sub(v[1], v[0]), sub(v[2], v[0])));
        const planar = v.every((p) => Math.abs(dot(n0, sub(p, v[0]))) < 1e-5);
        if (planar) addPlane(v[0], v[1], v[2]);
        else for (let i = 1; i + 1 < v.length; i++) addPlane(v[0], v[i], v[i + 1]);
      }
      if (pl.length > MAX_PLANES) console.warn('shape has', pl.length, 'planes; truncating');
      shape = { type: 0, planes: pl.slice(0, MAX_PLANES) };
    }
    return {
      shape, pos: mp.com, quat: q, mass, invMass: isStatic ? 0 : 1 / mass,
      iinv: isStatic ? [0, 0, 0] : values.map((x) => 1 / Math.max(x * density, 1e-9)),
      particles: parts, verts: rv, material: opts.material ?? MAT.WOOD, toLocal,
    };
  }

  /**
   * Weld joint. axis: world direction used to split the load into normal / shear and
   * twist / bending. Strengths in N and N·m. Compliance in m/N and rad/(N·m).
   */
  addJoint(a, b, anchor, axis, s) {
    if (a === b) return -1;
    const A = this.bodies[a], B = this.bodies[b];
    if (A.invMass === 0 && B.invMass === 0 && !A.realInvMass && !B.realInvMass) return -1;
    const qi = qconj(A.quat);
    this.joints.push({
      a, b, anchor, axisW: axis,
      la: A.toLocal(anchor), lb: B.toLocal(anchor),
      q0: qmul(qconj(A.quat), B.quat),
      axis: norm(qrot(qi, axis)),
      compLin: s.compLin ?? 0, compAng: s.compAng ?? 0,
      tension: s.tension, compression: s.compression ?? s.tension * 4,
      shear: s.shear, bend: s.bend, torsion: s.torsion ?? s.bend,
      plastic: !!s.plastic, plasticLimit: s.plasticLimit ?? 0.6, brittle: !!s.brittle,
    });
    return this.joints[this.joints.length - 1];
  }

  addStatic(center, half, quat = [0, 0, 0, 1], opts = {}) {
    this.statics.push({ center, half, quat, friction: opts.friction ?? 1, color: opts.color ?? 0x8a8478, visible: opts.visible ?? true });
  }

  /**
   * Welds every pair of bodies from listA x listB that touch (particles within
   * 2r + tol). The anchor is the mean of the touching particle midpoints.
   */
  connectTouching(listA, listB, strength, tol = 0.25) {
    const r = this.r, lim = (2 * r * (1 + tol)) ** 2;
    const world = (b) => b._w ?? (b._w = b.particles.map((p) => add(qrot(b.quat, p), b.pos)));
    const bound = (b) => {
      if (b._bb) return b._bb;
      const w = world(b); const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
      for (const p of w) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k] - r * 1.5); hi[k] = Math.max(hi[k], p[k] + r * 1.5); }
      return (b._bb = [lo, hi]);
    };
    let count = 0;
    const setB = new Set(listB);
    for (const ia of listA) {
      const A = this.bodies[ia], [la, ha] = bound(A);
      for (const ib of setB) {
        if (ia === ib) continue;
        const B = this.bodies[ib], [lb, hb] = bound(B);
        if (la[0] > hb[0] || lb[0] > ha[0] || la[1] > hb[1] || lb[1] > ha[1] || la[2] > hb[2] || lb[2] > ha[2]) continue;
        let m = [0, 0, 0], k = 0;
        for (const p of world(A)) for (const q of world(B)) {
          const d = sub(p, q);
          if (dot(d, d) < lim) { m = add(m, mul(add(p, q), 0.5)); k++; }
        }
        if (!k) continue;
        m = mul(m, 1 / k);
        const s = typeof strength === 'function' ? strength(A, B, k) : strength;
        if (!s) continue;
        this.addJoint(ia, ib, m, norm(sub(B.pos, A.pos)), s);
        count++;
      }
    }
    return count;
  }

  /**
   * Calls fn(a, b) for every pair where a probe particle of body a lies inside the
   * contact range of body b's exact shape in the initial configuration.
   */
  findSoftPairs(fn) {
    const r = this.r, cs = 2 * r;
    const grid = new Map();
    const key = (x, y, z) => x * 73856093 ^ y * 19349663 ^ z * 83492791;
    const pts = [];
    for (const b of this.bodies) for (const lp of b.particles) {
      const w = add(qrot(b.quat, lp), b.pos);
      const i = pts.length;
      pts.push([w, b, lp[3]]);
      const k = key(Math.floor(w[0] / cs), Math.floor(w[1] / cs), Math.floor(w[2] / cs));
      if (!grid.has(k)) grid.set(k, []);
      grid.get(k).push(i);
    }
    const sdOf = (b, p) => {
      const l = qrot(qconj(b.quat), sub(p, b.pos));
      if (b.shape.type === 1) return len(l) - b.shape.radius;
      let m = -1e9;
      for (const pl of b.shape.planes) m = Math.max(m, pl[0] * l[0] + pl[1] * l[1] + pl[2] * l[2] - pl[3]);
      return m;
    };
    const seen = new Set();
    const nB = this.bodies.length;
    for (const [w, A, ri] of pts) {
      if (A.invMass === 0) continue;
      const cx = Math.floor(w[0] / cs), cy = Math.floor(w[1] / cs), cz = Math.floor(w[2] / cs);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const list = grid.get(key(cx + dx, cy + dy, cz + dz));
        if (!list) continue;
        for (const j of list) {
          const B = pts[j][1];
          if (B === A || seen.has(A.id * nB + B.id)) continue;
          if (sdOf(B, w) < ri + 0.002) { seen.add(A.id * nB + B.id); fn(A.id, B.id); }
        }
      }
    }
  }

  /** Packs everything into GPU-ready arrays. */
  finish() {
    const nB = this.bodies.length;
    // adjacency slots for "welded bodies don't collide"
    const SV = SHAPE_VEC4 * 4;
    const shapeU = new Uint32Array(nB * SV);
    const shapeF = new Float32Array(shapeU.buffer);
    for (const b of this.bodies) {
      const o = b.id * SV;
      shapeU.fill(0xffffffff, o, o + ADJ);
      packHull(b, shapeU, shapeF, o + ADJ);
    }
    const fill = new Uint8Array(nB);
    const slotOf = (b, other) => {
      if (fill[b] >= ADJ) return 0xffff;
      const s = fill[b]++;
      shapeU[b * SV + s] = other;
      return s;
    };
    // greedy graph colouring: joints sharing a dynamic body get different colours
    const used = this.bodies.map(() => new Set());
    for (const j of this.joints) {
      const dynA = this.bodies[j.a].invMass > 0, dynB = this.bodies[j.b].invMass > 0;
      let c = 0;
      while ((dynA && used[j.a].has(c)) || (dynB && used[j.b].has(c))) c++;
      j.color = c;
      if (dynA) used[j.a].add(c);
      if (dynB) used[j.b].add(c);
    }
    const joints = this.joints.slice().sort((x, y) => x.color - y.color);
    const adjSet = new Set();
    for (const j of joints) { adjSet.add(j.a * nB + j.b); adjSet.add(j.b * nB + j.a); }
    const nColors = joints.length ? joints[joints.length - 1].color + 1 : 0;
    const colorRanges = [];
    for (let c = 0, i = 0; c < nColors; c++) {
      const s = i; while (i < joints.length && joints[i].color === c) i++;
      colorRanges.push([s, i - s]);
    }

    const jb = new ArrayBuffer(Math.max(1, joints.length) * JOINT_FLOATS * 4);
    const jf = new Float32Array(jb), ju = new Uint32Array(jb);
    const slots = joints.map((j) => [slotOf(j.a, j.b), slotOf(j.b, j.a)]);
    // soft adjacency: as-built touching pairs that are not welded (see contacts kernel)
    this.findSoftPairs((a, b) => {
      if (fill[a] < ADJ && !adjSet.has(a * nB + b)) { shapeU[a * SV + fill[a]++] = (b | 0x80000000) >>> 0; adjSet.add(a * nB + b); }
    });
    joints.forEach((j, i) => {
      j.index = i;
      j.slots = slots[i];
      packJoint(j, jf, ju, i * JOINT_FLOATS);
    });

    let nP = 0;
    for (const b of this.bodies) nP += b.particles.length;
    const pb = new ArrayBuffer(nP * PARTICLE_FLOATS * 4);
    const pf = new Float32Array(pb), pu = new Uint32Array(pb);
    const bb = new ArrayBuffer(nB * BODY_FLOATS * 4);
    const bf = new Float32Array(bb), bu = new Uint32Array(bb);
    let p = 0;
    for (const b of this.bodies) {
      b.firstParticle = p;
      packBody(b, bf, bu, b.id * BODY_FLOATS, p, b.particles.length);
      packParticles(b, pf, pu, p * PARTICLE_FLOATS);
      p += b.particles.length;
    }
    return {
      radius: this.r, groundY: this.groundY,
      nBodies: nB, nParticles: nP, nJoints: joints.length,
      bodies: bf, particles: pf, joints: jf, shapes: shapeU, colorRanges, statics: this.statics,
      bodyList: this.bodies,
    };
  }
}

// ----------------------------------------------------------------- GPU record packing
export function packBody(b, bf, bu, o, pstart, pcount) {
  bf.set([...b.pos, b.invMass, ...b.quat, ...b.pos, b.invMass, ...b.quat, 0, 0, 0, 0, 0, 0, 0, 0, ...b.iinv, 0], o);
  bu.set([pstart, pcount, b.material, 0], o + 28);
}
export function packParticles(b, pf, pu, o) {
  for (const lp of b.particles) {
    const l3 = [lp[0], lp[1], lp[2]];
    pf.set(l3, o); pu[o + 3] = b.id;
    const wp = add(qrot(b.quat, l3), b.pos);
    pf.set(wp, o + 4); pu[o + 7] = b.id;
    pf.set(wp, o + 8); pf[o + 11] = lp[3];
    pf.fill(0, o + 12, o + 20);
    o += PARTICLE_FLOATS;
  }
}
/** header (type, plane count, radius) + planes, at u32 offset o */
export function packHull(b, shapeU, shapeF, o) {
  shapeU[o] = b.shape.type; shapeU[o + 1] = b.shape.planes.length;
  shapeF[o + 2] = b.shape.radius ?? 0; shapeU[o + 3] = 0;
  shapeF.fill(0, o + 4, o + 4 + MAX_PLANES * 4);
  b.shape.planes.forEach((p, k) => shapeF.set(p, o + 4 + k * 4));
}
export function packJoint(j, jf, ju, o) {
  const [sa, sb] = j.slots;
  ju.set([j.a, j.b, (sa & 0xffff) | ((sb & 0xffff) << 16), (j.plastic ? 2 : 0) | (j.brittle ? 4 : 0)], o);
  jf.set([...j.la, j.compLin, ...j.lb, j.compAng, ...j.q0, ...j.axis, j.compression, j.tension, j.shear, j.bend, j.torsion, 0, 0, 0, 0, 0, 0, 0, j.plasticLimit], o + 4);
}

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
