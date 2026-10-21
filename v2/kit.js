// Structural "kits": wood members made of grain-aligned fibres, glass panes
// pre-scored with a radial/concentric fracture pattern, jittered stone blocks,
// spheres. All build rigid pieces + breakable welds through WorldBuilder.

import { v3, frameFrom, boxPoly, hexPoly, spherePoly, MAT, FLAG } from './builder.js';

const { add, sub, mul, dot, cross, len, norm } = v3;

// --------------------------------------------------------------------- materials
// Strengths are real-ish orders of magnitude (Pa); `scale` lets scenes make things
// more or less fragile. Wood is strongly anisotropic: ~20x weaker across the grain.
export const WOOD = {
  density: 520,
  tensile: 45e6,     // along grain
  compressive: 35e6, // along grain
  bending: 40e6,
  perp: 1.0e6,       // tension perpendicular to grain (splitting)
  shear: 2.5e6,      // shear parallel to grain
};
export const STONE = { density: 2300, tensile: 0.5e6, shear: 1.0e6, bending: 0.5e6 };

/**
 * A rectangular timber from p0 to p1, built as nw x nh fibres, each cut into
 * segments of length ~seg with staggered joints (like real grain, so cracks run
 * along the fibres and leave long splinters).
 * Returns { bodies, fibres: [[ids...]], frame, p0, length }.
 */
export function woodMember(W, p0, p1, width, height, o = {}) {
  const rand = o.rand ?? W.rand;
  const [u, v, w] = frameFrom(sub(p1, p0), o.up ?? [0, 1, 0]);
  const L = len(sub(p1, p0));
  const fs = o.fibre ?? 0.1;
  const nh = o.nh ?? Math.max(1, Math.round(height / fs));
  const nw = o.nw ?? Math.max(1, Math.round(width / fs));
  const seg = o.seg ?? 0.6;
  const s = (o.strength ?? 1);
  const mat = o.wood ?? WOOD;
  const fh = height / nh, fw = width / nw;
  const texOff = [rand() * 50, rand() * 2 - 1, rand() * 2 - 1];
  const tint = o.tint ?? rand();
  const tex = (p) => { const d = sub(p, p0); return [dot(d, u) + texOff[0], dot(d, v) + texOff[1], dot(d, w) + texOff[2]]; };
  const fibres = [];
  const segs = []; // per fibre: [{id, s0, s1}]
  const N = o.capRes ?? 3;
  // Splintered cut surface shared by the two segments meeting at a cut: a random
  // spiky height field along the grain. Intact, the two halves interlock seamlessly;
  // once the joint breaks the ends look like torn, splintered fibres.
  const cutField = (amp) => {
    const f = [];
    for (let i = 0; i <= N; i++) for (let j = 0; j <= N; j++) {
      const r = rand() * 2 - 1;
      f.push(amp * Math.sign(r) * Math.pow(Math.abs(r), 0.6));
    }
    return f;
  };
  for (let iv = 0; iv < nh; iv++) for (let iw = 0; iw < nw; iw++) {
    // staggered cut positions
    const cuts = [0];
    let x = seg * (0.35 + 0.65 * rand());
    while (x < L - seg * 0.4) { cuts.push(x); x += seg * (0.75 + 0.5 * rand()); }
    cuts.push(L);
    const list = [];
    const cy = (iv + 0.5) * fh - height / 2, cz = (iw + 0.5) * fw - width / 2;
    // displacement fields of every cut (member ends stay flat)
    const fields = cuts.map((c, k) => {
      if (k === 0 || k === cuts.length - 1 || o.flatCuts) return null;
      const amp = Math.min((o.splinter ?? 1.3) * Math.max(fh, fw), 0.3 * (c - cuts[k - 1]), 0.3 * (cuts[k + 1] - c));
      return cutField(amp);
    });
    for (let k = 0; k + 1 < cuts.length; k++) {
      const s0 = cuts[k], s1 = cuts[k + 1];
      const c = v3.frame(p0, u, v, w, (s0 + s1) / 2, cy, cz);
      const I = FLAG.INTERIOR;
      const flags = [s0 > 0 ? I : 0, s1 < L ? I : 0, iv > 0 ? I : 0, iv < nh - 1 ? I : 0, iw > 0 ? I : 0, iw < nw - 1 ? I : 0];
      const renderTris = segmentMesh(p0, u, v, w, s0, s1, cy, cz, fh, fw, N, fields[k], fields[k + 1], flags);
      const id = W.addBody(boxPoly(c, [u, v, w], [(s1 - s0) / 2, fh / 2, fw / 2], flags), {
        density: mat.density, material: o.material ?? MAT.WOOD, tex, tint, sampleFrame: [u, v, w], static: o.static, renderTris,
      });
      list.push({ id, s0, s1, cy, cz });
    }
    segs.push({ iv, iw, list });
    fibres.push(list.map((x) => x.id));
  }
  const A = fh * fw;
  // along-grain joints between consecutive segments of a fibre
  for (const { list } of segs) {
    for (let k = 0; k + 1 < list.length; k++) {
      const a = list[k], b = list[k + 1];
      const q = (o.plastic ? 1 : 0.6 + 0.8 * rand()) * s; // natural flaws: ±40 % strength scatter
      W.addJoint(a.id, b.id, v3.frame(p0, u, v, w, a.s1, a.cy, a.cz), u, {
        tension: mat.tensile * A * q, compression: mat.compressive * A * s * 3,
        shear: mat.shear * A * q * 2, bend: mat.bending * fw * fh * Math.min(fw, fh) / 6 * q,
        torsion: mat.shear * A * Math.min(fw, fh) / 3 * q,
        // wood yields a little (fibres crushing on the compression side) before it snaps
        compAng: o.compAng ?? 0, plastic: o.plastic ?? true, plasticLimit: o.plasticLimit ?? (o.plastic ? 1.2 : 0.07),
      });
    }
  }
  // cross-grain joints between neighbouring fibres wherever their segments overlap
  const byIdx = new Map(segs.map((f) => [f.iv * 1000 + f.iw, f]));
  for (const f of segs) {
    for (const [dv, dw] of [[1, 0], [0, 1]]) {
      const g = byIdx.get((f.iv + dv) * 1000 + f.iw + dw);
      if (!g) continue;
      const contactW = dv ? fw : fh;
      for (const a of f.list) for (const b of g.list) {
        const lo = Math.max(a.s0, b.s0), hi = Math.min(a.s1, b.s1);
        const ov = hi - lo;
        if (ov < 0.05) continue;
        // effective overlap: short slivers next to a staggered cut still bond over
        // a reasonable length of grain (keeps strengths uniform along the member)
        const ovE = Math.max(ov, 0.4 * seg);
        const area = ovE * contactW * (o.plastic ? 1 : 0.6 + 0.8 * rand()) * (o.lateral ?? 1);
        const anchor = v3.frame(p0, u, v, w, (lo + hi) / 2, (a.cy + b.cy) / 2, (a.cz + b.cz) / 2);
        W.addJoint(a.id, b.id, anchor, dv ? v : w, {
          tension: mat.perp * area * s, compression: mat.compressive * area * s,
          shear: mat.shear * area * s, bend: mat.perp * area * Math.max(ovE, contactW) / 3 * s,
          torsion: mat.shear * area * Math.max(ovE, contactW) / 3 * s,
          compAng: o.compAng ?? 0,
        });
      }
    }
  }
  const bodies = fibres.flat();
  return { bodies, fibres, frame: [u, v, w], p0, p1, length: L, width, height };
}

/**
 * Render triangles of one fibre segment: a box whose two end caps are (N x N) grids
 * displaced along the grain by the cut fields d0 / d1 (null = flat).
 */
function segmentMesh(p0, u, v, w, s0, s1, cy, cz, fh, fw, N, d0, d1, flags) {
  const P = (s, d, i, j) => v3.frame(p0, u, v, w, s + (d ? d[i * (N + 1) + j] : 0), cy - fh / 2 + (fh * i) / N, cz - fw / 2 + (fw * j) / N);
  const tris = [];
  const quad = (a, b, c, d, f) => { tris.push([a, b, c, f], [a, c, d, f]); };
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    // end cap (+u): (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1) is CCW seen from +u
    quad(P(s1, d1, i, j), P(s1, d1, i + 1, j), P(s1, d1, i + 1, j + 1), P(s1, d1, i, j + 1), flags[1]);
    quad(P(s0, d0, i, j), P(s0, d0, i, j + 1), P(s0, d0, i + 1, j + 1), P(s0, d0, i + 1, j), flags[0]);
  }
  // four sides as strips; orientation fixed by the outward direction
  const side = (pts0, pts1, out, f) => {
    for (let k = 0; k < N; k++) {
      let a = pts0[k], b = pts0[k + 1], c = pts1[k + 1], d = pts1[k];
      const n = cross(sub(b, a), sub(d, a));
      if (dot(n, out) < 0) [b, d] = [d, b];
      quad(a, b, c, d, f);
    }
  };
  const row = (s, d, fixI, fixJ) => [...Array(N + 1).keys()].map((t) => P(s, d, fixI ?? t, fixJ ?? t));
  side(row(s0, d0, 0, null), row(s1, d1, 0, null), mul(v, -1), flags[2]);
  side(row(s0, d0, N, null), row(s1, d1, N, null), v, flags[3]);
  side(row(s0, d0, null, 0), row(s1, d1, null, 0), mul(w, -1), flags[4]);
  side(row(s0, d0, null, N), row(s1, d1, null, N), w, flags[5]);
  return tris;
}

/** Nail / bolt joint strengths (much weaker than solid wood, no real moment capacity). */
export function nail(strength = 1) {
  return { tension: 6e3 * strength, compression: 2e6 * strength, shear: 12e3 * strength, bend: 1.5e3 * strength, torsion: 1.5e3 * strength };
}

/**
 * Stone / concrete block: a grid of jittered hexahedral chunks (they tile exactly)
 * welded by weak mortar-like joints → breaks into irregular chunky debris.
 */
export function stoneBlock(W, center, frame, size, cell = 0.25, o = {}) {
  const rand = o.rand ?? W.rand;
  const [u, v, w] = frame;
  const n = size.map((s) => Math.max(1, Math.round(s / cell)));
  const d = size.map((s, i) => s / n[i]);
  const jit = o.jitter ?? 0.3;
  const corner = new Map();
  const C = (i, j, k) => {
    const key = i + ',' + j + ',' + k;
    if (!corner.has(key)) {
      const jx = i > 0 && i < n[0] ? (rand() - 0.5) * jit * d[0] : 0;
      const jy = j > 0 && j < n[1] ? (rand() - 0.5) * jit * d[1] : 0;
      const jz = k > 0 && k < n[2] ? (rand() - 0.5) * jit * d[2] : 0;
      corner.set(key, v3.frame(center, u, v, w, i * d[0] - size[0] / 2 + jx, j * d[1] - size[1] / 2 + jy, k * d[2] - size[2] / 2 + jz));
    }
    return corner.get(key);
  };
  const ids = new Map();
  const tex = (p) => { const q = sub(p, center); return [dot(q, u), dot(q, v), dot(q, w)]; };
  const I = FLAG.INTERIOR;
  const tint = o.tint ?? rand();
  for (let i = 0; i < n[0]; i++) for (let j = 0; j < n[1]; j++) for (let k = 0; k < n[2]; k++) {
    const corners = [];
    for (let c = 0; c < 8; c++) corners.push(C(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)));
    const flags = [i > 0 ? I : 0, i < n[0] - 1 ? I : 0, j > 0 ? I : 0, j < n[1] - 1 ? I : 0, k > 0 ? I : 0, k < n[2] - 1 ? I : 0];
    const id = W.addBody(hexPoly(corners, flags), { density: o.density ?? STONE.density, material: MAT.STONE, tex, tint, sampleFrame: frame, static: o.static });
    ids.set(i + ',' + j + ',' + k, id);
  }
  const s = o.strength ?? 1;
  const mat = o.stone ?? STONE;
  for (let i = 0; i < n[0]; i++) for (let j = 0; j < n[1]; j++) for (let k = 0; k < n[2]; k++) {
    for (const [di, dj, dk, ax, a1, a2] of [[1, 0, 0, u, 1, 2], [0, 1, 0, v, 0, 2], [0, 0, 1, w, 0, 1]]) {
      const b = ids.get((i + di) + ',' + (j + dj) + ',' + (k + dk));
      if (b === undefined) continue;
      const a = ids.get(i + ',' + j + ',' + k);
      const area = d[a1] * d[a2];
      const anchor = v3.frame(center, u, v, w, (i + 0.5 + di * 0.5) * d[0] - size[0] / 2, (j + 0.5 + dj * 0.5) * d[1] - size[1] / 2, (k + 0.5 + dk * 0.5) * d[2] - size[2] / 2);
      const jr = 0.6 + 0.8 * rand(); // flaws: random strength variation
      W.addJoint(a, b, anchor, ax, {
        tension: mat.tensile * area * s * jr, compression: mat.tensile * area * 30 * s, shear: mat.shear * area * s * jr,
        bend: mat.bending * area * Math.min(d[a1], d[a2]) / 6 * s * jr * 2, torsion: mat.shear * area * Math.min(d[a1], d[a2]) / 3 * s * jr,
      });
    }
  }
  return { bodies: [...ids.values()], ids, n };
}

export function sphereBody(W, center, radius, o = {}) {
  return W.addBody(spherePoly(center, radius, o.detail ?? 2), { density: o.density ?? 7800, material: o.material ?? MAT.METAL, static: o.static, tint: o.tint, shell: o.shell });
}

export function boxBody(W, center, frame, half, o = {}) {
  return W.addBody(boxPoly(center, frame, half), { density: o.density ?? 2300, material: o.material ?? MAT.STONE, static: o.static, sampleFrame: frame, tex: o.tex, tint: o.tint });
}
