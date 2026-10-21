// Scene definitions modelled on classic destruction demo reels.
import { WorldBuilder, v3, boxPoly, MAT, mulberry32 } from './builder.js';
import { woodMember, nail, stoneBlock, sphereBody, boxBody } from './kit.js';
import { GlassPane } from './glass.js';

const { add, sub, mul, norm } = v3;
const AX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/** Weld every pair of touching members (AABB prefilter on member level). */
function nailAll(W, members, strength, tol = 0.25) {
  const box = (m) => {
    if (m._bb) return m._bb;
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (const id of m.bodies) {
      const b = W.bodies[id];
      for (const v of b.verts) {
        const p = add(b.pos, rotq(b.quat, v.p));
        for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
      }
    }
    const e = W.r * 3;
    return (m._bb = [lo.map((x) => x - e), hi.map((x) => x + e)]);
  };
  let n = 0;
  for (let i = 0; i < members.length; i++) for (let j = i + 1; j < members.length; j++) {
    const [la, ha] = box(members[i]), [lb, hb] = box(members[j]);
    if (la[0] > hb[0] || lb[0] > ha[0] || la[1] > hb[1] || lb[1] > ha[1] || la[2] > hb[2] || lb[2] > ha[2]) continue;
    n += W.connectTouching(members[i].bodies, members[j].bodies, typeof strength === 'function' ? strength(members[i], members[j]) : strength, tol);
  }
  return n;
}
function rotq(q, v) {
  const t = mul([q[1] * v[2] - q[2] * v[1], q[2] * v[0] - q[0] * v[2], q[0] * v[1] - q[1] * v[0]], 2);
  return add(add(v, mul(t, q[3])), [q[1] * t[2] - q[2] * t[1], q[2] * t[0] - q[0] * t[2], q[0] * t[1] - q[1] * t[0]]);
}
const member = (W, list, p0, p1, wdt, hgt, o) => { const m = woodMember(W, p0, p1, wdt, hgt, { capRes: W.lite ? 2 : 3, ...o }); list.push(m); return m; };

// ------------------------------------------------------------------ trestle bridge
function trestle(W, opt) {
  const members = [], sills = [];
  const bents = opt.small ? 2 : opt.lite ? 5 : 7, bay = 2.4, H = opt.small ? 3.2 : 5.6;
  const x0 = -((bents - 1) * bay) / 2;
  const post = 0.24, f = { fibre: 0.12, seg: opt.lite ? 1.0 : 0.7 };
  for (let b = 0; b < bents; b++) {
    const x = x0 + b * bay;
    // sill and cap run across the bridge (z)
    sills.push(member(W, members, [x, 0.32, -2.3], [x, 0.32, 2.3], post + 0.06, post, f));
    member(W, members, [x, H - 0.15, -1.6], [x, H - 0.15, 1.6], post + 0.06, 0.3, f);
    // posts: two plumb, two battered
    const batter = opt.small ? 0.45 : 0.8;
    for (const [zb, zt] of [[-0.55, -0.55], [0.55, 0.55], [-1.2 - batter, -1.2], [1.2 + batter, 1.2]]) {
      member(W, members, [x, 0.44 + 0.001, zb], [x, H - 0.3 - 0.001, zt], post, post, { ...f, up: [1, 0, 0] });
    }
    // X bracing on both faces of the bent + mid girt
    for (const side of [-1, 1]) {
      const xf = x + side * (post / 2 + 0.04);
      const zb = 1.2 + batter * 0.9, zt = 1.15;
      member(W, members, [xf, 0.7, -zb], [xf, H - 0.55, zt], 0.08, 0.22, { ...f, up: [1, 0, 0], nw: 1, nh: 2 });
      member(W, members, [xf, 0.7, zb], [xf, H - 0.55, -zt], 0.08, 0.22, { ...f, up: [1, 0, 0], nw: 1, nh: 2 });
      member(W, members, [xf + side * 0.08, H * 0.5, -1.2 - batter * 0.55], [xf + side * 0.08, H * 0.5, 1.2 + batter * 0.55], 0.08, 0.22, { ...f, up: [1, 0, 0], nw: 1, nh: 2 });
    }
  }
  // longitudinal struts between bents (sway bracing) on the plumb posts
  for (let b = 0; b + 1 < bents; b++) {
    const xa = x0 + b * bay, xb = xa + bay;
    for (const z of [-0.55 - 0.16, 0.55 + 0.16]) {
      member(W, members, [xa - 0.1, H * 0.33, z], [xb + 0.1, H * 0.33, z], 0.08, 0.2, { ...f, nw: 1, nh: 2, up: [0, 1, 0] });
      member(W, members, [xa - 0.1, H * 0.7, z], [xb + 0.1, H * 0.7, z], 0.08, 0.2, { ...f, nw: 1, nh: 2, up: [0, 1, 0] });
    }
  }
  // stringers on the caps, ties on stringers
  const yS = H + 0.18;
  for (const z of [-0.95, -0.35, 0.35, 0.95]) {
    for (let b = 0; b + 1 < bents; b++) {
      const xa = x0 + b * bay, xb = xa + bay;
      member(W, members, [xa + (b === 0 ? -0.3 : 0.002), yS, z], [xb - (b + 2 === bents ? -0.3 : 0.002), yS, z], 0.24, 0.36, { fibre: 0.12, seg: opt.lite ? 1.2 : 0.8 });
    }
  }
  const yT = H + 0.36 + 0.075;
  for (let x = x0 - 0.2; x <= x0 + (bents - 1) * bay + 0.2; x += 0.5) {
    member(W, members, [x, yT, -1.35], [x, yT, 1.35], 0.2, 0.15, { fibre: 0.12, seg: 0.9, nw: opt.lite ? 1 : 2, nh: 1 });
  }
  // nailed / bolted connections everywhere members touch
  const nails = nailAll(W, members, nail(opt.nail ?? 1));
  // sills welded to the concrete footings (static bodies at the footing tops)
  for (let b = 0; b < bents; b++) {
    const x = x0 + b * bay;
    const foot = W.addBody(boxPolyStatic([x, 0.1, 0], [0.35, 0.1, 2.4]), { static: true, material: MAT.STONE, density: 2300 });
    W.connectTouching([foot], sills[b].bodies, { tension: 1e6, shear: 1e6, bend: 1e6, torsion: 1e6 }, 0.4);
  }
  // a loaded ore wagon (heavy block) parked above the deck: "Drop load" releases it
  const xm = x0 + bay * Math.floor((bents - 1) / 2) + bay / 2;
  // (kept out of sight until the action drops it onto the deck)
  const load = park(W, boxBody(W, [xm, -40, 0], AX, [1.3, 0.6, 0.75], { density: 4000 }));
  return {
    camera: { pos: [9, 4.5, 11], target: [0, H * 0.55, 0] },
    action: { label: 'Drop load', run: release(W, load, [0, -6, 0], [0, 0, 0], [xm, yT + 2.8, 0]) },
    info: `${members.length} members, ${nails} nailed joints`,
  };
}
function boxPolyStatic(c, h) { return boxPoly(c, AX, h); }

// ------------------------------------------------------------------ helpers
/** Keep a body frozen (static) until released by an event. */
function park(W, id) {
  const b = W.bodies[id];
  b.realInvMass = b.invMass;
  b.invMass = 0;
  return id;
}
/** Event action: wake a parked body with a velocity. */
function release(W, id, vel = [0, 0, 0], ang = [0, 0, 0], at = null) {
  const b = W.bodies[id];
  return (sim) => sim.setBody(id, { pos: at ?? b.pos, quat: b.quat, vel, ang, invMass: b.realInvMass });
}
const STEEL = { density: 7850, tensile: 250e6, compressive: 250e6, bending: 250e6, perp: 250e6, shear: 150e6 };

// ------------------------------------------------------------------ material test
function materials(W, opt) {
  const members = [];
  const len = 4;
  // supports are analytic static boxes (cheaper than static bodies with probes)
  const supports = (z) => [-len / 2 + 0.25, len / 2 - 0.25].map((x) => W.addStatic([x, 0.5, z], [0.25, 0.5, 0.3], [0, 0, 0, 1], { color: 0x9a948a }));
  // wood beam (fibres) and steel bar (plastic welds) side by side
  supports(0.8);
  const beam = member(W, members, [-len / 2, 1.15, 0.8], [len / 2, 1.15, 0.8], 0.21, 0.28, { fibre: 0.07, seg: opt.seg ?? 0.45, strength: opt.strength ?? 1, lateral: opt.lat ?? 1 });
  supports(-0.8);
  member(W, members, [-len / 2, 1.08, -0.8], [len / 2, 1.08, -0.8], 0.16, 0.16, {
    fibre: 0.16, seg: 0.35, material: MAT.METAL, wood: STEEL, plastic: true, strength: opt.steel ?? 0.02,
  });
  // stone wall behind
  stoneBlock(W, [0, 1.25, -3.2], AX, [4.2, 2.5, 0.4], 0.21, { strength: opt.stone ?? 1 });
  // concrete weights dropped on the beams
  const wA = park(W, boxBody(W, [0.15, 3.0, 0.8], AX, [0.3, 0.25, 0.3], { density: 2400 }));
  const wB = park(W, boxBody(W, [-0.1, 3.0, -0.8], AX, [0.3, 0.25, 0.3], { density: 2400 }));
  return {
    camera: { pos: [3.6, 2.6, 5.6], target: [0, 1.1, -0.6] },
    events: [{ t: 0.8, run: release(W, wA) }, { t: 1.6, run: release(W, wB) }],
    action: { label: 'Drop weights', run: (sim) => { release(W, wA)(sim); release(W, wB)(sim); } },
    info: 'wood beam, steel bar, stone wall',
  };
}

// ------------------------------------------------------------------ glass tank
function glassTank(W, opt) {
  const base = W.addBody(boxPoly([0, 0.15, 0], AX, [1.0, 0.15, 1.0]), { static: true, material: MAT.STONE, density: 2300 });
  const hw = 0.6, H = 2.1, t = 0.016, y0 = 0.3, yc = y0 + H / 2;
  const fine = opt.lite ? 0.75 : 1;
  const P = (c, u, w, focus) => new GlassPane(W, c, u, [0, 1, 0], w, H, t, { focus, sectors: Math.round(22 * fine), rings: Math.round(7 * fine) });
  const front = P([0, yc, hw + t / 2], [1, 0, 0], 2 * hw + 2 * t, [0.05, -0.15]);
  const back = P([0, yc, -hw - t / 2], [-1, 0, 0], 2 * hw + 2 * t);
  const right = P([hw + t / 2, yc, 0], [0, 0, -1], 2 * hw);
  const left = P([-hw - t / 2, yc, 0], [0, 0, 1], 2 * hw);
  const panes = [front, right, back, left];
  const seal = { tension: 4e3, compression: 1e6, shear: 4e3, bend: 300, torsion: 300 };
  for (let i = 0; i < 4; i++) {
    const a = panes[i], b = panes[(i + 1) % 4];
    W.connectTouching([...a.sides.left, ...a.sides.right], [...b.sides.left, ...b.sides.right], seal, 0.6);
  }
  for (const p of panes) W.connectTouching([base], p.sides.bottom, { tension: 2e4, compression: 1e7, shear: 2e4, bend: 2e3, torsion: 2e3 }, 0.6);
  // wooden pole standing inside the tank
  const members = [];
  const pole = member(W, members, [0, y0 + 0.001, 0], [0, y0 + 2.7, 0], 0.16, 0.16, { fibre: 0.08, seg: 0.45, up: [1, 0, 0] });
  W.connectTouching([base], pole.bodies, { tension: 5e4, compression: 1e8, shear: 5e4, bend: 5e3, torsion: 5e3 }, 0.6);
  return { camera: { pos: [2.4, 1.9, 4.2], target: [0, 1.3, 0] }, info: 'glass tank', panes };
}

// ------------------------------------------------------------------ hut on a cliff + boulder
function hut(W, opt) {
  const members = [];
  const top = 4;
  W.addStatic([0, top / 2, 0], [7, top / 2, 5], [0, 0, 0, 1], { color: 0x8a8174 });
  const lx = 3.0, lz = 2.4, wallH = 2.0, x0 = 1.2; // hut centred at x0
  const f = { fibre: 0.1, seg: opt.lite ? 0.8 : 0.6 };
  const y = (h) => top + h;
  // sills and floor boards
  for (const z of [-lz / 2 + 0.08, lz / 2 - 0.08]) member(W, members, [x0 - lx / 2, y(0.08), z], [x0 + lx / 2, y(0.08), z], 0.16, 0.16, f);
  const nFloor = Math.round(lx / 0.25);
  for (let i = 0; i < nFloor; i++) {
    const x = x0 - lx / 2 + (i + 0.5) * (lx / nFloor);
    member(W, members, [x, y(0.185), -lz / 2], [x, y(0.185), lz / 2], lx / nFloor - 0.004, 0.05, { ...f, nh: 1 });
  }
  const fl = y(0.21);
  // corner posts and top plates
  const cp = 0.12, pin = 0.11;
  const corners = [[x0 - lx / 2 + pin, -lz / 2 + pin], [x0 + lx / 2 - pin, -lz / 2 + pin], [x0 + lx / 2 - pin, lz / 2 - pin], [x0 - lx / 2 + pin, lz / 2 - pin]];
  for (const [x, z] of corners) member(W, members, [x, fl + 0.001, z], [x, fl + wallH, z], cp, cp, { ...f, up: [1, 0, 0] });
  const tp = fl + wallH + 0.06;
  member(W, members, [corners[0][0] - 0.06, tp, corners[0][1]], [corners[1][0] + 0.06, tp, corners[1][1]], cp, cp, f);
  member(W, members, [corners[3][0] - 0.06, tp, corners[3][1]], [corners[2][0] + 0.06, tp, corners[2][1]], cp, cp, f);
  member(W, members, [corners[0][0], tp + cp, corners[0][1] - 0.06], [corners[3][0], tp + cp, corners[3][1] + 0.06], cp, cp, f);
  member(W, members, [corners[1][0], tp + cp, corners[1][1] - 0.06], [corners[2][0], tp + cp, corners[2][1] + 0.06], cp, cp, f);
  // wall boards (vertical) just outside the frame; door in the +x wall
  const bw = 0.2, bt = 0.04, wb = { ...f, nh: 1, nw: 2 };
  const boards = (ax, from, to, fixed, normal) => {
    const L = Math.abs(to - from), n = Math.round(L / bw);
    for (let i = 0; i < n; i++) {
      const c = from + (to - from) * (i + 0.5) / n;
      if (normal[0] > 0 && Math.abs(c - 0.15) < 0.4) continue; // doorway
      const p = ax === 'x' ? [c, 0, fixed] : [fixed, 0, c];
      const p0 = [p[0], fl + 0.001, p[2]], p1 = [p[0], fl + wallH + 0.1, p[2]];
      member(W, members, p0, p1, L / n - 0.004, bt, { ...wb, up: normal });
    }
  };
  const ox = lx / 2 - pin + cp / 2 + bt / 2, oz = lz / 2 - pin + cp / 2 + bt / 2;
  boards('x', x0 - ox - bt / 2, x0 + ox + bt / 2, -oz, [0, 0, -1]);
  boards('x', x0 - ox - bt / 2, x0 + ox + bt / 2, oz, [0, 0, 1]);
  boards('z', -oz + bt / 2, oz - bt / 2, x0 - ox, [-1, 0, 0]);
  boards('z', -oz + bt / 2, oz - bt / 2, x0 + ox, [1, 0, 0]);
  // gable roof: rafters from the side plates to a ridge, boards along x
  const ridgeY = tp + 1.0, eave = oz + 0.25;
  member(W, members, [x0 - lx / 2 - 0.2, ridgeY, 0], [x0 + lx / 2 + 0.2, ridgeY, 0], 0.1, 0.16, f);
  const slope = Math.atan2(ridgeY - tp, eave);
  for (const side of [-1, 1]) {
    for (let i = 0; i < 4; i++) {
      const x = x0 - lx / 2 + 0.1 + i * (lx - 0.2) / 3;
      const p0 = [x, tp + cp / 2 + 0.06 - Math.tan(slope) * 0.25, side * eave], p1 = [x, ridgeY + 0.02, side * 0.05];
      member(W, members, p0, p1, 0.08, 0.12, { ...f, nw: 1, nh: 1, up: [0, 1, 0] });
    }
    const n = 7;
    const nrm = [0, Math.cos(slope), side * Math.sin(slope)];
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) / n;
      const zc = side * (eave * (1 - t) + 0.05 * t), yc = tp + 0.03 + (ridgeY - tp) * t + 0.13;
      member(W, members, [x0 - lx / 2 - 0.25, yc, zc], [x0 + lx / 2 + 0.25, yc, zc], 0.04, eave / n / Math.cos(slope) - 0.01, { ...f, nh: 1, up: [0, nrm[2] * -side, side] });
    }
  }
  const nails = nailAll(W, members, nail(opt.nail ?? 0.6), 0.35);
  // boulder parked on a ledge, released rolling towards the hut
  const R = 0.95;
  // a higher ledge behind the hut: the boulder rolls off it and lands on the roof
  const ledgeTop = top + 4.5;
  W.addStatic([x0 - 8.5, ledgeTop / 2, 0], [3.5, ledgeTop / 2, 4], [0, 0, 0, 1], { color: 0x8a8174 });
  const boulder = park(W, sphereBody(W, [x0 - 6.2, ledgeTop + R + 0.02, 0.15], R, { density: 2600, material: MAT.STONE, shell: W.r * 1.6, detail: 3 }));
  const v = 6.5;
  return {
    camera: { pos: [x0 + 7, top + 3.2, 9.5], target: [x0 - 1.8, top + 2.0, 0] },
    events: [{ t: 1.0, run: release(W, boulder, [v, 0, 0], [0, 0, -v / R]) }],
    action: { label: 'Roll boulder', run: release(W, boulder, [v, 0, 0], [0, 0, -v / R]) },
    info: `${members.length} members, ${nails} nails`,
    settings: {},
  };
}

// ------------------------------------------------------------------ window in a wall
// A two-light window: wooden frame (sill, head, jambs, mullion) set into a concrete
// wall opening, each light a glass pane puttied into the frame.
function windowWall(W, opt) {
  const members = [];
  const wallT = 0.3, ox = 1.5, y0 = 0.8, y1 = 3.0, top = 4.2, half = 4.5;
  const wall = { color: 0xa7a39b };
  W.addStatic([-(half + ox) / 2, top / 2, 0], [(half - ox) / 2, top / 2, wallT / 2], [0, 0, 0, 1], wall);
  W.addStatic([(half + ox) / 2, top / 2, 0], [(half - ox) / 2, top / 2, wallT / 2], [0, 0, 0, 1], wall);
  W.addStatic([0, y0 / 2, 0], [ox, y0 / 2, wallT / 2], [0, 0, 0, 1], wall);
  W.addStatic([0, (y1 + top) / 2, 0], [ox, (top - y1) / 2, wallT / 2], [0, 0, 0, 1], wall);
  // frame: a = face width, d = depth (into the wall)
  const a = 0.1, d = 0.14, g = 0.001;
  const f = { fibre: 0.05, seg: opt.lite ? 0.55 : 0.4, nw: opt.lite ? 2 : 3, nh: 2 };
  const sill = member(W, members, [-ox, y0 + a / 2, 0], [ox, y0 + a / 2, 0], d, a, f);
  const head = member(W, members, [-ox, y1 - a / 2, 0], [ox, y1 - a / 2, 0], d, a, f);
  const jambs = [-1, 1].map((sx) => member(W, members, [sx * (ox - a / 2), y0 + a + g, 0], [sx * (ox - a / 2), y1 - a - g, 0], d, a, { ...f, up: [1, 0, 0] }));
  const mullion = member(W, members, [0, y0 + a + g, 0], [0, y1 - a - g, 0], d, a, { ...f, up: [1, 0, 0] });
  const nails = nailAll(W, members, nail(4));
  // frame anchored in the masonry: hidden static strips inside the wall around the opening
  const anchor = (c, h) => W.addBody(boxPoly(c, AX, h), { static: true, material: MAT.STONE, density: 2300 });
  const fix = { tension: 1e5, compression: 1e8, shear: 1e5, bend: 2e4, torsion: 2e4 };
  W.connectTouching([anchor([0, y0 - 0.02, 0], [ox, 0.02, d / 2])], sill.bodies, fix, 0.4);
  W.connectTouching([anchor([0, y1 + 0.02, 0], [ox, 0.02, d / 2])], head.bodies, fix, 0.4);
  for (const [i, sx] of [[0, -1], [1, 1]]) W.connectTouching([anchor([sx * (ox + 0.02), (y0 + y1) / 2, 0], [0.02, (y1 - y0) / 2, d / 2])], jambs[i].bodies, fix, 0.4);
  // two glass lights, puttied into the frame on all four sides
  const t = 0.012, pw = ox - 1.5 * a, ph = y1 - y0 - 2 * a, yc = (y0 + y1) / 2;
  const fine = opt.lite ? 0.75 : 1;
  const putty = { tension: 2.5e4, compression: 1e7, shear: 2.5e4, bend: 600, torsion: 600 };
  const panes = [-1, 1].map((sx) => {
    const p = new GlassPane(W, [sx * (a / 2 + pw / 2), yc, 0], [1, 0, 0], [0, 1, 0], pw, ph, t, { sectors: Math.round(22 * fine), rings: Math.round(7 * fine) });
    const side = sx < 0 ? [jambs[0], mullion] : [mullion, jambs[1]];
    W.connectTouching(p.sides.left, side[0].bodies, putty, 0.5);
    W.connectTouching(p.sides.right, side[1].bodies, putty, 0.5);
    W.connectTouching(p.sides.bottom, sill.bodies, putty, 0.5);
    W.connectTouching(p.sides.top, head.bodies, putty, 0.5);
    return p;
  });
  const throwAt = (x, y) => (sim, st) => st.fire([x * 0.6 + 0.4, y - 0.3, 6], [x - (x * 0.6 + 0.4), y - (y - 0.3), -6]);
  return {
    camera: { pos: [1.4, 1.9, 5.4], target: [0, 1.8, 0] },
    events: [{ t: 1.2, run: throwAt(-0.75, 2.1) }],
    action: { label: 'Throw at window', run: (sim, st) => throwAt(Math.random() < 0.5 ? -0.75 : 0.75, 1.4 + Math.random() * 1.0)(sim, st) },
    info: `${members.length} frame members, ${nails} joints`,
    panes,
  };
}

// ------------------------------------------------------------------ hall roof beams
// Heavy timbers spanning a stone hall, purlins nailed across them and a glass
// skylight on the purlins. Crates fall through the skylight onto the beams.
function hallRoof(W, opt) {
  const members = [];
  const wallH = 5.0, wx = 4.0, hallL = 9.5;
  const stone = { color: 0x8f8a80 };
  for (const sx of [-1, 1]) W.addStatic([sx * wx, wallH / 2, 0], [0.5, wallH / 2, hallL / 2], [0, 0, 0, 1], stone);
  // beams rest on the wall tops (simply supported, friction only)
  const bw = 0.22, bh = 0.3, L = wx + 0.25;
  const f = opt.lite ? { fibre: 0.1, seg: 0.8 } : { fibre: 0.075, seg: 0.6 };
  const zs = [-3.2, -1.6, 0, 1.6, 3.2];
  const beams = zs.map((z) => member(W, members, [-L, wallH + bh / 2, z], [L, wallH + bh / 2, z], bw, bh, { ...f, strength: opt.strength ?? 0.75 }));
  // purlins along the hall on top of the beams
  const yP = wallH + bh + 0.06;
  for (const x of [-2.6, -0.9, 0.9, 2.6]) {
    member(W, members, [x, yP, -3.5], [x, yP, 3.5], 0.1, 0.12, { fibre: 0.06, seg: opt.lite ? 0.8 : 0.6, nw: 2, nh: 2 });
  }
  const nails = nailAll(W, members, nail(1.5), 0.35);
  // glass skylight over the middle bay, bedded on the two inner purlins
  const t = 0.014, gw = 1.8 + 0.1, gd = 3.0;
  const fine = opt.lite ? 0.7 : 1;
  const pane = new GlassPane(W, [0, yP + 0.06 + t / 2, 0], [1, 0, 0], [0, 0, -1], gw, gd, t, { sectors: Math.round(20 * fine), rings: Math.round(6 * fine), strength: 3 });
  const purl = members.slice(beams.length);
  W.connectTouching(pane.bodies, [...purl[1].bodies, ...purl[2].bodies], { tension: 8e3, compression: 1e7, shear: 8e3, bend: 200, torsion: 200 }, 0.6);
  // crates parked out of sight, dropped by events / the action button
  const crate = (s) => park(W, boxBody(W, [0, -40 - s, 0], AX, [0.35, 0.35, 0.35], { density: 2400 }));
  const c1 = crate(0), c2 = crate(3), c3 = crate(6);
  const drop = [
    { t: 1.0, run: release(W, c1, [0, -2, 0], [0.4, 0, 0.3], [0.2, wallH + 5.5, 0.05]) },
    { t: 3.2, run: release(W, c2, [0, -2, 0], [0, 0.5, 0.6], [-1.7, wallH + 5.5, 1.6]) },
    { t: 5.0, run: release(W, c3, [0, -3, 0], [0.5, 0, 0], [1.9, wallH + 5.5, -1.6]) },
  ];
  return {
    camera: { pos: [2.0, 1.4, 7.8], target: [0, 4.9, 0] },
    events: drop,
    action: { label: 'Drop crates', run: (sim, st) => drop.forEach((e) => e.run(sim, st)) },
    info: `${members.length} members, ${nails} nails`,
    panes: [pane],
  };
}

// ------------------------------------------------------------------ registry
export const SCENES = {
  walltest: { title: 'wall', hidden: true, radius: 0.035, build: (W, o) => { stoneBlock(W, [0, 1.25, 0], AX, [o.wl ?? 4.2, 2.5, 0.4], o.cell ?? 0.26, {}); return { camera: { pos: [6, 2, 6], target: [0, 1, 0] } }; } },
  window: { title: 'Window', radius: 0.03, build: (W, o) => windowWall(W, o), projR: 0.08 },
  roof: { title: 'Roof beams', radius: 0.045, build: (W, o) => hallRoof(W, o), projR: 0.2 },
  trestle: { title: 'Trestle bridge', radius: 0.06, build: (W, o) => trestle(W, o) },
  glass: { title: 'Glass tank', radius: 0.035, build: (W, o) => glassTank(W, o), projR: 0.16 },
  hut: { title: 'Hut + boulder', radius: 0.05, build: (W, o) => hut(W, o) },
  materials: { title: 'Wood / steel / stone', radius: 0.035, build: (W, o) => materials(W, o) },
};

export function buildScene(name, opt = {}) {
  const def = SCENES[name] ?? SCENES.trestle;
  const W = new WorldBuilder({ radius: def.radius });
  W.rand = mulberry32(opt.seed ?? 7);
  W.lite = !!opt.lite;
  const meta = def.build(W, opt);
  // projectile pool: parked (static) far below the world until fired
  const projR = opt.projR ?? def.projR ?? 0.28;
  meta.projectiles = [];
  for (let i = 0; i < (opt.pool ?? 6); i++) {
    meta.projectiles.push(park(W, sphereBody(W, [i * 3, -50, 0], projR, { density: 3000, shell: W.r * 1.6 })));
  }
  meta.projR = projR;
  const data = W.finish();
  for (const p of meta.panes ?? []) p.collectExternal();
  return { W, data, meta };
}
