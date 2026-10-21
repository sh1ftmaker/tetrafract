// GPU rigid-body destruction solver (raw WGSL compute on the three.js WebGPU device).
//
// Method: every breakable object is pre-split into small rigid pieces (wood fibre
// segments, glass shards, stone chunks). Pieces are welded together by stiff
// breakable joints and solved with XPBD (Müller et al. 2020, "Detailed Rigid Body
// Simulation with Extended Position Based Dynamics") using many small substeps
// ("Small Steps in Physics Simulation", Macklin et al. 2019).
//
//  * A rigid piece can never flex, so stiffness only depends on the joints. Joints
//    are solved Gauss-Seidel style: they are graph-coloured on the CPU at build time
//    so that joints of one colour never share a body and can run in parallel.
//  * Each joint is a full 6-DOF weld (anchor point + relative orientation). After each
//    substep the Lagrange multipliers give the joint force/torque, which is decomposed
//    into tension/compression along the joint axis, shear, bending and torsion and
//    compared with anisotropic strengths. Wood fibres are strong along the grain and
//    weak across it, so beams break into long splinters. Plastic joints (steel, and
//    wood for a few degrees before it snaps) cap the transmitted moment and keep the
//    rotation that gets through. Brittle joints (glass) fail on the first overload,
//    others need two consecutive overloaded substeps (filters solver noise).
//  * Collision: each piece carries probe spheres ("particles") in its volume and an exact
//    convex shape (≤ 12 planes, or a sphere). A hashed uniform grid over the particles
//    finds nearby bodies; each particle is then tested against the neighbour's real convex
//    hull, so flat faces rest flat on each other (no bumpy sphere stacking). Contacts are
//    Jacobi-averaged XPBD contacts with Coulomb friction. Pieces still welded together
//    ignore each other (per-body adjacency list), so welds alone define the structure.
//
// Buffers (≤ 8 storage buffers in a single shared layout):
//   bodies, particles, joints, shapes (adjacency + hull planes), gridCount(+stats), gridSlots, poses

export const BODY_FLOATS = 32;     // 128 bytes
export const JOINT_FLOATS = 32;    // 128 bytes
export const PARTICLE_FLOATS = 20; // 80 bytes
export const ADJ = 24;             // adjacency slots per body (multiple of 4)
export const MAX_PLANES = 12;      // convex hull planes per body
export const SHAPE_VEC4 = ADJ / 4 + 1 + MAX_PLANES; // per body: adjacency, header, planes
export const SLOTS = 8;            // particle slots per grid cell
export const MAX_STATICS = 24;
const STATS = 8;                   // atomic counters after the grid counts

const WGSL = /* wgsl */`
struct Params {
  h: f32, gravity: f32, radius: f32, cellSize: f32,
  nBodies: u32, nParticles: u32, nJoints: u32, hashMask: u32,
  omega: f32, muS: f32, muK: f32, linDamp: f32,
  nStatics: u32, groundY: f32, breakScale: f32, maxDepen: f32,
  angDamp: f32, maxSpeed: f32, pad3: f32, slop: f32,
  shatterV: f32, pad0: f32, pad1: f32, pad2: f32,
};
struct Statics { box: array<vec4f, ${MAX_STATICS * 3}> }; // center, quat, half(w = friction)
struct Range { start: u32, count: u32 };

struct Body {
  pos: vec4f,   // xyz centre of mass, w = inverse mass (0 = static)
  rot: vec4f,   // orientation quaternion (principal frame)
  ppos: vec4f,  // position at start of substep
  prot: vec4f,
  vel: vec4f,   // linear velocity
  ang: vec4f,
  iinv: vec4f,  // inverse principal inertia (local)
  info: vec4u,  // first particle, particle count, material, flags
};
struct Particle {
  local: vec4f, // position in body frame, w = bitcast body index
  world: vec4f, // current world position, w = bitcast body index
  prev: vec4f,  // world position at start of substep, w = probe radius (depth inside own body)
  dlin: vec4f,  // Jacobi contact output: linear impulse sum, w = contact count
  dang: vec4f,  // angular impulse sum
};
struct Joint {
  ids: vec4u,   // body a, body b, adjacency slots (a | b<<16), flags (1 broken, 2 plastic, 4 brittle, bits 8+ overload count)
  la: vec4f,    // anchor in A frame, w = linear compliance
  lb: vec4f,    // anchor in B frame, w = angular compliance
  q0: vec4f,    // rest relative rotation conj(qA) * qB
  axis: vec4f,  // joint axis in A frame, w = compressive strength
  str: vec4f,   // tensile, shear, bending, torsion strengths (N, N, Nm, Nm)
  lin: vec4f,   // linear impulse accumulated this substep (world), w = plastic strain
  angl: vec4f,  // angular impulse accumulated this substep (world), w = plastic strain limit
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> S: Statics;
@group(0) @binding(2) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(3) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(4) var<storage, read_write> joints: array<Joint>;
@group(0) @binding(5) var<storage, read_write> shapes: array<vec4u>; // per body: 4 adjacency, header (type, nPlanes, radius), planes
@group(0) @binding(6) var<storage, read_write> gridCount: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> gridSlots: array<u32>;
@group(0) @binding(8) var<storage, read_write> poses: array<vec4f>;
@group(1) @binding(0) var<uniform> R: Range;

const SL: u32 = ${SLOTS}u;
const AV: u32 = ${ADJ / 4}u; // adjacency vec4s per body
const SV: u32 = ${ADJ / 4 + 1 + MAX_PLANES}u;
const NONE: u32 = 0xffffffffu;

fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn qconj(q: vec4f) -> vec4f { return vec4f(-q.xyz, q.w); }
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
// apply a small rotation vector (radians, world) to quaternion q
fn qadd(q: vec4f, w: vec3f) -> vec4f {
  return normalize(q + 0.5 * qmul(vec4f(w, 0.0), q));
}
// world-space inverse inertia times v
fn invI(rot: vec4f, iinv: vec3f, v: vec3f) -> vec3f {
  return qrot(rot, qrot(qconj(rot), v) * iinv);
}
fn cellOf(x: vec3f) -> vec3i { return vec3i(floor(x / P.cellSize)); }
fn hashCell(c: vec3i) -> u32 {
  let u = bitcast<vec3u>(c);
  return ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u)) & P.hashMask;
}

// ---------------------------------------------------------------- substep start
fn integrateBody(i: u32) {
  var b = bodies[i];
  b.ppos = b.pos;
  b.prot = b.rot;
  if (b.pos.w > 0.0) {
    var v = b.vel.xyz;
    var w = b.ang.xyz;
    v.y -= P.gravity * P.h;
    v *= P.linDamp;
    w *= P.angDamp;
    let sp = length(v);
    if (sp > P.maxSpeed) { v *= P.maxSpeed / sp; }
    let ws = length(w);
    if (ws > 80.0) { w *= 80.0 / ws; }
    b.vel = vec4f(v, b.vel.w);
    b.ang = vec4f(w, 0.0);
    b.pos = vec4f(b.pos.xyz + v * P.h, b.pos.w);
    b.rot = qadd(b.rot, w * P.h);
  }
  bodies[i] = b;
}

// substep start: predict body poses + clear the hash grid (one dispatch for both)
@compute @workgroup_size(64) fn beginSubstep(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i < P.nBodies) { integrateBody(i); }
  if (i <= P.hashMask) { atomicStore(&gridCount[i], 0u); }
}

// world positions for the grid + previous positions (friction, depenetration limit)
@compute @workgroup_size(64) fn insert(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.nParticles) { return; }
  let lp = particles[i].local;
  let b = bodies[bitcast<u32>(lp.w)];
  let x = b.pos.xyz + qrot(b.rot, lp.xyz);
  particles[i].world = vec4f(x, lp.w);
  particles[i].prev = vec4f(b.ppos.xyz + qrot(b.prot, lp.xyz), particles[i].prev.w);
  let h = hashCell(cellOf(x));
  let s = atomicAdd(&gridCount[h], 1u);
  if (s < SL) { gridSlots[h * SL + s] = i; }
}

@compute @workgroup_size(64) fn updateWorld(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.nParticles) { return; }
  let lp = particles[i].local;
  let b = bodies[bitcast<u32>(lp.w)];
  particles[i].world = vec4f(b.pos.xyz + qrot(b.rot, lp.xyz), lp.w);
}

// ---------------------------------------------------------------- contacts
struct Acc { dl: vec3f, da: vec3f, n: f32 };

// One XPBD contact on body A at contact point pc with normal n (pointing into A's side),
// penetration pen, relative tangential displacement rel. wB = generalized inverse mass of
// the other body. Friction is a positional Coulomb model (static if slip < muS * pen).
fn contact(acc: ptr<function, Acc>, A: Body, pc: vec3f, n: vec3f, pen: f32, rel: vec3f, wB: f32, mu: f32) {
  let rA = pc - A.pos.xyz;
  let rn = cross(rA, n);
  let wA = A.pos.w + dot(rn, invI(A.rot, A.iinv.xyz, rn));
  let w = wA + wB;
  let dlam = pen / w;
  var p = dlam * n;
  let rt = rel - dot(rel, n) * n;
  let lt = length(rt);
  if (lt > 1e-7) {
    let corr = select(min(P.muK * mu * pen, lt), lt, lt < P.muS * mu * pen);
    p -= (corr / w) * (rt / lt);
  }
  (*acc).dl += p;
  (*acc).da += cross(rA, p);
  (*acc).n += 1.0;
}

// shatter level of a brittle piece hit at speed vin by mass m: 0 = no shatter, 1 = this
// piece only, each further level reaches one more ring of shards (impact energy doublings)
fn impactLevel(vin: f32, m: f32) -> f32 {
  let v = vin * sqrt(P.breakScale);
  if (v < P.shatterV) { return 0.0; }
  return clamp(1.0 + log2(max(0.5 * m * v * v / 25.0, 1.0)), 1.0, 12.0);
}

// limit how fast pre-existing overlap is resolved (prevents explosive pops)
fn limitPenV(pen: f32, pen0: f32, vmax: f32) -> f32 {
  if (pen0 > 0.0) { return pen - max(pen0 - vmax * P.h, 0.0); }
  return pen;
}
fn limitPen(pen: f32, pen0: f32) -> f32 { return limitPenV(pen, pen0, P.maxDepen); }

@compute @workgroup_size(64) fn contacts(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.nParticles) { return; }
  let me = particles[i];
  let bi = bitcast<u32>(me.local.w);
  let A = bodies[bi];
  var acc = Acc(vec3f(0.0), vec3f(0.0), 0.0);
  // Glass shatters on impact: a shard struck faster than shatterV loses its brittle
  // welds at once. (Its own inertia is too small for the solver to build up the stress
  // a real impact would: the shard just gets shoved aside.) The impact energy sets a
  // shatter level, which spreads that many welds outwards through the crack pattern
  // (breakJoint), so a hard hit bursts the whole pane and a pebble makes a small hole.
  let brittle = A.info.z == 1u;
  var hit = 0.0;
  if (A.pos.w > 0.0) {
    let x = me.world.xyz;
    let r = me.prev.w; // probe radius = depth of this particle inside its own body, so the
                       // probe sphere never pokes out of the real surface
    let reach2 = 16.0 * P.radius * P.radius;
    var seen: array<u32, 12>;
    var ns = 0u;
    let base = cellOf(x);
    for (var dz = -1; dz <= 1; dz++) {
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let c = base + vec3i(dx, dy, dz);
          let h = hashCell(c);
          let cnt = min(atomicLoad(&gridCount[h]), SL);
          for (var s = 0u; s < cnt; s++) {
            let j = gridSlots[h * SL + s];
            let o = particles[j].world;
            let bj = bitcast<u32>(o.w);
            if (bj == bi) { continue; }
            let d = x - o.xyz;
            if (dot(d, d) >= reach2) { continue; }
            // each neighbouring body is tested once against its exact shape
            var dup = false;
            for (var q = 0u; q < ns; q++) { if (seen[q] == bj) { dup = true; } }
            if (dup) { continue; }
            if (ns < 12u) { seen[ns] = bj; ns++; }
            let bv = vec4u(bj);
            let sv = vec4u(bj | 0x80000000u);
            var weld = false;
            var soft = false;
            for (var s = 0u; s < AV; s++) {
              let a = shapes[bi * SV + s];
              weld = weld || any(a == bv);
              soft = soft || any(a == sv);
            }
            if (weld) { continue; }
            // "soft" neighbours: touching/overlapping in the as-built state but not welded
            // (e.g. diagonal stone chunks). Ignored until one of them has cracked.
            let B = bodies[bj];
            if (soft && ((A.info.w | B.info.w) & 1u) == 0u) { continue; }
            let hdr = shapes[bj * SV + AV];
            let qi = qconj(B.rot);
            let lx = qrot(qi, x - B.pos.xyz);
            let lp = qrot(qconj(B.prot), me.prev.xyz - B.ppos.xyz);
            var sd = -1e9;
            var sd0 = 0.0;
            var nl = vec3f(0.0, 1.0, 0.0);
            var pw = 0.0;
            if (hdr.x == 1u) {
              // sphere
              let R = bitcast<f32>(hdr.z);
              let l = length(lx);
              sd = l - R;
              if (l > 1e-6) { nl = lx / l; }
              sd0 = length(lp) - R;
            } else {
              for (var k = 0u; k < hdr.y; k++) {
                let pl = bitcast<vec4f>(shapes[bj * SV + AV + 1u + k]);
                let sk = dot(pl.xyz, lx) - pl.w;
                if (sk > sd) { sd = sk; nl = pl.xyz; pw = pl.w; }
              }
              sd0 = dot(nl, lp) - pw;
            }
            if (sd >= r) { continue; }
            // small allowed overlap: pieces resting on each other (or touching fragments of
            // one cracked object) don't fight the joints over sub-millimetre errors
            let slop = P.slop * P.radius * select(1.0, 3.0, soft);
            let pen = limitPen(r - sd - slop, r - sd0 - slop);
            if (pen <= 0.0) { continue; }
            let n = qrot(B.rot, nl);
            let pc = x - n * (0.5 * (r + sd));
            var wB = 0.0;
            if (B.pos.w > 0.0) {
              let rB = cross(pc - B.pos.xyz, n);
              wB = B.pos.w + dot(rB, invI(B.rot, B.iinv.xyz, rB));
            }
            // relative displacement of the two material points during this substep
            let pcB0 = B.ppos.xyz + qrot(B.prot, qrot(qi, pc - B.pos.xyz));
            let rel = (x - me.prev.xyz) - (pc - pcB0);
            if (brittle) {
              let vin = -dot(rel, n) / P.h;
              let mB = select(1.0 / A.pos.w, 1.0 / B.pos.w, B.pos.w > 0.0);
              hit = max(hit, impactLevel(vin, mB));
            }
            contact(&acc, A, pc, n, pen, rel, wB, 1.0);
          }
        }
      }
    }
    // ground plane
    let rel = x - me.prev.xyz;
    // the ground and static boxes are immovable: resolve overlap with them faster
    let gp = limitPenV(r - (x.y - P.groundY), r - (me.prev.y - P.groundY), 6.0 * P.maxDepen);
    if (gp > 0.0) {
      if (brittle) { hit = max(hit, impactLevel(-rel.y / P.h, 1.0 / A.pos.w)); }
      contact(&acc, A, x - vec3f(0.0, r, 0.0), vec3f(0.0, 1.0, 0.0), gp, rel, 0.0, 1.0);
    }
    // static oriented boxes (cliffs, supports)
    for (var k = 0u; k < P.nStatics; k++) {
      let c = S.box[k * 3u].xyz;
      let q = S.box[k * 3u + 1u];
      let he = S.box[k * 3u + 2u];
      let lx = qrot(qconj(q), x - c);
      let lpv = qrot(qconj(q), me.prev.xyz - c);
      let cl = clamp(lx, -he.xyz, he.xyz);
      let dl = lx - cl;
      let dist = length(dl);
      var n = vec3f(0.0);
      var pen = 0.0;
      var pen0 = 0.0;
      if (dist > 1e-6) {
        if (dist >= r) { continue; }
        n = dl / dist;
        pen = r - dist;
        pen0 = r - length(lpv - clamp(lpv, -he.xyz, he.xyz));
      } else {
        // centre inside the box: push out through the nearest face
        let fd = he.xyz - abs(lx);
        var ax = 0;
        if (fd.y < fd.x && fd.y <= fd.z) { ax = 1; } else if (fd.z < fd.x && fd.z < fd.y) { ax = 2; }
        n[ax] = sign(lx[ax]);
        pen = fd[ax] + r;
        pen0 = he[ax] - abs(lpv[ax]) + r;
      }
      pen = limitPenV(pen, pen0, 6.0 * P.maxDepen);
      if (pen <= 0.0) { continue; }
      let nw = qrot(q, n);
      if (brittle) { hit = max(hit, impactLevel(-dot(rel, nw) / P.h, 1.0 / A.pos.w)); }
      contact(&acc, A, x - nw * r, nw, pen, rel, 0.0, he.w);
    }
  }
  let lvl = u32(hit);
  if (lvl > ((A.info.w >> 8u) & 0xffu)) { bodies[bi].info.w = (A.info.w & 0xffu) | 2u | (lvl << 8u); }
  particles[i].dlin = vec4f(acc.dl, acc.n);
  particles[i].dang = vec4f(acc.da, 0.0);
}

// gather the contact impulses of all particles of a body; Jacobi average
@compute @workgroup_size(64) fn applyContacts(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.nBodies) { return; }
  let b = bodies[i];
  if (b.pos.w == 0.0) { return; }
  var dl = vec3f(0.0);
  var da = vec3f(0.0);
  var n = 0.0;
  let s0 = b.info.x;
  for (var k = 0u; k < b.info.y; k++) {
    let p = particles[s0 + k];
    dl += p.dlin.xyz;
    n += p.dlin.w;
    da += p.dang.xyz;
  }
  if (n == 0.0) { return; }
  let f = P.omega / n;
  bodies[i].pos = vec4f(b.pos.xyz + dl * (b.pos.w * f), b.pos.w);
  bodies[i].rot = qadd(b.rot, invI(b.rot, b.iinv.xyz, da) * f);
}

// ---------------------------------------------------------------- joints (one colour per dispatch)
@compute @workgroup_size(64) fn solveJoints(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= R.count) { return; }
  let k = R.start + g.x;
  var J = joints[k];
  if ((J.ids.w & 1u) != 0u) { return; }
  let ia = J.ids.x;
  let ib = J.ids.y;
  var pa = bodies[ia].pos; var qa = bodies[ia].rot; let ia3 = bodies[ia].iinv.xyz;
  var pb = bodies[ib].pos; var qb = bodies[ib].rot; let ib3 = bodies[ib].iinv.xyz;
  let h2 = P.h * P.h;

  // positional part: anchors must coincide
  let rA = qrot(qa, J.la.xyz);
  let rB = qrot(qb, J.lb.xyz);
  let d = (pb.xyz + rB) - (pa.xyz + rA);
  let c = length(d);
  if (c > 1e-9) {
    let n = d / c;
    let ra = cross(rA, n);
    let rb = cross(rB, n);
    let w = pa.w + dot(ra, invI(qa, ia3, ra)) + pb.w + dot(rb, invI(qb, ib3, rb));
    let alpha = J.la.w / h2;
    let dl = (c - alpha * dot(J.lin.xyz, n)) / (w + alpha);
    let p = dl * n;
    pa = vec4f(pa.xyz + p * pa.w, pa.w);
    pb = vec4f(pb.xyz - p * pb.w, pb.w);
    qa = qadd(qa, invI(qa, ia3, cross(rA, p)));
    qb = qadd(qb, -invI(qb, ib3, cross(rB, p)));
    J.lin = vec4f(J.lin.xyz + p, J.lin.w);
  }
  // angular part: qB must equal qA * q0
  let qd = qmul(qmul(qa, J.q0), qconj(qb));
  var th = 2.0 * qd.xyz;
  if (qd.w < 0.0) { th = -th; }
  let ang = length(th);
  if (ang > 1e-9) {
    let n = th / ang;
    let w = dot(n, invI(qa, ia3, n)) + dot(n, invI(qb, ib3, n));
    let alpha = J.lb.w / h2;
    let dl = (ang - alpha * dot(J.angl.xyz, n)) / (w + alpha);
    var p = dl * n;
    if ((J.ids.w & 2u) != 0u) {
      // rigid-plastic joint: the transmitted moment is capped at the yield moment, beyond
      // that the joint rotates (like friction caps tangential impulse)
      let cap = J.str.z * h2 / P.breakScale;
      let tot = J.angl.xyz + p;
      let l = length(tot);
      if (l > cap) { p = tot * (cap / l) - J.angl.xyz; }
    }
    qa = qadd(qa, -invI(qa, ia3, p));
    qb = qadd(qb, invI(qb, ib3, p));
    J.angl = vec4f(J.angl.xyz + p, J.angl.w);
  }
  // static bodies may be shared by several joints of one colour: never write them
  if (pa.w > 0.0) { bodies[ia].pos = pa; bodies[ia].rot = qa; }
  if (pb.w > 0.0) { bodies[ib].pos = pb; bodies[ib].rot = qb; }
  joints[k].lin = J.lin;
  joints[k].angl = J.angl;
}

// ---------------------------------------------------------------- substep end
fn velocityBody(i: u32) {
  var b = bodies[i];
  if (b.pos.w == 0.0) { return; }
  let v = (b.pos.xyz - b.ppos.xyz) / P.h;
  let dq = qmul(b.rot, qconj(b.prot));
  var w = 2.0 * dq.xyz / P.h;
  if (dq.w < 0.0) { w = -w; }
  bodies[i].vel = vec4f(v, 0.0);
  bodies[i].ang = vec4f(w, 0.0);
}

fn clearAdj(body: u32, slot: u32) {
  if (slot >= ${ADJ}u) { return; }
  var v = shapes[body * SV + slot / 4u];
  v[slot % 4u] = NONE;
  shapes[body * SV + slot / 4u] = v;
}

// Joint failure: impulse / h^2 = force. Decompose into the joint frame and compare
// with the (anisotropic) strengths. Plastic joints (metal) yield instead of breaking:
// their rest state is moved towards the current state, until a strain limit is hit.
fn breakJoint(k: u32) {
  var J = joints[k];
  if ((J.ids.w & 1u) != 0u) { return; }
  let qa = bodies[J.ids.x].rot;
  let qb = bodies[J.ids.y].rot;
  let inv = 1.0 / (P.h * P.h);
  let F = J.lin.xyz * inv;
  let T = J.angl.xyz * inv;
  let ax = qrot(qa, J.axis.xyz);
  let fn_ = dot(F, ax);
  let fs = length(F - fn_ * ax);
  let tw = dot(T, ax);
  let tb = length(T - tw * ax);
  let rLin = select(-fn_ / J.axis.w, fn_ / J.str.x, fn_ > 0.0) + fs / J.str.y;
  let plastic = (J.ids.w & 2u) != 0u;
  var ratio = rLin + tb / J.str.z + abs(tw) / J.str.w;
  if (plastic) {
    // plastic flow: whatever rotation the capped moment let through becomes permanent
    let qrel = qmul(qconj(qa), qb);
    var q0 = J.q0;
    if (dot(q0, qrel) < 0.0) { q0 = -q0; }
    let dq = qmul(qrel, qconj(q0));
    let flow = 2.0 * length(dq.xyz);
    let capped = length(J.angl.xyz) >= 0.98 * J.str.z * P.h * P.h / P.breakScale;
    if (capped && flow > 1e-6) {
      J.lin.w += flow;
      joints[k].q0 = qrel;
    }
    ratio = rLin + select(0.0, 2.0, J.lin.w > J.angl.w) + abs(tw) / J.str.w;
  } else {
    J.lin.w = max(J.lin.w, ratio * P.breakScale); // peak load ratio (debug / stats)
  }
  // brittle failure needs the overload on two consecutive substeps: real impacts last
  // several substeps, single-substep solver noise spikes do not break anything
  var over = (J.ids.w >> 8u) & 0xffu;
  if (ratio * P.breakScale > 1.0) { over = over + 1u; } else { over = 0u; }
  if ((J.ids.w & 4u) != 0u) {
    // spread of an impact shatter (see contacts): one ring of welds per substep
    let ia = bodies[J.ids.x].info.w;
    let ib = bodies[J.ids.y].info.w;
    let la = (ia >> 8u) & 0xffu;
    let lb = (ib >> 8u) & 0xffu;
    if (max(la, lb) > 0u) {
      over = 255u;
      if (la > lb + 1u) { bodies[J.ids.y].info.w = (ib & 0xffu) | 2u | ((la - 1u) << 8u); }
      if (lb > la + 1u) { bodies[J.ids.x].info.w = (ia & 0xffu) | 2u | ((lb - 1u) << 8u); }
    }
  }
  joints[k].ids.w = (J.ids.w & 0xffu) | (min(over, 255u) << 8u);
  if (over >= select(2u, 1u, (J.ids.w & 4u) != 0u)) { // flag 4: glass-like, breaks at once
      joints[k].ids.w = J.ids.w | 1u;
      clearAdj(J.ids.x, J.ids.z & 0xffffu);
      clearAdj(J.ids.y, J.ids.z >> 16u);
      atomicAdd(&gridCount[P.hashMask + 1u], 1u);
      // mark both pieces as cracked (render shows fresh fracture surfaces)
      bodies[J.ids.x].info.w = bodies[J.ids.x].info.w | 1u;
      bodies[J.ids.y].info.w = bodies[J.ids.y].info.w | 1u;
  }
  joints[k].lin = vec4f(0.0, 0.0, 0.0, J.lin.w);
  joints[k].angl = vec4f(0.0, 0.0, 0.0, J.angl.w);
}

// substep end: velocities from positions + joint failure test (one dispatch for both)
@compute @workgroup_size(64) fn endSubstep(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i < P.nBodies) { velocityBody(i); }
  if (i < P.nJoints) { breakJoint(i); }
}

@compute @workgroup_size(64) fn exportPoses(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.nBodies) { return; }
  let b = bodies[i];
  poses[i * 2u] = vec4f(b.pos.xyz, f32(b.info.w & 1u));
  poses[i * 2u + 1u] = b.rot;
}
`;

const KERNELS = ['beginSubstep', 'insert', 'updateWorld', 'contacts', 'applyContacts', 'solveJoints', 'endSubstep', 'exportPoses'];

/** Creates and steps the GPU world. `data` comes from WorldBuilder.finish(). */
export class GpuPhysics {
  constructor(device, data, posesBuffer) {
    this.device = device;
    this.data = data;
    const nB = data.nBodies, nP = data.nParticles, nJ = data.nJoints;
    this.nBodies = nB; this.nParticles = nP; this.nJoints = nJ;
    const radius = data.radius;
    let H = 1 << 14;
    while (H < nP * 2) H <<= 1;
    this.H = H;

    this.settings = {
      substeps: 16, iterations: 1, gravity: 9.81, omega: 1.0, muS: 0.7, muK: 0.5,
      breakScale: 1.0, maxDepen: 0.6, linDamp: 0.9995, angDamp: 0.999, maxSpeed: 80, slop: 0.1, jointIters: 2, shatterV: 3,
    };

    const S = GPUBufferUsage;
    const mk = (size, usage, label) => device.createBuffer({ size: Math.max(16, size), usage, label });
    this.paramBuf = mk(256, S.UNIFORM | S.COPY_DST, 'params');
    this.staticBuf = mk(MAX_STATICS * 48, S.UNIFORM | S.COPY_DST, 'statics');
    this.bodyBuf = mk(nB * BODY_FLOATS * 4, S.STORAGE | S.COPY_DST | S.COPY_SRC, 'bodies');
    this.particleBuf = mk(nP * PARTICLE_FLOATS * 4, S.STORAGE | S.COPY_DST | S.COPY_SRC, 'particles');
    this.jointBuf = mk(Math.max(1, nJ) * JOINT_FLOATS * 4, S.STORAGE | S.COPY_DST | S.COPY_SRC, 'joints');
    this.shapeBuf = mk(nB * SHAPE_VEC4 * 16, S.STORAGE | S.COPY_DST, 'shapes');
    this.gridCountBuf = mk((H + STATS) * 4, S.STORAGE | S.COPY_DST | S.COPY_SRC, 'gridCount');
    this.gridSlotBuf = mk(H * SLOTS * 4, S.STORAGE, 'gridSlots');
    this.posesBuf = posesBuffer;
    this.statsRead = mk(STATS * 4, S.MAP_READ | S.COPY_DST, 'statsRead');

    const q = device.queue;
    q.writeBuffer(this.bodyBuf, 0, data.bodies);
    q.writeBuffer(this.particleBuf, 0, data.particles);
    if (nJ) q.writeBuffer(this.jointBuf, 0, data.joints);
    q.writeBuffer(this.shapeBuf, 0, data.shapes);
    q.writeBuffer(this.gridCountBuf, 0, new Uint32Array(H + STATS));
    this.setStatics(data.statics);

    // colour ranges as dynamic-offset uniforms (one 256-byte slot per colour)
    const colors = data.colorRanges;
    this.colorBuf = mk(Math.max(1, colors.length) * 256, S.UNIFORM | S.COPY_DST, 'colors');
    const cr = new Uint32Array(Math.max(1, colors.length) * 64);
    colors.forEach(([s, c], i) => { cr[i * 64] = s; cr[i * 64 + 1] = c; });
    q.writeBuffer(this.colorBuf, 0, cr);
    this.colors = colors;

    this.radius = radius;
    this.writeParams();
    this.stats = { broken: 0 };
    this.readPending = false;
  }

  async init() {
    const device = this.device;
    const module = device.createShaderModule({ code: WGSL, label: 'tetrafract-v2' });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    for (const m of info.messages) console[m.type === 'error' ? 'error' : 'warn'](`WGSL ${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
    if (errs.length) throw new Error('WGSL compile failed: ' + errs[0].message);

    const st = { type: 'storage' };
    const C = GPUShaderStage.COMPUTE;
    this.bgl0 = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: C, buffer: { type: 'uniform' } },
      { binding: 1, visibility: C, buffer: { type: 'uniform' } },
      ...[2, 3, 4, 5, 6, 7, 8].map((binding) => ({ binding, visibility: C, buffer: st })),
    ] });
    this.bgl1 = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 8 } },
    ] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl0, this.bgl1] });
    device.pushErrorScope('validation');
    this.pipes = {};
    for (const k of KERNELS) {
      this.pipes[k] = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: k }, label: k });
    }
    this.bg0 = device.createBindGroup({ layout: this.bgl0, entries: [
      this.paramBuf, this.staticBuf, this.bodyBuf, this.particleBuf, this.jointBuf, this.shapeBuf, this.gridCountBuf, this.gridSlotBuf, this.posesBuf,
    ].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.bg1 = device.createBindGroup({ layout: this.bgl1, entries: [{ binding: 0, resource: { buffer: this.colorBuf, size: 8 } }] });
    const err = await device.popErrorScope();
    if (err) throw new Error('Pipeline validation: ' + err.message);
    // export initial poses
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.bg0);
    pass.setBindGroup(1, this.bg1, [0]);
    this.dispatch(pass, 'exportPoses', this.nBodies);
    pass.end();
    device.queue.submit([enc.finish()]);
    return this;
  }

  setStatics(list) {
    const a = new Float32Array(MAX_STATICS * 12);
    list.slice(0, MAX_STATICS).forEach((s, i) => {
      a.set([s.center[0], s.center[1], s.center[2], 0, s.quat[0], s.quat[1], s.quat[2], s.quat[3], s.half[0], s.half[1], s.half[2], s.friction ?? 1], i * 12);
    });
    this.nStatics = Math.min(list.length, MAX_STATICS);
    this.device.queue.writeBuffer(this.staticBuf, 0, a);
  }

  writeParams() {
    const s = this.settings;
    const b = new ArrayBuffer(96);
    const f = new Float32Array(b), u = new Uint32Array(b);
    const h = (1 / 60) / s.substeps;
    f[0] = h; f[1] = s.gravity; f[2] = this.radius; f[3] = this.radius * 2;
    u[4] = this.nBodies; u[5] = this.nParticles; u[6] = this.nJoints; u[7] = this.H - 1;
    f[8] = s.omega; f[9] = s.muS; f[10] = s.muK; f[11] = Math.pow(s.linDamp, 1);
    u[12] = this.nStatics ?? 0; f[13] = this.data.groundY ?? 0; f[14] = s.breakScale; f[15] = s.maxDepen;
    f[16] = s.angDamp; f[17] = s.maxSpeed; f[18] = 0; f[19] = s.slop;
    f[20] = s.shatterV;
    this.device.queue.writeBuffer(this.paramBuf, 0, b);
    this.h = h;
  }

  dispatch(pass, k, n) {
    pass.setPipeline(this.pipes[k]);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
  }

  /** Encodes one 1/60 s frame (all substeps) into the encoder. dtScale < 1 = slow motion. */
  encodeFrame(enc, timeScale = 1) {
    const s = this.settings;
    // slow motion: same substep size, fewer substeps per displayed frame
    const n = Math.max(1, Math.round(s.substeps * timeScale));
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.bg0);
    pass.setBindGroup(1, this.bg1, [0]);
    for (let i = 0; i < n; i++) {
      this.dispatch(pass, 'beginSubstep', Math.max(this.nBodies, this.H));
      this.dispatch(pass, 'insert', this.nParticles);
      for (let it = 0; it < s.iterations; it++) {
        if (it > 0) this.dispatch(pass, 'updateWorld', this.nParticles);
        this.dispatch(pass, 'contacts', this.nParticles);
        this.dispatch(pass, 'applyContacts', this.nBodies);
        if (this.nJoints) {
          pass.setPipeline(this.pipes.solveJoints);
          for (let r = 0; r < s.jointIters; r++) {
            // alternate the colour order between sweeps (symmetric Gauss-Seidel)
            for (let k = 0; k < this.colors.length; k++) {
              const c = r % 2 ? this.colors.length - 1 - k : k;
              pass.setBindGroup(1, this.bg1, [c * 256]);
              pass.dispatchWorkgroups(Math.ceil(this.colors[c][1] / 64));
            }
          }
        }
      }
      this.dispatch(pass, 'endSubstep', Math.max(this.nBodies, this.nJoints));
    }
    this.dispatch(pass, 'exportPoses', this.nBodies);
    pass.end();
    this.simTime = (this.simTime ?? 0) + n * this.h;
  }

  /** Non-blocking stats readback (broken joint count). */
  requestStats(enc) {
    if (this.readPending) return false;
    enc.copyBufferToBuffer(this.gridCountBuf, this.H * 4, this.statsRead, 0, STATS * 4);
    return true;
  }
  afterSubmit(didRequest) {
    if (!didRequest) return;
    this.readPending = true;
    this.statsRead.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Uint32Array(this.statsRead.getMappedRange().slice(0));
      this.statsRead.unmap();
      this.stats.broken = a[0];
      this.readPending = false;
    }).catch(() => {});
  }

  /** Overwrites one body's state (used to launch projectiles / boulders). */
  setBody(i, { pos, quat = [0, 0, 0, 1], vel = [0, 0, 0], ang = [0, 0, 0], invMass }) {
    const f = new Float32Array(24);
    f.set([pos[0], pos[1], pos[2], invMass, quat[0], quat[1], quat[2], quat[3]], 0);
    f.set([pos[0], pos[1], pos[2], invMass, quat[0], quat[1], quat[2], quat[3]], 8);
    f.set([vel[0], vel[1], vel[2], 0, ang[0], ang[1], ang[2], 0], 16);
    this.device.queue.writeBuffer(this.bodyBuf, i * BODY_FLOATS * 4, f);
  }

  /** Full readback of a buffer (tests only). */
  async read(buffer, size) {
    const tmp = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, tmp, 0, size);
    this.device.queue.submit([enc.finish()]);
    await tmp.mapAsync(GPUMapMode.READ);
    const out = tmp.getMappedRange().slice(0);
    tmp.unmap(); tmp.destroy();
    return out;
  }
  async readBodies() { return new Float32Array(await this.read(this.bodyBuf, this.nBodies * BODY_FLOATS * 4)); }
  async readJoints() { return new Uint32Array(await this.read(this.jointBuf, Math.max(1, this.nJoints) * JOINT_FLOATS * 4)); }

  destroy() {
    for (const b of [this.paramBuf, this.staticBuf, this.bodyBuf, this.particleBuf, this.jointBuf, this.shapeBuf, this.gridCountBuf, this.gridSlotBuf, this.colorBuf]) b.destroy();
    try { this.statsRead.destroy(); } catch {}
  }
}
