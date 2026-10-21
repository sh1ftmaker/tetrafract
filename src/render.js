// Renders every tet face straight from the simulation's GPU buffers. Faces
// shared by two still-connected tets collapse to nothing in the vertex shader;
// once the GPU flags a face as broken it pops into view as fresh crack surface.
//
// Shading is procedural in rest space, so a material's grain, strata and speckle run
// straight through a piece and show on every fresh fracture face. A screen-space bump
// (one noise sample, derivatives for the gradient) roughens the flat tet facets so
// broken surfaces read as torn fibres or granular stone rather than polished crystal.

import * as THREE from 'three/webgpu';
import {
  Fn, attribute, varying, vec2, vec3, vec4, float, int, select, cross, normalize, mix, smoothstep, clamp,
  mx_noise_float, mx_fractal_noise_float, uniform, transformNormalToView, abs, fract, max, dot, sign, sin, pow,
  dFdx, dFdy, positionView, instanceIndex, uv, length,
} from 'three/tsl';
import { MAT, TET_FACES } from './builder.js';
import { MAX_PARTICLES } from './sim.js';

export const showTets = uniform(0);
const DYING = 0.35; // must match sim.js

export function createMeshes(sim) {
  const m = sim.mesh;
  const groups = new Map(); // material -> vertex data
  for (let t = 0; t < m.T; t++) {
    const mat = m.tetMat[t];
    if (!groups.has(mat)) groups.set(mat, { meta: [], rest: [] });
    const g = groups.get(mat);
    for (let f = 0; f < 4; f++) {
      const tri = TET_FACES[f];
      for (let r = 0; r < 3; r++) {
        // this corner first, then the other two in winding order
        const k0 = tri[r], k1 = tri[(r + 1) % 3], k2 = tri[(r + 2) % 3];
        g.meta.push(t, k0 | (k1 << 2) | (k2 << 4) | (f << 6) | (m.tetGrain[t] << 8));
        const n = m.tetNodes[t * 4 + k0];
        g.rest.push(m.nodePos[n * 3], m.nodePos[n * 3 + 1], m.nodePos[n * 3 + 2]);
      }
    }
  }
  const meshes = [];
  for (const [mat, g] of groups) {
    const geo = new THREE.BufferGeometry();
    const count = g.meta.length / 2;
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    geo.setAttribute('tetMeta', new THREE.BufferAttribute(new Uint32Array(g.meta), 2));
    geo.setAttribute('restPos', new THREE.BufferAttribute(new Float32Array(g.rest), 3));
    const mesh = new THREE.Mesh(geo, makeMaterial(sim, mat));
    mesh.frustumCulled = false;
    mesh.castShadow = mat !== MAT.GLASS;
    mesh.receiveShadow = true;
    if (mat === MAT.GLASS) mesh.renderOrder = 2;
    meshes.push(mesh);
  }
  meshes.push(createParticles(sim));
  return meshes;
}

/** View-space normal of the flat facet, perturbed by the screen-space gradient of h. */
function bumped(nWorld, h, strength) {
  const N = transformNormalToView(nWorld).normalize().toVar();
  const dpx = dFdx(positionView), dpy = dFdy(positionView);
  const dhx = dFdx(h), dhy = dFdy(h);
  const r1 = cross(dpy, N), r2 = cross(N, dpx);
  const det = dot(dpx, r1);
  const grad = sign(det).mul(r1.mul(dhx).add(r2.mul(dhy)));
  return normalize(abs(det).mul(N).sub(grad.mul(strength)));
}

function makeMaterial(sim, mat) {
  const material = mat === MAT.GLASS || mat === MAT.JELLY
    ? new THREE.MeshPhysicalNodeMaterial()
    : new THREE.MeshStandardNodeMaterial();

  const vNormal = varying(vec3(0, 1, 0), 'vNrm');
  const vRest = varying(vec3(0), 'vRest');
  const vCrack = varying(float(0), 'vCrack');
  const vGrain = varying(float(0), 'vGrain');
  const { verts, cornerVertex, slotFace, faceState, tetAge } = sim;

  material.positionNode = Fn(() => {
    const meta = attribute('tetMeta', 'uvec2');
    const t = int(meta.x).toVar();
    const bits = int(meta.y).toVar();
    const k0 = bits.mod(4), k1 = bits.div(4).mod(4), k2 = bits.div(16).mod(4);
    const f = bits.div(64).mod(4);
    const base = t.mul(4);
    const at = (k) => verts.element(cornerVertex.element(base.add(k)).mul(4)).xyz.toVar();
    const p0 = at(0), p1 = at(1), p2 = at(2), p3 = at(3);
    const pick = (k) => select(k.equal(0), p0, select(k.equal(1), p1, select(k.equal(2), p2, p3)));
    const a = pick(k0).toVar(), b = pick(k1).toVar(), c = pick(k2).toVar();
    vNormal.assign(normalize(cross(b.sub(a), c.sub(a))));

    const face = slotFace.element(t.mul(4).add(f)).toVar();
    const interior = face.greaterThanEqual(0);
    const broken = faceState.element(max(face, int(0))).greaterThan(0);
    const age = tetAge.element(t).toVar();
    const visible = face.lessThan(0).or(broken).or(showTets.greaterThan(0.5)).and(age.greaterThanEqual(0));
    vCrack.assign(select(interior, float(1), float(0)));
    vRest.assign(attribute('restPos', 'vec3'));
    vGrain.assign(float(bits.div(256)));

    // retiring debris shrinks to nothing before it leaves the solver
    const shrink = select(age.greaterThanEqual(100), clamp(float(100 + DYING).sub(age).div(DYING), 0, 1), float(1));
    const cent = p0.add(p1).add(p2).add(p3).mul(0.25);
    const s = mix(float(1), float(0.78), showTets).mul(shrink);
    return select(visible, cent.add(a.sub(cent).mul(s)), vec3(0, -1000, 0));
  })();

  const crack = vCrack;
  // coordinates with the grain axis first: (along, across1, across2)
  const grainSpace = () => {
    const g = vGrain, p = vRest;
    const along = select(g.equal(0), p.x, select(g.equal(1), p.y, p.z));
    const u = select(g.equal(0), p.y, p.x), w = select(g.equal(2), p.y, p.z);
    return vec3(along, u, w);
  };

  if (mat === MAT.WOOD) {
    const q = grainSpace();
    // long fibres: fast across the grain, slow along it
    const fib = mx_noise_float(vec3(q.x.mul(2.5), q.y.mul(70), q.z.mul(70))).mul(0.5).add(0.5);
    material.colorNode = Fn(() => {
      const n = mx_fractal_noise_float(vec3(q.y.mul(9), q.z.mul(9), q.x.mul(0.6)), 3, 2.0, 0.5).mul(0.5).add(0.5);
      const rings = smoothstep(0.15, 0.85, fract(n.mul(6.0)));
      const dark = vec3(0.34, 0.21, 0.1), light = vec3(0.64, 0.45, 0.26);
      const outer = mix(dark, light, rings.mul(0.7).add(fib.mul(0.3)));
      // fresh wood: same rings, paler and fibrous, with dark gaps between torn fibres
      const fresh = mix(vec3(0.62, 0.45, 0.26), vec3(0.9, 0.76, 0.54), rings.mul(0.5).add(fib.mul(0.5))).mul(fib.mul(0.35).add(0.68));
      return vec4(mix(outer, fresh, crack), 1);
    })();
    material.normalNode = bumped(vNormal, fib, mix(float(0.004), float(0.02), crack));
    material.roughness = 0.8;
  } else if (mat === MAT.GLASS) {
    material.colorNode = vec4(mix(vec3(0.75, 0.92, 0.95), vec3(0.88, 1.0, 1.0), crack), 1);
    material.opacityNode = mix(float(0.22), float(0.6), crack);
    material.normalNode = transformNormalToView(vNormal);
    material.transparent = true;
    material.depthWrite = false;
    material.side = THREE.DoubleSide;
    material.roughness = 0.03;
    material.metalness = 0.0;
    material.specularIntensity = 1;
    material.envMapIntensity = 1.6;
  } else if (mat === MAT.STEEL) {
    const q = grainSpace();
    const brush = mx_noise_float(vec3(q.x.mul(1.5), q.y.mul(120), q.z.mul(120))).mul(0.5).add(0.5);
    const grit = mx_noise_float(vRest.mul(90)).mul(0.5).add(0.5);
    material.colorNode = vec4(mix(vec3(0.62, 0.64, 0.68).add(brush.mul(0.1)), vec3(0.85, 0.85, 0.87).mul(grit.mul(0.3).add(0.8)), crack), 1);
    material.roughnessNode = mix(brush.mul(0.18).add(0.26), grit.mul(0.3).add(0.4), crack);
    material.normalNode = bumped(vNormal, mix(brush, grit, crack), mix(float(0.002), float(0.012), crack));
    material.metalness = 0.65;
  } else if (mat === MAT.STONE || mat === MAT.ROCK) {
    const rock = mat === MAT.ROCK;
    const base = rock ? vec3(0.42, 0.39, 0.36) : vec3(0.62, 0.61, 0.58);
    const grain = mx_noise_float(vRest.mul(rock ? 18 : 26)).mul(0.5).add(0.5);
    material.colorNode = Fn(() => {
      const n = mx_fractal_noise_float(vRest.mul(rock ? 2.5 : 5), 3, 2.0, 0.5).mul(0.5).add(0.5);
      const speck = smoothstep(0.62, 0.74, grain).mul(0.12);
      const outer = base.mul(n.mul(0.35).add(0.75)).sub(speck);
      // fresh break: paler aggregate, darker in the pits so the facet looks granular
      const fresh = (rock ? vec3(0.6, 0.57, 0.52) : vec3(0.74, 0.72, 0.68)).mul(n.mul(0.15).add(0.85)).mul(grain.mul(0.35).add(0.72)).sub(speck);
      return vec4(mix(outer, fresh, crack), 1);
    })();
    material.normalNode = bumped(vNormal, grain, mix(float(0.003), float(0.022), crack));
    material.roughness = 0.95;
  } else if (mat === MAT.JELLY) {
    material.normalNode = transformNormalToView(vNormal);
    material.colorNode = vec4(0.25, 0.85, 0.35, 1);
    material.transmission = 0.0;
    material.opacity = 0.85;
    material.transparent = true;
    material.roughness = 0.15;
    material.clearcoat = 1;
  }
  return material;
}

/**
 * Debris particles as camera-facing sprites read straight from the sim's ring buffer:
 * stone dust puffs that swell and fade, spinning wood splinters, glass slivers, sparks.
 */
function createParticles(sim) {
  const P = sim.particles;
  const material = new THREE.SpriteNodeMaterial();
  const vInfo = varying(vec4(0), 'vPart'); // kind, seed, age, life
  material.positionNode = Fn(() => {
    const a = P.element(instanceIndex.mul(2));
    const kw = P.element(instanceIndex.mul(2).add(1)).w;
    const kind = kw.floor(), seed = fract(kw);
    const dust = kind.equal(3).or(kind.equal(5));
    const life = select(dust, float(2.2), select(kind.equal(2), float(0.6), select(kind.equal(1), float(2.5), float(3.5))));
    vInfo.assign(vec4(kind, seed, a.w, life));
    return a.xyz;
  })();
  // round: interpolated varyings can arrive as 2.9999998 on some GPUs, and kind is compared exactly
  const kind = vInfo.x.add(0.5).floor(), seed = vInfo.y, age = vInfo.z, life = vInfo.w;
  const dust = kind.equal(3).or(kind.equal(5));
  const alive = age.lessThan(life);
  material.scaleNode = Fn(() => {
    const k = seed.mul(0.8).add(0.6);
    const s = select(dust, vec2(age.mul(0.55).add(0.12)),
      select(kind.equal(0), vec2(0.016, 0.055),
        select(kind.equal(2), vec2(0.008, 0.045), vec2(0.014, 0.026))));
    return select(alive, s.mul(k), vec2(0));
  })();
  material.rotationNode = seed.mul(6.28).add(age.mul(seed.sub(0.5)).mul(select(dust, float(0.6), float(14))));
  material.colorNode = Fn(() => {
    const c = select(dust, select(kind.equal(5), vec3(0.5, 0.47, 0.43), vec3(0.68, 0.66, 0.62)),
      select(kind.equal(0), vec3(0.8, 0.62, 0.4).mul(seed.mul(0.4).add(0.6)),
        select(kind.equal(2), vec3(1.0, 0.6, 0.2).mul(2.5),
          vec3(0.62, 0.74, 0.72).mul(seed.mul(0.25).add(0.8)))));
    return vec4(c, 1);
  })();
  material.opacityNode = Fn(() => {
    const d = length(uv().sub(0.5));
    const fade = clamp(life.sub(age).div(select(dust, life, float(0.6))), 0, 1);
    const puff = float(1).sub(smoothstep(0.15, 0.5, d)).mul(0.3).mul(pow(fade, 1.5));
    // glass: a thin translucent triangular sliver, a little denser along its edges
    const tri = uv().y.sub(abs(uv().x.sub(0.5)).mul(2)).toVar();
    const sliver = select(tri.greaterThan(0), mix(float(0.6), float(0.3), smoothstep(0.0, 0.25, tri)), float(0));
    const solid = select(kind.equal(1), sliver, float(1)).mul(fade);
    return select(dust, puff, solid);
  })();
  material.transparent = true;
  material.depthWrite = false;
  const sprite = new THREE.Sprite(material);
  sprite.count = MAX_PARTICLES;
  sprite.frustumCulled = false;
  sprite.renderOrder = 3;
  return sprite;
}
