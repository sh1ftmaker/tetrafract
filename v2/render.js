// Rendering: one merged mesh per material. Every vertex stores its body index and its
// position/normal in the body frame; the vertex shader fetches the body pose written
// by the physics (`exportPoses`) from a storage buffer shared with the solver.

import * as THREE from 'three/webgpu';
import {
  Fn, attribute, storage, varying, color, positionWorld, vec2, vec3, vec4, float, int, normalLocal, positionLocal, mix, smoothstep, fract, abs, sin,
  mx_noise_float, mx_fractal_noise_float, clamp, length, pow, normalView, positionViewDirection, dot, max, step, mod, uniform, select,
} from 'three/tsl';
import { MAT } from './builder.js';

/** Storage attribute holding [pos, quat] per body; its GPUBuffer is written by the solver. */
export function createPoseBuffer(renderer, nBodies) {
  const attr = new THREE.StorageBufferAttribute(new Float32Array(nBodies * 8), 4);
  renderer.backend.createStorageAttribute(attr);
  const gpu = renderer.backend.get(attr).buffer;
  const node = storage(attr, 'vec4', nBodies * 2).toReadOnly();
  return { attr, gpu, node };
}

const rotate = (q, v) => {
  const t = q.xyz.cross(v).mul(2);
  return v.add(t.mul(q.w)).add(q.xyz.cross(t));
};

/** 1 once the body lost any weld (fresh fracture faces become visible). */
const crackedOf = (poses) => varying(poses.element(int(attribute('body', 'vec2').x).mul(2)).w);

function bodyTransform(material, poses) {
  material.positionNode = Fn(() => {
    const bi = int(attribute('body', 'vec2').x);
    const p = poses.element(bi.mul(2));
    const q = poses.element(bi.mul(2).add(1));
    normalLocal.assign(rotate(q, normalLocal));
    return rotate(q, positionLocal).add(p.xyz);
  })();
}

// --------------------------------------------------------------------- materials
function woodMaterial(poses) {
  const m = new THREE.MeshStandardNodeMaterial({ roughness: 0.85, metalness: 0 });
  bodyTransform(m, poses);
  const tex = attribute('tex', 'vec4');
  const tint = attribute('body', 'vec2').y;
  m.colorNode = Fn(() => {
    const t = tex.xyz;
    const interior = step(0.5, mod(tex.w, 2)).mul(crackedOf(poses));
    // growth rings: distance from a pith line running along the grain, warped by noise
    const across = t.yz.add(vec2(0.9, 0.7));
    const warp = mx_noise_float(t.mul(vec3(0.35, 2.5, 2.5))).mul(0.06);
    const ring = fract(length(across).add(warp).mul(28));
    const ringMask = smoothstep(0.55, 0.95, ring);
    // fine streaks along the grain
    const streak = mx_noise_float(vec3(t.x.mul(0.6), t.y.mul(60), t.z.mul(60))).mul(0.5).add(0.5);
    const blotch = mx_fractal_noise_float(t.mul(vec3(0.4, 3, 3)), 3, 2, 0.5).mul(0.5).add(0.5);
    // weathered timber (grey-brown) with per-member variation
    const base = mix(color(0x6b5843), color(0x8e7a5f), tint);
    let c = mix(base, base.mul(0.6), ringMask.mul(0.75));
    c = c.mul(streak.mul(0.35).add(0.75)).mul(blotch.mul(0.3).add(0.8));
    // fresh break: pale, fibrous, yellow-orange
    const fresh = mix(color(0xd2ad78), color(0xe9d1a2), streak).mul(ringMask.mul(-0.15).add(1));
    return vec4(mix(c, fresh, interior), 1);
  })();
  m.roughnessNode = float(0.85);
  return m;
}

function glassMaterial(poses) {
  const m = new THREE.MeshStandardNodeMaterial({ roughness: 0.05, metalness: 0.1, transparent: true, depthWrite: false, side: THREE.DoubleSide });
  bodyTransform(m, poses);
  const tex = attribute('tex', 'vec4');
  const interior = step(0.5, mod(tex.w, 2));
  // latent crack faces stay invisible until the shard has actually cracked
  const rim = step(1.5, mod(tex.w, 4)).mul(float(1).sub(interior.mul(float(1).sub(crackedOf(poses)))));
  const hidden = interior.mul(float(1).sub(crackedOf(poses)));
  m.colorNode = mix(vec3(0.55, 0.75, 0.72), vec3(0.8, 0.97, 0.93), rim);
  // Fresnel: more opaque at grazing angles; fracture edges bright and almost opaque
  const fres = pow(float(1).sub(abs(dot(normalView, positionViewDirection))), 3);
  // window glare: diagonal sheen bands fixed to the pane, so cracks show as breaks in them
  const band = sin(tex.x.add(tex.y.mul(0.55)).mul(2.6));
  const sheen = smoothstep(0.55, 0.95, band).mul(0.22).add(smoothstep(0.97, 1.0, band).mul(0.15));
  m.opacityNode = mix(fres.mul(0.5).add(0.1).add(sheen), float(0.9), rim).mul(float(1).sub(hidden));
  m.emissiveNode = vec3(0.25, 0.42, 0.4).mul(rim).add(vec3(0.08, 0.12, 0.12).mul(fres)).add(vec3(0.55, 0.62, 0.62).mul(sheen));
  return m;
}

function stoneMaterial(poses) {
  const m = new THREE.MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 });
  bodyTransform(m, poses);
  const tex = attribute('tex', 'vec4');
  const tint = attribute('body', 'vec2').y;
  m.colorNode = Fn(() => {
    const interior = step(0.5, mod(tex.w, 2)).mul(crackedOf(poses));
    const n1 = mx_fractal_noise_float(tex.xyz.mul(3), 4, 2, 0.5).mul(0.5).add(0.5);
    const n2 = mx_noise_float(tex.xyz.mul(40)).mul(0.5).add(0.5);
    const base = mix(color(0x7c766c), color(0x968d7e), tint);
    const outside = base.mul(n1.mul(0.45).add(0.72)).mul(n2.mul(0.2).add(0.9));
    const inside = mix(color(0xb3ab9c), color(0xcfc7b8), n2).mul(n1.mul(0.2).add(0.85));
    // interior and exterior both cracked: show fresh stone only where it really broke
    return vec4(mix(outside, inside, interior), 1);
  })();
  return m;
}

function metalMaterial(poses) {
  const m = new THREE.MeshStandardNodeMaterial({ color: 0x8a9099, roughness: 0.45, metalness: 0.45 });
  bodyTransform(m, poses);
  return m;
}

const FACTORY = { [MAT.WOOD]: woodMaterial, [MAT.GLASS]: glassMaterial, [MAT.STONE]: stoneMaterial, [MAT.METAL]: metalMaterial };

/** Builds a THREE.Group with one mesh per material from the builder's body list. */
export function buildMeshes(bodyList, poses) {
  const group = new THREE.Group();
  const meshes = new Map();
  group.userData.meshes = meshes;
  const byMat = new Map();
  for (const b of bodyList) {
    if (!byMat.has(b.material)) byMat.set(b.material, []);
    byMat.get(b.material).push(b);
  }
  for (const [mat, list] of byMat) {
    let nv = 0;
    for (const b of list) nv += b.verts.length;
    const pos = new Float32Array(nv * 3), nrm = new Float32Array(nv * 3), tex = new Float32Array(nv * 4), body = new Float32Array(nv * 2);
    let i = 0;
    for (const b of list) {
      b.vstart = i;
      for (const v of b.verts) {
        pos.set(v.p, i * 3); nrm.set(v.n, i * 3); tex.set([...v.t, v.flags], i * 4); body.set([b.id, b.tint], i * 2);
        i++;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    g.setAttribute('tex', new THREE.BufferAttribute(tex, 4));
    g.setAttribute('body', new THREE.BufferAttribute(body, 2));
    const mesh = new THREE.Mesh(g, FACTORY[mat](poses));
    mesh.frustumCulled = false;
    mesh.castShadow = mat !== MAT.GLASS;
    mesh.receiveShadow = true;
    if (mat === MAT.GLASS) mesh.renderOrder = 2;
    group.add(mesh);
    meshes.set(mat, mesh);
  }
  return group;
}

export function buildStatics(statics) {
  const group = new THREE.Group();
  for (const s of statics) {
    if (!s.visible) continue;
    const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.95 });
    // rock: layered noise in world space
    const p = positionWorld;
    const n1 = mx_fractal_noise_float(p.mul(vec3(0.6, 1.8, 0.6)), 4, 2, 0.5).mul(0.5).add(0.5);
    const n2 = mx_noise_float(p.mul(9)).mul(0.5).add(0.5);
    mat.colorNode = color(s.color).mul(n1.mul(0.5).add(0.7)).mul(n2.mul(0.15).add(0.92));
    const m = new THREE.Mesh(new THREE.BoxGeometry(s.half[0] * 2, s.half[1] * 2, s.half[2] * 2), mat);
    m.position.fromArray(s.center);
    m.quaternion.fromArray(s.quat);
    m.castShadow = m.receiveShadow = true;
    group.add(m);
  }
  return group;
}

export function disposeGroup(g) {
  g.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
}
