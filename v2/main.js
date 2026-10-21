// App shell: renderer, camera, UI, frame loop, projectile firing and test hooks.
import * as THREE from 'three/webgpu';
import { Fn, vec3, vec4, float, color, positionWorld, positionWorldDirection, mix, smoothstep, length, pow, max, mx_fractal_noise_float } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GpuPhysics, BODY_FLOATS } from './physics.js';
import { createPoseBuffer, buildMeshes, buildStatics, disposeGroup } from './render.js';
import { SCENES, buildScene } from './scenes.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const OFFSCREEN = params.has('offscreen');
const MANUAL = params.has('manual');          // tests drive the simulation explicitly
const FIXED = params.has('fixed') ? +params.get('fixed') : 0; // substeps per frame override

function fail(msg) {
  $('error').textContent = msg;
  $('error').hidden = false;
  $('loading').hidden = true;
  throw new Error(msg);
}
if (!navigator.gpu) fail('This demo needs WebGPU (desktop Chrome / Edge 113+).\nThe destruction physics runs in WebGPU compute shaders.');

// Quality: phones / tablets get the "lite" preset (coarser pieces, fewer substeps,
// 1x pixel ratio, smaller shadow map). Override with ?q=lite or ?q=hd.
const MOBILE = navigator.userAgentData?.mobile ?? /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
const LITE = params.has('q') ? params.get('q') === 'lite' : MOBILE;
const PRESET = LITE ? { substeps: 10, minSubsteps: 6, dpr: 1, shadow: 1024, aa: false } : { substeps: 16, minSubsteps: 10, dpr: Math.min(devicePixelRatio, 2), shadow: 2048, aa: true };

const renderer = new THREE.WebGPURenderer({ antialias: PRESET.aa && !OFFSCREEN, powerPreference: 'high-performance' });
await renderer.init();
if (!renderer.backend.isWebGPUBackend) fail('WebGPU backend unavailable – the physics needs compute shaders.');
const device = renderer.backend.device;
device.lost.then((i) => console.error('device lost', i.reason, i.message));
device.addEventListener?.('uncapturederror', (e) => console.error('GPU error:', e.error.message));

const W0 = +(params.get('w') ?? innerWidth), H0 = +(params.get('h') ?? innerHeight);
renderer.setPixelRatio(OFFSCREEN ? 1 : PRESET.dpr);
renderer.setSize(W0, H0);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
if (!OFFSCREEN) document.body.prepend(renderer.domElement);
const rt = OFFSCREEN ? new THREE.RenderTarget(W0, H0, { samples: 1 }) : null;

const scene = new THREE.Scene();
// sky: horizon haze to blue zenith
scene.backgroundNode = mix(color(0xd4dbe0), color(0x5f86b3), pow(max(positionWorldDirection.y, 0), 0.6));
scene.fog = new THREE.Fog(0xc9d2d8, 35, 90);
{
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;
}
const camera = new THREE.PerspectiveCamera(50, W0 / H0, 0.05, 300);
const controls = new OrbitControls(camera, OFFSCREEN ? document.body : renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;

scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x5d5040, 1.3));
const sun = new THREE.DirectionalLight(0xfff0dc, 2.8);
sun.position.set(-8, 16, 10);
sun.castShadow = true;
sun.shadow.mapSize.set(PRESET.shadow, PRESET.shadow);
Object.assign(sun.shadow.camera, { left: -14, right: 14, top: 14, bottom: -14, near: 1, far: 60 });
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.03;
scene.add(sun, sun.target);

// ground: dry dirt with a little grass noise
{
  const mat = new THREE.MeshStandardNodeMaterial({ roughness: 1 });
  mat.colorNode = Fn(() => {
    const p = positionWorld.xz;
    const n = mx_fractal_noise_float(vec3(p.x, p.y, 0).mul(0.35), 4, 2, 0.5).mul(0.5).add(0.5);
    const n2 = mx_fractal_noise_float(vec3(p.x, p.y, 3).mul(3), 2, 2, 0.5).mul(0.5).add(0.5);
    const dirt = mix(color(0x6f6250), color(0x5f6446), n).mul(n2.mul(0.25).add(0.85));
    return vec4(dirt, 1);
  })();
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(300, 300), mat);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);
}

// ------------------------------------------------------------------ state
const state = {
  name: params.get('scene') ?? 'trestle',
  sim: null, meta: null, data: null, group: null, statics: null, poses: null,
  slow: false, paused: false, lite: LITE, substeps: FIXED || PRESET.substeps, gpuBound: 0, nextProj: 0, inflight: 0, frames: 0, simFrames: 0,
  events: [],
  fire: (...a) => fire(...a),
};

async function loadScene(name) {
  $('loading').hidden = false;
  // Detach the running sim first so the animation loop stops encoding work for it
  // while we wait for the GPU and free its buffers.
  const token = (state.loadToken = (state.loadToken ?? 0) + 1);
  const old = state.sim;
  state.sim = null;
  await new Promise((r) => setTimeout(r, 0));
  if (state.group) { scene.remove(state.group); disposeGroup(state.group); state.group = null; }
  if (state.statics) { scene.remove(state.statics); disposeGroup(state.statics); state.statics = null; }
  // wait for in-flight GPU work before freeing buffers
  await device.queue.onSubmittedWorkDone();
  old?.destroy();
  const oldPoses = state.poses;
  state.poses = null;
  if (oldPoses?.attr) renderer.backend.destroyAttribute?.(oldPoses.attr);

  const opt = { small: params.has('small'), lite: state.lite, projR: state.userSize ? +$('size').value : undefined };
  for (const [k, v] of params) if (k.startsWith('o_')) opt[k.slice(2)] = isNaN(+v) ? v : +v;
  const t0 = performance.now();
  const { data, meta } = buildScene(name, opt);
  const tBuild = performance.now() - t0;
  const poses = createPoseBuffer(renderer, data.nBodies);
  const sim = new GpuPhysics(device, data, poses.gpu);
  sim.settings.substeps = state.substeps;
  if (params.has('iters')) sim.settings.iterations = +params.get('iters');
  sim.settings.breakScale = Math.pow(2, -$('tough').value);
  Object.assign(sim.settings, meta.settings ?? {});
  for (const [k, v] of params) if (k.startsWith('s_')) sim.settings[k.slice(2)] = +v;
  sim.writeParams();
  await sim.init();
  if (token !== state.loadToken) { // a newer reset started meanwhile; it owns the scene
    await device.queue.onSubmittedWorkDone();
    sim.destroy();
    if (poses.attr) renderer.backend.destroyAttribute?.(poses.attr);
    return;
  }
  state.name = name; state.sim = sim; state.meta = meta; state.data = data; state.poses = poses;
  state.group = buildMeshes(data.bodyList, poses.node);
  state.statics = buildStatics(data.statics);
  scene.add(state.group, state.statics);
  state.nextProj = 0;
  state.adapt = null;
  if (!state.userSize) { $('size').value = meta.projR; sliders(); }
  state.simFrames = 0;
  state.events = params.has('noev') ? [] : (meta.events ?? []).map((e) => ({ ...e, done: false }));
  const cam = params.get('cam')?.split(',').map(Number);
  camera.position.fromArray(cam ? cam.slice(0, 3) : meta.camera.pos);
  controls.target.fromArray(cam ? cam.slice(3, 6) : meta.camera.target);
  controls.update();
  const action = $('action');
  action.hidden = !meta.action;
  if (meta.action) action.textContent = meta.action.label;
  for (const b of document.querySelectorAll('#scenes button')) b.classList.toggle('on', b.dataset.scene === name);
  $('loading').hidden = true;
  console.log(`scene ${name}: ${data.nBodies} bodies, ${data.nParticles} particles, ${data.nJoints} joints, ${data.colorRanges.length} colours, build ${tBuild.toFixed(0)} ms. ${meta.info ?? ''}`);
  const url = new URL(location.href);
  url.searchParams.set('scene', name);
  if (!OFFSCREEN) history.replaceState(null, '', url);
}

/**
 * Launch a projectile from `origin` along `dir`. If it is going to hit an intact
 * glass pane, the pane is first re-cut around the predicted impact point.
 */
async function fire(origin, dir, speed = +$('speed').value) {
  const { sim, meta, data } = state;
  if (!sim) return;
  const id = meta.projectiles[state.nextProj++ % meta.projectiles.length];
  const b = data.bodyList[id];
  const d = new THREE.Vector3(...dir).normalize();
  const o = new THREE.Vector3(...origin).addScaledVector(d, 0.8);
  const vel = d.multiplyScalar(speed).toArray();
  let best = null;
  for (const pane of meta.panes ?? []) {
    if (!pane.intact) continue;
    const h = pane.hitPoint(o.toArray(), vel);
    if (h && (!best || h.t < best.h.t)) best = { pane, h };
  }
  if (best) {
    const { pane, h } = best;
    pane.intact = false; // one re-cut per pane
    if (await paneUntouched(pane) && state.sim === sim) {
      pane.refracture([h.x, h.y], sim, data, state.group.userData.meshes);
      console.log(`glass re-cut around (${h.x.toFixed(2)}, ${h.y.toFixed(2)})`);
    }
  }
  if (state.sim !== sim) return;
  sim.setBody(id, { pos: o.toArray(), vel, ang: [0, 0, 0], invMass: b.realInvMass });
}

/** Reads the pane's bodies back: true if no shard cracked or moved since build. */
async function paneUntouched(pane) {
  const sim = state.sim;
  const n = pane.ids.length;
  const buf = await sim.read(sim.bodyBuf, (pane.ids[0] + n) * BODY_FLOATS * 4);
  const f = new Float32Array(buf), u = new Uint32Array(buf);
  for (const id of pane.ids) {
    const o = id * BODY_FLOATS, rest = state.data.bodyList[id].pos;
    if (u[o + 31] & 1) return false;
    if (Math.hypot(f[o] - rest[0], f[o + 1] - rest[1], f[o + 2] - rest[2]) > 0.03) return false;
  }
  return true;
}

/** Encodes and submits one displayed frame of simulation. */
function simulateFrame() {
  const { sim } = state;
  const enc = device.createCommandEncoder();
  sim.encodeFrame(enc, state.slow ? 0.25 : 1);
  const req = sim.requestStats(enc);
  device.queue.submit([enc.finish()]);
  sim.afterSubmit(req);
  state.simFrames++;
  const t = sim.simTime;
  for (const e of state.events) if (!e.done && t >= e.t) { e.done = true; e.run(sim, state); }
  state.inflight++;
  device.queue.onSubmittedWorkDone().then(() => state.inflight--);
}

function render() {
  if (rt) renderer.setRenderTarget(rt);
  renderer.render(scene, camera);
  if (rt) renderer.setRenderTarget(null);
}

// ------------------------------------------------------------------ UI
for (const [k, s] of Object.entries(SCENES)) {
  if (s.hidden) continue;
  const b = document.createElement('button');
  b.textContent = s.title; b.dataset.scene = k;
  b.onclick = () => loadScene(k);
  $('scenes').append(b);
}
const sliders = () => {
  $('speedv').textContent = $('speed').value + ' m/s';
  $('sizev').textContent = Math.round($('size').value * 200) + ' cm';
  const t = Math.pow(2, +$('tough').value);
  $('toughv').textContent = t.toFixed(2) + '×';
  if (state.sim) { state.sim.settings.breakScale = 1 / t; state.sim.writeParams(); }
};
for (const id of ['speed', 'size', 'tough']) $(id).oninput = sliders;
$('size').onchange = () => { state.userSize = true; loadScene(state.name); };
sliders();
$('reset').onclick = () => loadScene(state.name);
const toggle = (key, el) => { state[key] = !state[key]; el.classList.toggle('on', state[key]); };
$('slow').onclick = () => toggle('slow', $('slow'));
$('pause').onclick = () => toggle('paused', $('pause'));
$('action').onclick = () => state.meta?.action?.run(state.sim, state);
$('quality').textContent = LITE ? 'Quality: Lite' : 'Quality: HD';
$('quality').onclick = () => { const u = new URL(location.href); u.searchParams.set('q', LITE ? 'hd' : 'lite'); location.href = u; };
$('collapse').onclick = () => document.body.classList.toggle('collapsed');
if (innerWidth < 600) document.body.classList.add('collapsed'); // phones: start with the panel folded
addEventListener('keydown', (e) => {
  if (e.key === 'r') loadScene(state.name);
  if (e.key === 's') toggle('slow', $('slow'));
  if (e.key === 'p') toggle('paused', $('pause'));
});

// tap / click (not drag) fires from the camera through the pointer
{
  let down = null;
  const el = renderer.domElement;
  el.addEventListener('pointerdown', (e) => { if (e.isPrimary) down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
  el.addEventListener('pointerup', (e) => {
    if (!down || !e.isPrimary) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    const dt = performance.now() - down.t;
    down = null;
    if (moved > 8 || dt > 450 || e.button !== 0) return;
    const ndc = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    fire(camera.position.toArray(), ray.ray.direction.toArray());
  });
}

addEventListener('resize', () => {
  if (OFFSCREEN) return;
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ------------------------------------------------------------------ loop
let last = performance.now(), acc = 0, fps = 60, statT = 0;
function updateStats(now) {
  if (now - statT < 300) return;
  statT = now;
  const s = state.sim, d = state.data;
  if (!s) return;
  const cells = [
    ['fps', fps.toFixed(0)], ['bodies', d.nBodies], ['particles', d.nParticles], ['joints', d.nJoints],
    ['broken', s.stats.broken], ['substeps', Math.max(1, Math.round(s.settings.substeps * (state.slow ? 0.25 : 1)))], ['quality', LITE ? 'lite' : 'HD'], ['sim t', (s.simTime ?? 0).toFixed(1) + 's'],
  ];
  $('stats').innerHTML = cells.map(([k, v]) => `<div><b>${v}</b><small>${k}</small></div>`).join('');
}

/**
 * If the GPU cannot sustain ~real-time physics (measured over 2 s windows after a warm-up),
 * trade a little stiffness for speed by lowering the substep count, down to a floor.
 */
function adaptSubsteps(now) {
  const a = state.adapt ??= { t0: now, f0: state.simFrames, slow: 0 };
  if (FIXED || state.paused || state.slow || now - a.t0 < 2000) return;
  const rate = (state.simFrames - a.f0) / ((now - a.t0) / 1000);
  a.t0 = now; a.f0 = state.simFrames;
  if ((state.sim.simTime ?? 0) < 2) return; // warm-up (pipeline compilation)
  a.slow = rate < 42 ? a.slow + 1 : 0;
  if (a.slow >= 2 && state.substeps > PRESET.minSubsteps) {
    state.substeps -= 2;
    state.sim.settings.substeps = state.substeps;
    state.sim.writeParams();
    a.slow = 0;
    console.log(`physics at ${rate.toFixed(0)} fps: substeps -> ${state.substeps}`);
  }
}

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  fps = fps * 0.95 + (1 / Math.max(dt, 1e-3)) * 0.05;
  controls.update();
  if (state.sim && !state.paused && !MANUAL) {
    acc = Math.min(acc + dt, 2 / 60);
    // backpressure: never queue more than 2 frames of physics on the GPU
    while (acc >= 1 / 60 - 1e-4 && state.inflight < 2) { simulateFrame(); acc -= 1 / 60; }
    if (state.inflight >= 2) acc = 0;
    adaptSubsteps(now);
  }
  render();
  updateStats(now);
  state.frames++;
}

await loadScene(state.name);
if (!MANUAL) renderer.setAnimationLoop(frame);

// ------------------------------------------------------------------ test hooks
window.__v2 = {
  state, renderer, camera, controls, fire, loadScene,
  async step(n = 1) {
    for (let i = 0; i < n; i++) { simulateFrame(); if (i % 4 === 3) await device.queue.onSubmittedWorkDone(); }
    await device.queue.onSubmittedWorkDone();
  },
  async bodies() {
    const a = await state.sim.readBodies();
    const out = [];
    for (let i = 0; i < state.sim.nBodies; i++) out.push(Array.from(a.subarray(i * BODY_FLOATS, i * BODY_FLOATS + 8)));
    return out;
  },
  async summary() {
    const a = await state.sim.readBodies();
    let nan = 0, minY = 1e9, maxY = -1e9, maxV = 0;
    const n = state.sim.nBodies;
    for (let i = 0; i < n; i++) {
      const o = i * BODY_FLOATS;
      if (a[o + 3] === 0) continue;
      for (let k = 0; k < 3; k++) if (!isFinite(a[o + k])) nan++;
      minY = Math.min(minY, a[o + 1]); maxY = Math.max(maxY, a[o + 1]);
      maxV = Math.max(maxV, Math.hypot(a[o + 16], a[o + 17], a[o + 18]));
    }
    return { t: +(state.sim.simTime ?? 0).toFixed(3), nan, minY: +minY.toFixed(3), maxY: +maxY.toFixed(3), maxV: +maxV.toFixed(3), broken: state.sim.stats.broken };
  },
  /** Max / mean displacement of dynamic bodies from their as-built position (rigidity test). */
  async drift() {
    const a = await state.sim.readBodies();
    let max = 0, sum = 0, n = 0;
    for (const b of state.data.bodyList) {
      if (b.invMass === 0) continue;
      const o = b.id * BODY_FLOATS;
      const d = Math.hypot(a[o] - b.pos[0], a[o + 1] - b.pos[1], a[o + 2] - b.pos[2]);
      max = Math.max(max, d); sum += d; n++;
    }
    return { max: +max.toFixed(4), mean: +(sum / n).toFixed(5) };
  },
  async capture() {
    render();
    const w = rt.width, h = rt.height;
    const px = await renderer.readRenderTargetPixelsAsync(rt, 0, 0, w, h);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(w, h);
    img.data.set(px.subarray(0, w * h * 4));
    ctx.putImageData(img, 0, 0);
    return c.toDataURL('image/png');
  },
};
window.__v2.ready = true;
