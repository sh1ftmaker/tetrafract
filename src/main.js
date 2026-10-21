import * as THREE from 'three/webgpu';
import { Fn, vec3, vec4, float, positionWorld, positionLocal, instanceIndex, fract, abs, min, smoothstep, mix, length } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Simulation, MAX_BALLS } from './sim.js';
import { createMeshes, showTets } from './render.js';
import { SCENES, TEST_SCENES, buildScene, makeWreckingBall } from './scenes.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

if (!navigator.gpu) {
  $('error').textContent = 'This demo needs WebGPU (Chrome / Edge 113+, Safari 18+). The physics runs in WebGPU compute shaders.';
  $('error').hidden = false;
  throw new Error('WebGPU unavailable');
}

const adapter = await navigator.gpu.requestAdapter();
const renderer = new THREE.WebGPURenderer({
  antialias: !params.has('noaa'),
  requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(adapter?.limits.maxStorageBuffersPerShaderStage ?? 8, 10) },
});
await renderer.init();
renderer.backend.device?.lost.then((info) => console.error('WebGPU device lost at', performance.now().toFixed(0), info.reason, info.message));
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.25));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xa9b8c6);
scene.fog = new THREE.Fog(0xa9b8c6, 18, 45);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;

scene.add(new THREE.HemisphereLight(0xdde8ff, 0x6b5a48, 1.1));
const sun = new THREE.DirectionalLight(0xfff1dd, 2.6);
sun.position.set(6, 12, 5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -10, right: 10, top: 10, bottom: -10, near: 1, far: 40 });
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.02;
scene.add(sun, sun.target);

// Ground: a grey grid floor like a 3D authoring viewport
{
  const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.95 });
  mat.colorNode = Fn(() => {
    const p = positionWorld.xz;
    const line = (s, w) => {
      const g = abs(fract(p.div(s).add(0.5)).sub(0.5)).mul(s);
      return float(1).sub(smoothstep(0, w, min(g.x, g.y)));
    };
    const base = vec3(0.47, 0.48, 0.5);
    const c = mix(base, vec3(0.36, 0.37, 0.39), line(1, 0.02)).sub(line(0.25, 0.006).mul(0.03));
    const fade = smoothstep(8, 30, length(p));
    return vec4(mix(c, vec3(0.6, 0.66, 0.72), fade), 1);
  })();
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), mat);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);
}

const ballGeo = new THREE.SphereGeometry(1, 32, 16);
let ballMesh = null;
/** Projectiles are drawn straight from the GPU ball buffer. */
function makeBallMesh(sim) {
  const mat = new THREE.MeshStandardNodeMaterial({ color: 0x2a2c30, metalness: 0.8, roughness: 0.35 });
  mat.positionNode = Fn(() => {
    const rec = sim.balls.element(instanceIndex.mul(3));
    return positionLocal.mul(rec.w).add(rec.xyz);
  })();
  const mesh = new THREE.InstancedMesh(ballGeo, mat, MAX_BALLS);
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  return mesh;
}
const chain = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicNodeMaterial({ color: 0x222222 }));
chain.frustumCulled = false;
scene.add(chain);

const state = {
  sceneKey: SCENES[params.get('scene')] || TEST_SCENES[params.get('scene')] ? params.get('scene') : 'trestle',
  tool: 'cannon',
  paused: false, slowmo: false,
  speed: 20, size: 0.2, mass: 25,
};
const SUBSTEPS = +(params.get('substeps') ?? 6);
// Solver iterations per substep for each quality level; the timestep never changes, so
// the level can move live without disturbing velocities.
const QUALITY = [
  { name: 'Low', iters: 4, dpr: 1 },
  { name: 'Medium', iters: 8, dpr: 1.25 },
  { name: 'High', iters: 14, dpr: 1.5 },
  { name: 'Ultra', iters: 24, dpr: 2 },
];
const quality = { auto: !params.has('iters'), level: 1, goodSince: 0, lastChange: 0, frames: 0, skipped: 0, windowStart: 0 };
const fixedSteps = +(params.get('fixed') ?? 0); // test hook: constant steps per frame
let sim = null, meshes = [], wrecker = null, acc = 0;

function loadScene(key) {
  state.sceneKey = key;
  if (sim) sim.dispose();
  for (const m of meshes) { scene.remove(m); if (!m.isSprite) m.geometry.dispose(); m.material.dispose(); } // sprites share one geometry
  if (ballMesh) { scene.remove(ballMesh); ballMesh.material.dispose(); }

  const def = SCENES[key] ?? TEST_SCENES[key];
  const mesh = buildScene(key);
  sim = new Simulation(renderer, mesh, {
    substeps: SUBSTEPS,
    iterations: params.has('iters') ? +params.get('iters') : QUALITY[quality.level].iters,
    rho: params.has('rho') ? +params.get('rho') : undefined,
    clusterWeight: params.has('cw') ? +params.get('cw') : undefined,
  });
  sim.u.toughness.value = toughness();
  meshes = createMeshes(sim);
  scene.add(...meshes);
  ballMesh = makeBallMesh(sim);
  scene.add(ballMesh);
  wrecker = def.wreckingBall ? sim.addBall(makeWreckingBall(def.wreckingBall)) : null;
  chain.visible = !!wrecker;
  camera.position.set(...def.camera);
  controls.target.set(...def.target);
  controls.update();
  acc = 0;
  document.querySelectorAll('[data-scene]').forEach((b) => b.classList.toggle('on', b.dataset.scene === key));
  $('tets').textContent = mesh.T.toLocaleString();
}

function toughness() { return Math.pow(2, parseFloat($('tough').value)); }

// ---- input: tap/click shoots or force-pushes, drag orbits ----
const ray = new THREE.Raycaster();
let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  const quick = performance.now() - down.t < 350;
  down = null;
  if (moved > 8 || !quick) return;
  const ndc = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const dir = ray.ray.direction.clone();
  if (state.tool === 'cannon') {
    sim.addBall({
      pos: camera.position.clone().addScaledVector(dir, 0.8), vel: dir.multiplyScalar(state.speed),
      radius: state.size, mass: state.mass, life: 12000,
    });
  } else if (state.tool === 'boulder') {
    // drop from 5 m above wherever the ray meets the ground
    const t = -camera.position.y / Math.min(dir.y, -0.05);
    const hit = camera.position.clone().addScaledVector(dir, t);
    sim.addBall({
      pos: new THREE.Vector3(hit.x, 5 + state.size, hit.z), vel: new THREE.Vector3(),
      radius: state.size, mass: state.mass, life: 20000,
    });
  } else {
    sim.push(camera.position, dir, 14);
  }
});

// ---- UI ----
const sceneBar = $('scenes');
for (const [key, def] of Object.entries(SCENES)) {
  const b = document.createElement('button');
  b.textContent = def.label;
  b.dataset.scene = key;
  b.onclick = () => loadScene(key);
  sceneBar.append(b);
}
document.querySelectorAll('[data-tool]').forEach((b) => (b.onclick = () => {
  state.tool = b.dataset.tool;
  document.querySelectorAll('[data-tool]').forEach((x) => x.classList.toggle('on', x === b));
}));
$('reset').onclick = () => loadScene(state.sceneKey);

$('pause').onclick = () => { state.paused = !state.paused; $('pause').classList.toggle('on', state.paused); };
$('slowmo').onclick = () => { state.slowmo = !state.slowmo; $('slowmo').classList.toggle('on', state.slowmo); };
$('showtets').onclick = () => { showTets.value = showTets.value ? 0 : 1; $('showtets').classList.toggle('on', !!showTets.value); };
$('tough').oninput = () => { $('toughv').textContent = toughness().toFixed(2) + '×'; if (sim) sim.u.toughness.value = toughness(); };
$('speed').oninput = () => { state.speed = +$('speed').value; $('speedv').textContent = state.speed + ' m/s'; };
$('size').oninput = () => { state.size = +$('size').value; $('sizev').textContent = Math.round(state.size * 200) + ' cm'; };
$('mass').oninput = () => { state.mass = Math.round(Math.pow(10, +$('mass').value)); $('massv').textContent = state.mass + ' kg'; };
$('collapse').onclick = () => document.body.classList.toggle('collapsed');
addEventListener('keydown', (e) => {
  if (e.key === 'r') loadScene(state.sceneKey);
  if (e.key === ' ') $('pause').click();
  if (e.key === 't') $('showtets').click();
});
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

loadScene(state.sceneKey);

// Test hook: headless software WebGPU can't present to a canvas, so render
// into a target and read it back instead (?offscreen).
const offscreen = params.has('offscreen') ? new THREE.RenderTarget(innerWidth, innerHeight) : null;
async function capture() {
  controls.update();
  renderer.setRenderTarget(offscreen);
  renderer.render(scene, camera);
  const w = offscreen.width, h = offscreen.height;
  const px = await renderer.readRenderTargetPixelsAsync(offscreen, 0, 0, w, h);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const img = new ImageData(new Uint8ClampedArray(px.buffer, px.byteOffset, w * h * 4), w, h);
  c.getContext('2d').putImageData(img, 0, 0);
  return c.toDataURL('image/png');
}

let last = performance.now(), fpsT = 0, frames = 0;
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const elapsed = Math.min((now - last) / 1000, 0.05);
  last = now;

  if (fixedSteps) sim.step(fixedSteps);
  else if (!state.paused) {
    acc += elapsed * (state.slowmo ? 0.2 : 1);
    let steps = Math.floor(acc / sim.dt);
    steps = Math.min(steps, SUBSTEPS); // never try to catch up: that spirals on slow GPUs
    acc = Math.min(acc - steps * sim.dt, sim.dt * SUBSTEPS);
    if (steps > 0 && !sim.step(steps)) { acc = 0; quality.skipped++; }
  }

  if (wrecker) {
    const p = chain.geometry.attributes.position;
    p.setXYZ(0, wrecker.pivot.x, wrecker.pivot.y, wrecker.pivot.z);
    p.setXYZ(1, wrecker.pos.x, wrecker.pos.y, wrecker.pos.z);
    p.needsUpdate = true;
  }

  controls.update();
  if (offscreen) renderer.setRenderTarget(offscreen);
  renderer.render(scene, camera);

  adaptQuality(now);
  frames++;
  if (now - fpsT > 500) {
    $('fps').textContent = Math.round((frames * 1000) / (now - fpsT));
    $('verts').textContent = sim.vertexCount.toLocaleString();
    $('cracks').textContent = sim.brokenCount.toLocaleString();
    fpsT = now; frames = 0;
  }
});

function setQuality(level) {
  quality.level = level;
  const q = QUALITY[level];
  if (sim && !params.has('iters')) sim.iterations = q.iters;
  renderer.setPixelRatio(Math.min(devicePixelRatio, q.dpr));
  renderer.setSize(innerWidth, innerHeight);
  $('quality').value = quality.auto ? 'auto' : String(level);
  $('qualityv').textContent = quality.auto ? `Auto · ${q.name}` : q.name;
}

// Every second: drop a level if we miss ~50 fps or the GPU made us skip steps; climb
// back up only after several smooth seconds.
function adaptQuality(now) {
  if (!quality.auto || fixedSteps) return;
  if (!quality.windowStart) quality.windowStart = now;
  quality.frames++;
  const span = now - quality.windowStart;
  if (span < 1000) return;
  const fps = (quality.frames * 1000) / span;
  const struggling = fps < 50 || quality.skipped > 3;
  if (struggling) {
    quality.goodSince = now;
    if (quality.level > 0 && now - quality.lastChange > 1500) { setQuality(quality.level - 1); quality.lastChange = now; }
  } else if (fps > 58 && now - quality.goodSince > 5000 && quality.level < QUALITY.length - 1 && now - quality.lastChange > 5000) {
    setQuality(quality.level + 1);
    quality.lastChange = now;
    quality.goodSince = now;
  }
  quality.frames = 0;
  quality.skipped = 0;
  quality.windowStart = now;
}

$('quality').onchange = () => {
  const v = $('quality').value;
  quality.auto = v === 'auto';
  quality.goodSince = quality.lastChange = performance.now();
  setQuality(quality.auto ? 1 : +v);
};
setQuality(quality.level);

window.__tetrafract = { get sim() { return sim; }, loadScene, camera, controls, renderer, capture, state };
