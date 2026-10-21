// Voxel -> tetrahedral mesh builder.
//
// Every object is a tetrahedral mesh. Cracks only ever run along
// tet faces: the GPU flags an overstressed shared face and then splits the
// mesh there by duplicating the vertices on it. This module builds the meshes
// and the static adjacency tables those kernels walk.

export const MAT = { WOOD: 0, GLASS: 1, STEEL: 2, STONE: 3, JELLY: 4, ROCK: 5 };

// stiffness: 0..1 blend toward the rigid tet goal (1 = stiff)
// toughness: face strain needed to break (scaled by a UI multiplier)
// spread: random variation of toughness per face (irregular cracks)
// yield/plastic: permanent bending once the tet strain passes `yield`
// cluster: pull toward the rigid fit of each surrounding block of cells (bending stiffness)
export const MATERIALS = [
  { name: 'Wood',  density: 600,  stiffness: 1.0,  toughness: 0.09, spread: 0.5, yield: 0.06, plastic: 0.05, friction: 0.6, grain: true , cluster: 1.0 },
  { name: 'Glass', density: 2500, stiffness: 1.0,  toughness: 0.025, spread: 0.7, yield: 9,   plastic: 0,    friction: 0.3 , cluster: 1.0 },
  { name: 'Steel', density: 7800, stiffness: 1.0,  toughness: 0.9,  spread: 0.2, yield: 0.012, plastic: 0.5, friction: 0.4 , cluster: 0.5 },
  { name: 'Stone', density: 2300, stiffness: 1.0,  toughness: 0.05, spread: 0.6, yield: 9,   plastic: 0,    friction: 0.8 , cluster: 1.0 },
  { name: 'Jelly', density: 1000, stiffness: 0.06, toughness: 99,   spread: 0,   yield: 9,   plastic: 0,    friction: 0.5 , cluster: 0.0 },
  { name: 'Rock',  density: 2600, stiffness: 1.0,  toughness: 1.5,  spread: 0.3, yield: 9,   plastic: 0,    friction: 0.9 , cluster: 1.0 },
];

// 5-tet split of a cube; the mirrored variant on odd cells keeps faces conforming.
// Corners: 0=(000) 1=(100) 2=(110) 3=(010) 4=(001) 5=(101) 6=(111) 7=(011)
const CUBE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const TETS_EVEN = [[0, 1, 3, 4], [2, 3, 1, 6], [5, 4, 6, 1], [7, 6, 4, 3], [1, 3, 4, 6]];
const TETS_ODD = [[1, 0, 2, 5], [3, 2, 0, 7], [4, 5, 7, 0], [6, 7, 5, 2], [0, 2, 5, 7]];
// Face opposite corner f, wound so the normal points out of a positively oriented tet.
export const TET_FACES = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class WorldBuilder {
  constructor(seed = 1) {
    this.rand = rng(seed);
    this.nodes = [];      // x,y,z
    this.gridNodes = [];  // x,y,z before warp (orientation reference)
    this.nodeVel = [];    // initial velocity
    this.nodePinned = [];
    this.tets = [];       // 4 node ids
    this.tetMat = [];
    this.tetObj = [];
    this.tetGrain = [];   // 0,1,2 axis of the wood grain
    this.tetPart = [];    // parts within one object are glued with weaker joints
    this.tetCell = [];    // voxel cell (i, j, k) the tet came from, for clustering
    this.tetExcl = [];    // same-object collision exclusion distance
    this.objects = [];
  }

  /**
   * Adds a voxelised object.
   *  min: world position of the voxel grid corner
   *  dims: [nx, ny, nz] cells, cell: [hx, hy, hz]
   *  cellFn(i,j,k) -> null | { grain, part }  (default: solid box)
   *  pin(x,y,z) -> bool  nodes held fixed in place (walls, ground anchors)
   *  velocity: [vx,vy,vz] or (x,y,z) -> [vx,vy,vz]
   *  warp([x,y,z], onSurface) -> [x,y,z]  reshape nodes, e.g. snap a voxel sphere's staircase
   *       surface onto the real curved surface
   */
  addVoxels({ material, min, dims, cell, cellFn, pin, velocity = [0, 0, 0], grain = 0, jitter = 0.22, jointStrength = 0.35, warp }) {
    const [nx, ny, nz] = dims;
    const h = typeof cell === 'number' ? [cell, cell, cell] : cell;
    const objectId = this.objects.length;
    const filled = new Array(nx * ny * nz).fill(null);
    const cidx = (i, j, k) => (k * ny + j) * nx + i;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = cellFn ? cellFn(i, j, k) : { grain, part: 0 };
      if (c) filled[cidx(i, j, k)] = { grain: c.grain ?? grain, part: c.part ?? 0 };
    }
    const isFilled = (i, j, k) => i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz && filled[cidx(i, j, k)] !== null;

    const nodeId = new Map();
    const getNode = (i, j, k) => {
      const key = (k * (ny + 1) + j) * (nx + 1) + i;
      let id = nodeId.get(key);
      if (id !== undefined) return id;
      // Jitter a node along an axis only when its neighbourhood is mirror
      // symmetric across that axis: outer surfaces stay flat, insides go irregular.
      const off = [0, 0, 0];
      for (let a = 0; a < 3; a++) {
        let ok = true, any = false;
        for (let d = 0; d < 8 && ok; d++) {
          const ci = [i - 1 + (d & 1), j - 1 + ((d >> 1) & 1), k - 1 + ((d >> 2) & 1)];
          const m = ci.slice(); m[a] = (a === 0 ? i : a === 1 ? j : k) * 2 - 1 - ci[a];
          const f = isFilled(ci[0], ci[1], ci[2]);
          any ||= f;
          if (f !== isFilled(m[0], m[1], m[2])) ok = false;
        }
        if (ok && any) off[a] = (this.rand() * 2 - 1) * jitter * h[a];
      }
      let x = min[0] + i * h[0] + off[0], y = min[1] + j * h[1] + off[1], z = min[2] + k * h[2] + off[2];
      this.gridNodes.push(x, y, z);
      if (warp) {
        let surface = false;
        for (let d = 0; d < 8; d++) if (!isFilled(i - 1 + (d & 1), j - 1 + ((d >> 1) & 1), k - 1 + ((d >> 2) & 1))) surface = true;
        [x, y, z] = warp([x, y, z], surface);
      }
      id = this.nodes.length / 3;
      this.nodes.push(x, y, z);
      this.nodeVel.push(...(typeof velocity === 'function' ? velocity(x, y, z) : velocity));
      this.nodePinned.push(pin && pin(x, y, z) ? 1 : 0);
      nodeId.set(key, id);
      return id;
    };

    const tetStart = this.tets.length / 4;
    const excl = 1.6 * Math.max(...h);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = filled[cidx(i, j, k)];
      if (!c) continue;
      const corner = CUBE.map(([a, b, d]) => getNode(i + a, j + b, k + d));
      for (const t of ((i + j + k) & 1) ? TETS_ODD : TETS_EVEN) {
        let ids = t.map((q) => corner[q]);
        // orient on the unwarped grid: a warp can fold a tet, and flipping it to match
        // would turn its surface faces inside out
        if (this._signedVolume(ids, this.gridNodes) < 0) ids = [ids[0], ids[2], ids[1], ids[3]];
        this.tets.push(...ids);
        this.tetMat.push(material);
        this.tetObj.push(objectId);
        this.tetGrain.push(c.grain);
        this.tetPart.push(c.part);
        this.tetCell.push(i, j, k);
        this.tetExcl.push(excl);
      }
    }
    this.objects.push({ material, tetStart, tetCount: this.tets.length / 4 - tetStart, jointStrength });
    return objectId;
  }

  _signedVolume(ids, n = this.nodes) {
    const p = ids.map((i) => [n[i * 3], n[i * 3 + 1], n[i * 3 + 2]]);
    const a = sub(p[1], p[0]), b = sub(p[2], p[0]), c = sub(p[3], p[0]);
    return dot(cross(a, b), c) / 6;
  }

  build() {
    const N = this.nodes.length / 3, T = this.tets.length / 4;
    const nodePos = new Float32Array(this.nodes);
    const tetNodes = new Int32Array(this.tets);
    const tetMass = new Float32Array(T), tetRadius = new Float32Array(T), tetRestC = new Float32Array(T * 3);
    for (let t = 0; t < T; t++) {
      const ids = [tetNodes[t * 4], tetNodes[t * 4 + 1], tetNodes[t * 4 + 2], tetNodes[t * 4 + 3]];
      const V = Math.abs(this._signedVolume(ids));
      tetMass[t] = V * MATERIALS[this.tetMat[t]].density;
      tetRadius[t] = Math.cbrt((3 * V) / (4 * Math.PI));
      for (const id of ids) for (let a = 0; a < 3; a++) tetRestC[t * 3 + a] += nodePos[id * 3 + a] * 0.25;
    }

    // Interior faces: two tets sharing the same three nodes.
    const faceMap = new Map();
    const slotFace = new Int32Array(T * 4).fill(-1);
    const faceTets = [], faceCorners = [], faceNodes = [], faceThr = [];
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < 4; f++) {
        const tri = TET_FACES[f].map((c) => tetNodes[t * 4 + c]);
        const key = tri.slice().sort((a, b) => a - b).join(',');
        const other = faceMap.get(key);
        if (other === undefined) { faceMap.set(key, t * 4 + f); continue; }
        const ta = other >> 2, tb = t;
        const fid = faceTets.length / 2;
        const sorted = key.split(',').map(Number);
        let pa = 0, pb = 0;
        sorted.forEach((n, i) => {
          for (let c = 0; c < 4; c++) {
            if (tetNodes[ta * 4 + c] === n) pa |= c << (2 * i);
            if (tetNodes[tb * 4 + c] === n) pb |= c << (2 * i);
          }
        });
        faceTets.push(ta, tb);
        faceCorners.push(pa, pb);
        faceNodes.push(...sorted);
        slotFace[other] = fid;
        slotFace[t * 4 + f] = fid;
        faceThr.push(this._faceThreshold(ta, tb, sorted, nodePos));
      }
    }

    // Per-node adjacency for the GPU split kernel. nodeData holds, per node, its
    // corners (t*4+k) followed by (cornerA, cornerB, face) triples: one for each
    // interior face touching the node, naming the two corners glued across it.
    const F = faceTets.length / 2;
    const corners = Array.from({ length: N }, () => []);
    const pairs = Array.from({ length: N }, () => []);
    for (let i = 0; i < T * 4; i++) corners[tetNodes[i]].push(i);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < 3; i++) {
        const n = faceNodes[f * 3 + i];
        const ka = (faceCorners[f * 2] >> (2 * i)) & 3, kb = (faceCorners[f * 2 + 1] >> (2 * i)) & 3;
        pairs[n].push(faceTets[f * 2] * 4 + ka, faceTets[f * 2 + 1] * 4 + kb, f);
      }
    }
    const nodeRange = new Int32Array(N * 4);
    const data = [];
    let maxCorners = 0;
    for (let n = 0; n < N; n++) {
      nodeRange.set([data.length, corners[n].length, data.length + corners[n].length, pairs[n].length / 3], n * 4);
      data.push(...corners[n], ...pairs[n]);
      maxCorners = Math.max(maxCorners, corners[n].length);
    }

    // Two ivec4 per face: (tetA, tetB, cornersA, cornersB), (node0, node1, node2, threshold bits)
    const faceData = new Int32Array(Math.max(F, 1) * 8);
    const faceDataF = new Float32Array(faceData.buffer);
    for (let f = 0; f < F; f++) {
      faceData.set([faceTets[f * 2], faceTets[f * 2 + 1], faceCorners[f * 2], faceCorners[f * 2 + 1],
        faceNodes[f * 3], faceNodes[f * 3 + 1], faceNodes[f * 3 + 2]], f * 8);
      faceDataF[f * 8 + 7] = faceThr[f];
    }

    return {
      N, T, F, maxCorners,
      nodePos, nodeVel: new Float32Array(this.nodeVel), nodePinned: new Uint8Array(this.nodePinned),
      tetNodes, tetMass, tetRadius, tetRestC,
      nodeObj: (() => { const o = new Int32Array(N); for (let i = 0; i < T * 4; i++) o[tetNodes[i]] = this.tetObj[i >> 2]; return o; })(),
      tetMat: new Uint8Array(this.tetMat), tetObj: new Int32Array(this.tetObj), tetGrain: new Uint8Array(this.tetGrain),
      tetExcl: new Float32Array(this.tetExcl),
      faceData, slotFace, nodeRange, nodeData: new Int32Array(data),
      ...this._clusters(T, tetNodes, tetMass, nodePos, faceTets, F),
      objects: this.objects,
    };
  }

  /**
   * Blocks of CLUSTER^3 voxel cells per object, in two layers offset by half a
   * block, for lattice shape matching (Rivers & James 2007). Each cluster lists
   * its tet corners and the interior faces that would invalidate it if broken.
   * clusterBuf: 4 vec4 per cluster: (rest centroid, weight) (quat) (centroid, layer)
   * (cornerStart, cornerCount, faceStart, faceCount) as floats.
   */
  _clusters(T, tetNodes, tetMass, nodePos, faceTets, F) {
    const S = 4;
    const layers = [new Map(), new Map()];
    const tetCluster = new Int32Array(T * 2).fill(-1);
    for (let t = 0; t < T; t++) {
      const w = MATERIALS[this.tetMat[t]].cluster;
      if (!w) continue;
      for (let l = 0; l < 2; l++) {
        const o = l * (S / 2);
        const key = [this.tetObj[t], ...[0, 1, 2].map((a) => Math.floor((this.tetCell[t * 3 + a] + o) / S))].join(',');
        let list = layers[l].get(key);
        if (!list) layers[l].set(key, (list = []));
        list.push(t);
      }
    }
    const clusters = [];
    for (let l = 0; l < 2; l++) for (const tets of layers[l].values()) if (tets.length >= 10) clusters.push({ l, tets });
    const C = clusters.length;
    const buf = new Float32Array(Math.max(C, 1) * 16);
    const data = [];
    clusters.forEach((cl, ci) => {
      for (const t of cl.tets) tetCluster[t * 2 + cl.l] = ci;
      let m = 0;
      const rc = [0, 0, 0];
      const cornerStart = data.length;
      for (const t of cl.tets) {
        for (let k = 0; k < 4; k++) {
          data.push(t * 4 + k);
          const n = tetNodes[t * 4 + k];
          for (let a = 0; a < 3; a++) rc[a] += nodePos[n * 3 + a] * tetMass[t] * 0.25;
          m += tetMass[t] * 0.25;
        }
      }
      const faceStart = data.length;
      clusters[ci].faces = 0;
      buf.set([rc[0] / m, rc[1] / m, rc[2] / m, MATERIALS[this.tetMat[cl.tets[0]]].cluster, 0, 0, 0, 1, rc[0] / m, rc[1] / m, rc[2] / m, cl.l,
        cornerStart, cl.tets.length * 4, faceStart, 0], ci * 16);
    });
    // interior faces per cluster, appended after the corner lists
    const faceLists = clusters.map(() => []);
    for (let f = 0; f < F; f++) {
      const a = faceTets[f * 2], b = faceTets[f * 2 + 1];
      for (let l = 0; l < 2; l++) {
        const ca = tetCluster[a * 2 + l];
        if (ca >= 0 && ca === tetCluster[b * 2 + l]) faceLists[ca].push(f);
      }
    }
    faceLists.forEach((list, ci) => {
      buf[ci * 16 + 14] = data.length;
      buf[ci * 16 + 15] = list.length;
      data.push(...list);
    });
    // each cluster keeps its own rest offsets (corner rest position - rest centroid), so
    // plastic flow can reshape it without touching other clusters
    const rest = new Float32Array(Math.max(data.length, 1) * 4);
    for (let ci = 0; ci < C; ci++) {
      const o = ci * 16, start = buf[o + 12], count = buf[o + 13];
      for (let i = start; i < start + count; i++) {
        const n = tetNodes[data[i]];
        for (let a = 0; a < 3; a++) rest[i * 4 + a] = nodePos[n * 3 + a] - buf[o + a];
      }
    }
    return { C, clusterBuf: buf, clusterData: new Int32Array(data.length ? data : [0]), clusterRest: rest };
  }

  _faceThreshold(ta, tb, tri, pos) {
    const m = MATERIALS[this.tetMat[ta]];
    let thr = m.toughness * (1 + (this.rand() * 2 - 1) * m.spread);
    if (this.tetMat[ta] !== this.tetMat[tb] || this.tetObj[ta] !== this.tetObj[tb]) return 1e9;
    if (this.tetPart[ta] !== this.tetPart[tb]) thr *= this.objects[this.tetObj[ta]].jointStrength;
    if (m.grain) {
      // Fibres resist cracks that cut across them; splitting along them is easy.
      const p = tri.map((i) => [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]]);
      const n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
      const along = Math.abs(n[this.tetGrain[ta]]) / Math.hypot(...n);
      thr *= 0.35 + 3.4 * along * along;
    }
    return thr;
  }
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
