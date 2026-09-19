// FusionLab 3D vessel view (Three.js, vendored; no bundler). Used by the Replay tab through window.Vessel3D.
// Geometry is what FAIR-MAST level 2 carries, revolved about the machine axis: the limiter contour (37 points), the
// PF filament boxes grouped into coil packs, EFIT's last closed flux surface of the scrubbed slice, and the field lines
// traced by Warp. Axisymmetric: ports, the outer tank and any 3D structure are not in the data, so they are not drawn.
// The glow is a rendering choice (rim light on the real boundary, coloured by core Te). It is not a synthetic camera image.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const TAU = 2 * Math.PI, NPHI = 128;
const CAM_AZ = Math.PI / 4;                       // toroidal angle the camera starts at; the cutaway opens towards it
const LINE_COLORS = ['#9ec5f4', '#3987e5', '#2f6fd0', '#ffd166', '#ffd166'];   // psi_N 0.3, 0.6, 0.9, two SOL lines
const TE_COLD = new THREE.Color(0xff4fa3), TE_HOT = new THREE.Color(0xffb866), TE_RANGE = [0.2, 1.4];   // keV
const VIEWS = {
  // dist is only the fallback: fit() sets the distance that frames the whole machine for the panel's aspect ratio
  cutaway: { fov: 38, dist: 8.4, elev: 20 * Math.PI / 180, target: [0, 0, 0], sway: 0.5, rate: 0.22,
             minDist: 0.3, maxDist: 18, minPolar: 0.05, maxPolar: Math.PI - 0.05,
             room: 1, env: 1, glow: 1, tube: 0.006, lineOpacity: 0.8 },
  // inside the tank on the midplane side, R = 1.75 m, Z = 0.15 m, looking at the centre column
  port: { fov: 70, dist: Math.hypot(1.75, 0.15), elev: Math.atan2(0.15, 1.75), target: [0, 0, 0], sway: 0.18, rate: 0.3,
          minDist: 0.9, maxDist: 1.84, minPolar: 1.37, maxPolar: 1.77,
          // inside a closed vessel the plasma is the lamp: room lights down, thin lines (they pass 0.3 m from the lens)
          room: 0.12, env: 0.18, glow: 0.8, tube: 0.0022, lineOpacity: 0.6 },
};
const COS = new Float32Array(NPHI + 1), SIN = new Float32Array(NPHI + 1);

let S = null;          // { el, renderer, scene, camera, controls, groups..., materials... }
let shot = null, view = 'cutaway', auto = true, t0 = 0;
let extent = { r: 1.9, h: 1.85 };
let last = { i: null, lines: null, glow: 0 };      // what is on screen, so a view change can restyle it                 // vessel radius and half height [m], from the wall contour

// Surface of revolution of an (R, Z) polyline. Data is z-up; three is y-up: (x, y, z) -> (x, z, -y), a proper rotation.
// flat: every profile segment keeps its own normal (sharp machined corners). smooth: closed curve, averaged normals.
function revolve(R, Z, { phi0 = 0, dphi = TAU, smooth = false } = {}) {
  const n = R.length, seg = [];
  for (let k = 0; k < n - 1; k++) {
    const dR = R[k + 1] - R[k], dZ = Z[k + 1] - Z[k], L = Math.hypot(dR, dZ);
    seg.push(L > 1e-9 ? [dZ / L, -dR / L] : [0, 0]);
  }
  const rows = [], quads = [];
  if (smooth) {
    for (let k = 0; k < n; k++) {
      const a = seg[(k - 1 + n - 1) % (n - 1)], b = seg[k % (n - 1)], L = Math.hypot(a[0] + b[0], a[1] + b[1]) || 1;
      rows.push([R[k], Z[k], (a[0] + b[0]) / L, (a[1] + b[1]) / L]);
      if (k < n - 1) quads.push([k, k + 1]);
    }
  } else {
    for (let k = 0; k < n - 1; k++) {
      rows.push([R[k], Z[k], seg[k][0], seg[k][1]], [R[k + 1], Z[k + 1], seg[k][0], seg[k][1]]);
      quads.push([2 * k, 2 * k + 1]);
    }
  }
  for (let j = 0; j <= NPHI; j++) { const ph = phi0 + dphi * j / NPHI; COS[j] = Math.cos(ph); SIN[j] = Math.sin(ph); }
  const W = NPHI + 1, pos = new Float32Array(rows.length * W * 3), nor = new Float32Array(rows.length * W * 3);
  rows.forEach(([r, z, nr, nz], a) => {
    for (let j = 0; j < W; j++) {
      const o = (a * W + j) * 3;
      pos[o] = r * COS[j]; pos[o + 1] = z; pos[o + 2] = -r * SIN[j];
      nor[o] = nr * COS[j]; nor[o + 1] = nz; nor[o + 2] = -nr * SIN[j];
    }
  });
  const idx = new Uint32Array(quads.length * NPHI * 6);
  let q = 0;
  for (const [a, b] of quads) for (let j = 0; j < NPHI; j++) {
    const A = a * W + j, B = b * W + j, Cc = A + 1, D = B + 1;   // winding agrees with the normals above
    idx[q++] = A; idx[q++] = Cc; idx[q++] = B; idx[q++] = B; idx[q++] = Cc; idx[q++] = D;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

// Thin tube along a traced field line, so the line has a width on any display. Frames come from the vertical axis:
// a field line is never vertical for long in a tokamak.
function tube(l, radius = 0.006, sides = 5) {
  const raw = [];
  for (let k = 0; k < l.x.length; k++) {
    if (l.x[k] === null || l.y[k] === null || l.z[k] === null) continue;
    const p = new THREE.Vector3(l.x[k], l.z[k], -l.y[k]);
    if (!raw.length || raw[raw.length - 1].distanceToSquared(p) > 1e-8) raw.push(p);
  }
  if (raw.length < 4) return null;
  const pts = new THREE.CatmullRomCurve3(raw, false, 'centripetal').getPoints(raw.length * 2);
  const n = pts.length, pos = new Float32Array(n * sides * 3), idx = new Uint32Array((n - 1) * sides * 6);
  const t = new THREE.Vector3(), u = new THREE.Vector3(), v = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), ex = new THREE.Vector3(1, 0, 0);
  for (let k = 0; k < n; k++) {
    t.subVectors(pts[Math.min(k + 1, n - 1)], pts[Math.max(k - 1, 0)]).normalize();
    u.crossVectors(up, t);
    if (u.lengthSq() < 1e-6) u.crossVectors(ex, t);
    u.normalize(); v.crossVectors(t, u);
    for (let s = 0; s < sides; s++) {
      const a = TAU * s / sides, c = Math.cos(a) * radius, d = Math.sin(a) * radius, o = (k * sides + s) * 3;
      pos[o] = pts[k].x + c * u.x + d * v.x; pos[o + 1] = pts[k].y + c * u.y + d * v.y; pos[o + 2] = pts[k].z + c * u.z + d * v.z;
    }
  }
  let q = 0;
  for (let k = 0; k < n - 1; k++) for (let s = 0; s < sides; s++) {
    const A = k * sides + s, B = k * sides + (s + 1) % sides, Cc = A + sides, D = B + sides;
    idx[q++] = A; idx[q++] = B; idx[q++] = Cc; idx[q++] = B; idx[q++] = D; idx[q++] = Cc;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

// PF filaments -> coil packs: union-find on centre distance, then the pack's bounding box in (R, Z).
function coilPacks(c, link = 0.07) {
  const n = c.R.length, p = Array.from({ length: n }, (_, k) => k);
  const find = a => { while (p[a] !== a) { p[a] = p[p[a]]; a = p[a]; } return a; };
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++)
    if (Math.hypot(c.R[i] - c.R[j], c.Z[i] - c.Z[j]) < link) { const a = find(i), b = find(j); if (a !== b) p[a] = b; }
  const box = new Map();
  for (let k = 0; k < n; k++) {
    const r = find(k), w = (c.W?.[k] ?? 0.02) / 2, h = (c.H?.[k] ?? 0.03) / 2;
    const b = box.get(r) || [Infinity, -Infinity, Infinity, -Infinity];
    b[0] = Math.min(b[0], c.R[k] - w); b[1] = Math.max(b[1], c.R[k] + w);
    b[2] = Math.min(b[2], c.Z[k] - h); b[3] = Math.max(b[3], c.Z[k] + h);
    box.set(r, b);
  }
  return [...box.values()];
}

// Small procedural studio so the steel has something to reflect (no image assets, works offline).
function studio(renderer) {
  const s = new THREE.Scene(), geo = new THREE.SphereGeometry(30, 32, 16), p = geo.attributes.position, col = [];
  for (let k = 0; k < p.count; k++) { const h = p.getY(k) / 60 + 0.5; col.push(0.015 + 0.20 * h, 0.02 + 0.24 * h, 0.035 + 0.32 * h); }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  s.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
  const panel = (x, y, z, w, h, v) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(v, v, v * 1.05), side: THREE.DoubleSide }));
    m.position.set(x, y, z); m.lookAt(0, 0, 0); s.add(m);
  };
  panel(10, 12, 8, 12, 9, 7); panel(-14, 6, -4, 9, 12, 3.5); panel(2, -10, 12, 14, 6, 1.5); panel(-4, 3, 16, 5, 10, 2.5);
  const gen = new THREE.PMREMGenerator(renderer), tex = gen.fromScene(s, 0.03).texture;
  gen.dispose(); s.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
  return tex;
}

const RIM_VERT = `varying vec3 vN; varying vec3 vV;
  void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz);
                gl_Position = projectionMatrix * mv; }`;
const RIM_FRAG = `uniform vec3 uColor; uniform float uPower; uniform float uGain; uniform float uBase; varying vec3 vN; varying vec3 vV;
  void main() { float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), uPower);
                gl_FragColor = vec4(uColor, clamp(uBase + uGain * f, 0.0, 1.0));
                #include <colorspace_fragment>
  }`;

function draw() {
  if (!S || !S.el.clientWidth || !S.el.clientHeight) return;   // hidden tab (display:none) has no size
  S.controls.update(); S.renderer.render(S.scene, S.camera);
}

function clear(group) {
  for (const o of [...group.children]) { o.geometry?.dispose(); group.remove(o); }
}

function mount(el) {
  if (S) { if (S.el === el) return true; unmount(); }
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  } catch (e) { return false; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x000000, 0);
  Object.assign(renderer.domElement.style, { width: '100%', height: '100%', display: 'block', touchAction: 'none' });
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.environment = studio(renderer);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.02, 80);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08; controls.enablePan = false;
  controls.addEventListener('start', () => { auto = false; });

  const hemi = new THREE.HemisphereLight(0xcfd8e6, 0x1a1d24, 0.9); scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 2.4); key.position.set(4, 6, 5); scene.add(key);
  const fill = new THREE.DirectionalLight(0x9db7ff, 1.0); fill.position.set(-5, 2, -3); scene.add(fill);
  const room = [[hemi, 0.9], [key, 2.4], [fill, 1.0]];
  const glow = [0, 1, 2, 3].map(() => { const l = new THREE.PointLight(0xff4fa3, 0, 0, 2); scene.add(l); return l; });

  const steel = new THREE.MeshStandardMaterial({ color: 0x8a929c, metalness: 0.85, roughness: 0.38, side: THREE.DoubleSide, envMapIntensity: 1.0 });
  const copper = new THREE.MeshStandardMaterial({ color: 0xa8683a, metalness: 0.9, roughness: 0.42, side: THREE.DoubleSide, envMapIntensity: 0.8 });
  const rim = new THREE.ShaderMaterial({ vertexShader: RIM_VERT, fragmentShader: RIM_FRAG, transparent: true, depthWrite: false,
    side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
    uniforms: { uColor: { value: TE_COLD.clone() }, uPower: { value: 2.4 }, uGain: { value: 0.55 }, uBase: { value: 0.025 } } });
  const lineMat = LINE_COLORS.map(c => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.8, depthWrite: false, toneMapped: false,
    blending: THREE.AdditiveBlending }));

  const g = { open: new THREE.Group(), full: new THREE.Group(), coils: new THREE.Group(), plasma: new THREE.Group(), lines: new THREE.Group() };
  Object.values(g).forEach(x => scene.add(x));

  S = { el, renderer, scene, camera, controls, glow, room, steel, copper, rim, lineMat, g, raf: 0 };
  S.ro = new ResizeObserver(resize); S.ro.observe(el);
  resize(); setView(view);
  if (shot) setShot(shot);
  const loop = now => {
    S.raf = requestAnimationFrame(loop);
    if (!el.clientWidth || !el.clientHeight) return;   // hidden tab (display:none): no work
    if (auto) place(now);
    controls.update();
    renderer.render(scene, camera);
  };
  S.raf = requestAnimationFrame(loop);
  return true;
}

function unmount() {
  if (!S) return;
  cancelAnimationFrame(S.raf); S.ro.disconnect(); S.controls.dispose();
  Object.values(S.g).forEach(clear);
  S.renderer.dispose(); S.renderer.domElement.remove(); S = null;
}

// Camera on its preset orbit; while nobody has dragged it sways slowly about the opening.
function place(now) {
  const v = VIEWS[view], az = CAM_AZ + v.sway * Math.sin((now - t0) * 1e-3 * v.rate), c = Math.cos(v.elev);
  const d = view === 'cutaway' ? fit() : v.dist;
  S.camera.position.set(v.target[0] + d * c * Math.cos(az), v.target[1] + d * Math.sin(v.elev), v.target[2] - d * c * Math.sin(az));
}

// Distance that frames the whole vessel (8% margin) for any panel aspect ratio. Silhouette of a cylinder seen from
// elevation e: half height h cos e + r sin e, half width r; the near rim sits about r cos e closer to the camera.
function fit() {
  const v = VIEWS.cutaway, t = Math.tan(v.fov * Math.PI / 360), a = S.camera.aspect || 1, e = v.elev;
  const halfH = extent.h * Math.cos(e) + extent.r * Math.sin(e), halfW = extent.r;
  return Math.min(1.08 * Math.max(halfH / t, halfW / (t * a)) + 0.3 * extent.r * Math.cos(e), v.maxDist);
}

function setView(name) {
  view = VIEWS[name] ? name : 'cutaway';
  if (!S) return;
  const v = VIEWS[view], c = S.controls;
  S.camera.fov = v.fov; S.camera.updateProjectionMatrix();
  c.target.set(...v.target);
  c.minDistance = v.minDist; c.maxDistance = v.maxDist; c.minPolarAngle = v.minPolar; c.maxPolarAngle = v.maxPolar;
  S.g.open.visible = view === 'cutaway'; S.g.full.visible = view === 'port';
  S.room.forEach(([l, k]) => { l.intensity = k * v.room; });
  S.steel.envMapIntensity = v.env; S.copper.envMapIntensity = 0.8 * v.env;
  S.glow.forEach(l => { l.intensity = last.glow * v.glow; });
  S.lineMat.forEach(m => { m.opacity = v.lineOpacity; });
  if (last.lines) buildLines(last.lines);
  auto = true; t0 = performance.now(); place(t0); draw();
}

function setShot(s) {
  shot = s; last = { i: null, lines: null, glow: 0 };
  if (!S) return;
  Object.values(S.g).forEach(clear); S.glow.forEach(l => { l.intensity = 0; });
  const w = s.wall;
  if (w) {
    const keep = w.R.map((r, k) => r !== null && w.Z[k] !== null), R = w.R.filter((_, k) => keep[k]), Z = w.Z.filter((_, k) => keep[k]);
    // opening of 90 degrees centred on the camera's starting azimuth
    S.g.open.add(new THREE.Mesh(revolve(R, Z, { phi0: CAM_AZ + Math.PI / 4, dphi: 1.5 * Math.PI }), S.steel));
    S.g.full.add(new THREE.Mesh(revolve(R, Z), S.steel));
    extent = { r: Math.max(...R), h: Math.max(...Z.map(Math.abs)) };
  }
  // coil packs are opened with the wall so they never cross the view of the plasma; caps make the cut faces solid
  const phi0 = CAM_AZ + Math.PI / 4, dphi = 1.5 * Math.PI;
  if (s.coils) for (const [r0, r1, z0, z1] of coilPacks(s.coils)) {
    S.g.coils.add(new THREE.Mesh(revolve([r0, r1, r1, r0, r0], [z0, z0, z1, z1, z0], { phi0, dphi }), S.copper));
    for (const ph of [phi0, phi0 + dphi]) {
      const c = Math.cos(ph), sn = -Math.sin(ph), cap = new THREE.BufferGeometry();
      cap.setAttribute('position', new THREE.Float32BufferAttribute([r0 * c, z0, r0 * sn, r1 * c, z0, r1 * sn, r1 * c, z1, r1 * sn, r0 * c, z1, r0 * sn], 3));
      cap.setIndex([0, 1, 2, 0, 2, 3]); cap.computeVertexNormals();
      S.g.coils.add(new THREE.Mesh(cap, S.copper));
    }
  }
  if (auto) place(performance.now());
  draw();
}

function setSlice(i, lines) {
  if (!S || !shot) return;
  const R0 = shot.lcfs?.R?.[i] || [], Z0 = shot.lcfs?.Z?.[i] || [], R = [], Z = [];
  for (let k = 0; k < R0.length; k++) if (R0[k] !== null && Z0[k] !== null) { R.push(R0[k]); Z.push(Z0[k]); }
  clear(S.g.plasma);
  const me = shot.measured || {}, te = me.Te0_keV?.[i];
  const f = (te === null || te === undefined || !isFinite(te)) ? 0.3 : Math.min(Math.max((te - TE_RANGE[0]) / (TE_RANGE[1] - TE_RANGE[0]), 0), 1);
  const col = TE_COLD.clone().lerp(TE_HOT, f);
  if (R.length > 3) {
    if (Math.hypot(R[0] - R[R.length - 1], Z[0] - Z[Z.length - 1]) > 1e-6) { R.push(R[0]); Z.push(Z[0]); }
    S.rim.uniforms.uColor.value.copy(col);
    S.g.plasma.add(new THREE.Mesh(revolve(R, Z, { smooth: true }), S.rim));
  }
  // the steel picks up the plasma's colour: four lights on the magnetic-axis ring
  const Rm = me.R_mag_m?.[i] ?? 0.85, Zm = me.Z_mag_m?.[i] ?? 0;
  last.i = i; last.glow = R.length > 3 ? 1.0 + 1.6 * f : 0;
  S.glow.forEach((l, k) => {
    const ph = CAM_AZ + k * Math.PI / 2;
    l.position.set(Rm * Math.cos(ph), Zm, -Rm * Math.sin(ph)); l.color.copy(col); l.intensity = last.glow * VIEWS[view].glow;
  });
  if (lines === null || lines === undefined) { S.g.lines.visible = false; last.lines = null; }
  else { last.lines = lines; buildLines(lines); }
  if (document.hidden) draw();   // rAF is paused in a background tab; otherwise the loop draws the next frame
}

function buildLines(lines) {
  clear(S.g.lines); S.g.lines.visible = true;
  lines.forEach((l, k) => { const g = l && tube(l, VIEWS[view].tube); if (g) S.g.lines.add(new THREE.Mesh(g, S.lineMat[Math.min(k, S.lineMat.length - 1)])); });
}

function resize() {
  if (!S) return;
  const w = S.el.clientWidth, h = S.el.clientHeight;
  if (!w || !h) return;
  S.renderer.setSize(w, h, false);
  S.camera.aspect = w / h; S.camera.updateProjectionMatrix();
  if (auto) place(performance.now());   // refit to the new aspect; never after the user has taken the camera
  draw();
}

window.Vessel3D = { mount, setShot, setSlice, setView, resize };
document.dispatchEvent(new CustomEvent('vessel3d:ready'));
