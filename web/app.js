// MiDAS fiber viewer. Reads web/data/<day>/ written by src/web/store.py; see that file for the format.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const TZ_H = 8; // display time zone UTC+8
const NO_DATA = [0.78, 0.78, 0.78];
const NOT_LOADED = [0.93, 0.93, 0.93];
const MARGIN_L = 120, MARGIN_R = 8; // shared x layout of timeline + waterfall

// ---------- state ----------
const S = {
  meta: null, base: "", nst: 0, hours: [], overview: null, events: [],
  t: 0, playing: false, speed: 10, sel: null, dirty: true,
  norm: "rel", cmap: "seismic", gain: 1, exag: 2, arrivals: false,
  range: { abs: [0, 1], rel: [0, 30] }, k: 1, unit: "counts", logK: 0,
};

// ---------- colormaps ----------
// matplotlib "seismic": (0,0,.3) (0,0,1) (1,1,1) (1,0,0) (.5,0,0) at 0, .25, .5, .75, 1
const SEISMIC = [[0, 0, 0.3], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0.5, 0, 0]];
function cmapRGB(x) { // x in [0,1]
  x = Math.min(1, Math.max(0, x));
  if (S.cmap === "gray") { const g = 1 - x; return [g, g, g]; }
  const p = x * 4, i = Math.min(3, Math.floor(p)), f = p - i, a = SEISMIC[i], b = SEISMIC[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
// RMS uses the upper half of seismic (white → red); gray goes white → black
const rmsRGB = (n) => S.cmap === "gray" ? cmapRGB(n) : cmapRGB(0.5 + 0.5 * Math.min(1, Math.max(0, n)));
// signed event amplitude uses the full map, n in [-1, 1]
const sigRGB = (n) => cmapRGB((n + 1) / 2);
// arrival times: sequential "viridis" (matplotlib), early = dark purple, late = yellow
const VIRIDIS = [[0.267, 0.005, 0.329], [0.229, 0.322, 0.546], [0.128, 0.567, 0.551], [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]];
function seqRGB(x) {
  x = Math.min(1, Math.max(0, x));
  const p = x * 4, i = Math.min(3, Math.floor(p)), f = p - i, a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ---------- time ----------
const parseUTC = (s) => Date.parse(/Z$/.test(s) ? s : s + "Z");
let dayMs = 0;
const pad2 = (n) => String(n).padStart(2, "0");
function fmt(t, tz = TZ_H, date = true) {
  const d = new Date(dayMs + t * 1000 + tz * 3600e3);
  const hms = `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())}`;
  return date ? `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ${hms}` : hms;
}
const setT = (t) => { S.t = Math.min(S.meta.seconds - 0.001, Math.max(0, t)); S.dirty = true; ensureHour(Math.floor(S.t / 3600)); };

// ---------- values ----------
const logOf = (q) => S.meta.rms.qmin + (q - 1) / 254 * (S.meta.rms.qmax - S.meta.rms.qmin) + S.logK;
function rmsNorm(q, s) { // q (uint8) → [0,1] under current norm/range, or null
  const st = S.meta.stations[s];
  const v = S.norm === "abs" ? logOf(q) : 20 * (logOf(q) - S.logK - st.median);
  const [lo, hi] = S.range[S.norm];
  return (v - lo) / (hi - lo);
}
function activeEvent(t) {
  for (const e of S.events) if (t >= e.t0 && t < e.t0 + e.n / e.fs) return e;
  return null;
}
function eventValue(e, s, t) { // counts (or ε/s), null if not loaded
  if (!e.data) return null;
  const i = Math.floor((t - e.t0) * e.fs);
  return e.data[i * S.nst + s] * e.scale[s] / e.full * S.k;
}
// ---------- arrival times (events[k].arrivals, written by src/web/arrivals.py) ----------
const arrivalsOn = (e) => S.arrivals && e && e.arrivals;
function arrivalMax(e) { // colour range: latest pick that the plane-wave fit kept
  if (e._arrMax !== undefined) return e._arrMax;
  const out = new Set(e.arrivals.outlier || []);
  let mx = 0;
  e.arrivals.rel_ms.forEach((v, i) => { if (v !== null && !out.has(i) && v > mx) mx = v; });
  return (e._arrMax = Math.max(mx, 1));
}
function stationRGB(s, t) {
  const st = S.meta.stations[s];
  if (!st.has_data) return NO_DATA;
  const e = activeEvent(t);
  if (e && e.data && arrivalsOn(e)) { const v = e.arrivals.rel_ms[s]; return v === null ? NO_DATA : seqRGB(v / arrivalMax(e)); }
  if (e && e.data) return sigRGB(Math.max(-1, Math.min(1, eventValue(e, s, t) / (e.vmax * S.k / S.gain))));
  const h = S.hours[Math.floor(t / 3600)];
  if (!h) return NOT_LOADED;
  const q = h[(Math.floor(t) % 3600) * S.nst + s];
  return q ? rmsRGB(rmsNorm(q, s)) : NO_DATA;
}

// ---------- loading ----------
async function bin(name, Type) { const r = await fetch(S.base + name); if (!r.ok) throw new Error(name); return new Type(await r.arrayBuffer()); }
const loading = new Set();
async function ensureHour(h) {
  if (h < 0 || h > 23 || S.hours[h] || loading.has("h" + h)) return;
  loading.add("h" + h);
  S.hours[h] = await bin(S.meta.rms.files[h], Uint8Array);
  S.dirty = true; status();
}
async function ensureEvent(e) {
  if (e.data || loading.has(e.file)) return;
  loading.add(e.file);
  e.data = await bin(e.file, Int16Array);
  S.dirty = true; status();
}
function status() {
  const nh = S.hours.filter(Boolean).length, ne = S.events.filter((e) => e.data).length;
  $("status").textContent = nh === 24 && ne === S.events.length ? `${S.nst} stations · all data loaded` : `loading: ${nh}/24 h, ${ne}/${S.events.length} events`;
}

async function loadDay(day) {
  S.base = `data/${day}/`;
  const meta = await (await fetch(S.base + "meta.json")).json();
  S.overview = await bin(meta.overview.file, Uint8Array);
  S.meta = meta; S.nst = meta.stations.length; S.hours = new Array(24).fill(null); S.sel = null;
  dayMs = parseUTC(meta.start);
  S.k = meta.counts_to_strain_rate || 1; S.unit = meta.counts_to_strain_rate ? "ε/s" : meta.units;
  S.logK = Math.log10(S.k);
  S.range.abs = meta.rms.default_range.map((v) => +(v + S.logK).toFixed(2));
  S.range.rel = [0, 30];
  S.events = meta.events.map((e, i) => ({ ...e, i, t0: (parseUTC(e.start) - dayMs) / 1000, origin: (parseUTC(e.time) - dayMs) / 1000, data: null }));
  $("badge").textContent = meta.demo ? "DEMO · synthetic data" : "";
  buildScene();
  setT(S.events.length ? S.events[0].origin - 20 : 0);
  syncSettings(); buildWaterfallImage(); status();
  // background: every hour from the current one outward, then events
  const h0 = Math.floor(S.t / 3600);
  for (let d = 0; d < 24; d++) for (const h of [h0 + d, h0 - d]) await ensureHour(h);
  for (const e of S.events) await ensureEvent(e);
}

// ---------- 3D scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0xffffff);
$("scene").appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 1, 50000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = false;
let world = null, points = null, marker = null, markerLabel = null, markerFor = null, home = null;

const toV = (x, y, depth) => new THREE.Vector3(x, -depth * S.exag, -y);

function label(text, pos, h = 0.028) {
  const c = document.createElement("canvas"), g = c.getContext("2d");
  g.font = "28px Courier New, monospace";
  c.width = Math.ceil(g.measureText(text).width) + 8; c.height = 36;
  g.font = "28px Courier New, monospace"; g.fillStyle = "#fff"; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = "#000"; g.textBaseline = "middle"; g.fillText(text, 4, 19);
  const tex = new THREE.CanvasTexture(c);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, sizeAttenuation: false }));
  sp.scale.set(h * c.width / c.height, h, 1); sp.position.copy(pos); sp.renderOrder = 10;
  return sp;
}

function buildScene() {
  if (world) scene.remove(world);
  world = new THREE.Group(); scene.add(world);
  const m = S.meta, box = new THREE.Box3();

  for (const ln of m.lines) {
    const g = new THREE.BufferGeometry().setFromPoints(ln.points.map((p) => toV(...p)));
    const mat = ln.approx ? new THREE.LineDashedMaterial({ color: 0x000000, dashSize: 12, gapSize: 10 }) : new THREE.LineBasicMaterial({ color: 0x000000 });
    const line = new THREE.Line(g, mat); if (ln.approx) line.computeLineDistances();
    world.add(line); g.computeBoundingBox(); box.union(g.boundingBox);
    const name = ln.section;
    if (/^Hole .*\(down\)/.test(name)) {
      const top = ln.points[0], bot = ln.points[ln.points.length - 1];
      world.add(label(name.replace(" (down)", ""), toV(top[0], top[1], -40 / S.exag)));
      world.add(label(`${bot[2]} m`, toV(bot[0], bot[1], bot[2] + 25 / S.exag), 0.022));
    } else if (!/\((back|up)\)|Hole|lead-in/.test(name)) {
      const mid = ln.points[Math.floor(ln.points.length / 2)];
      world.add(label(name.replace(" (forth)", "") + (ln.approx ? " ?" : ""), toV(mid[0] + 40, mid[1], 0), 0.022));
    }
  }

  const pos = new Float32Array(S.nst * 3);
  m.stations.forEach((s, i) => toV(s.x, s.y, s.depth).toArray(pos, i * 3));
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(S.nst * 3), 3));
  points = new THREE.Points(g, new THREE.PointsMaterial({ size: 8, sizeAttenuation: false, vertexColors: true }));
  points.renderOrder = 2; world.add(points);
  // selected station: hollow black square drawn on top of everything + a label to its right
  const rc = document.createElement("canvas"); rc.width = rc.height = 64;
  const rg = rc.getContext("2d"); rg.strokeStyle = "#000"; rg.lineWidth = 8; rg.strokeRect(4, 4, 56, 56);
  marker = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(rc), depthTest: false, sizeAttenuation: false }));
  marker.scale.set(0.04, 0.04, 1); marker.renderOrder = 20; marker.visible = false; world.add(marker);
  markerLabel = null; markerFor = null;

  // ground grid (100 m), north arrow, scale note
  const size = Math.ceil(Math.max(box.max.x - box.min.x, box.max.z - box.min.z) / 100 + 4) * 100;
  const c = box.getCenter(new THREE.Vector3());
  const grid = new THREE.GridHelper(size, size / 100, 0xcccccc, 0xe4e4e4);
  grid.position.set(Math.round(c.x / 100) * 100, 0, Math.round(c.z / 100) * 100); world.add(grid);
  const a0 = new THREE.Vector3(box.min.x - 150, 0, box.max.z), a1 = a0.clone().add(new THREE.Vector3(0, 0, -200));
  world.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), a0, 200, 0x000000, 40, 20));
  world.add(label("N", a1.clone().add(new THREE.Vector3(0, 0, -30))));
  world.add(label("grid 100 m", a0.clone().add(new THREE.Vector3(0, 0, 40)), 0.022));

  home = { c, r: box.getSize(new THREE.Vector3()).length() };
  arrow = null; arrowFor = undefined; // rebuilt by the main loop
  view("reset");
}

function view(kind) {
  const { c, r } = home;
  controls.target.copy(c);
  if (kind === "top") camera.position.set(c.x, c.y + r * 1.3, c.z + 1);
  else if (kind === "side") camera.position.set(c.x, c.y, c.z + r * 1.3);
  else camera.position.set(c.x + r * 0.5, c.y + r * 0.6, c.z + r * 0.9);
  controls.update();
}
document.querySelectorAll("#views button").forEach((b) => b.onclick = () => view(b.dataset.view));

function resize() {
  const el = $("scene"), w = el.clientWidth, h = el.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  for (const id of ["timeline", "waterfall", "colorbar", "wave"]) {
    const cv = $(id); cv.width = cv.clientWidth * devicePixelRatio; cv.height = cv.clientHeight * devicePixelRatio;
  }
  drawColorbar(); S.dirty = true;
}
new ResizeObserver(resize).observe(document.body);

// screen-space picking: nearest station within 12 px
function pick(ev) {
  if (!points) return null;
  const rect = renderer.domElement.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const p = points.geometry.attributes.position.array, v = new THREE.Vector3();
  let best = null, bd = 144;
  for (let i = 0; i < S.nst; i++) {
    v.fromArray(p, i * 3).project(camera);
    if (v.z > 1) continue;
    const dx = (v.x + 1) / 2 * rect.width - mx, dy = (1 - v.y) / 2 * rect.height - my, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
const stationText = (i) => {
  const s = S.meta.stations[i];
  return `#${i} ${s.section}\nch ${s.ch0}–${s.ch1}  ${s.depth > 0 ? `depth ${s.depth.toFixed(0)} m` : "surface"}${s.approx ? "  (route approx.)" : ""}${s.has_data ? "" : "\nno data"}`;
};
let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => downAt = [e.clientX, e.clientY]);
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const i = pick(e); S.sel = i; S.dirty = true;
});
renderer.domElement.addEventListener("pointermove", (e) => {
  const i = pick(e), tip = $("tooltip");
  if (i === null) { tip.hidden = true; return; }
  const rect = renderer.domElement.getBoundingClientRect();
  tip.hidden = false; tip.textContent = stationText(i) + "\n" + valueText(i, S.t);
  tip.style.left = `${e.clientX - rect.left + 14}px`; tip.style.top = `${e.clientY - rect.top + 14}px`;
});
renderer.domElement.addEventListener("pointerleave", () => $("tooltip").hidden = true);

function valueText(i, t) {
  const s = S.meta.stations[i];
  if (!s.has_data) return "";
  const e = activeEvent(t);
  if (e && e.data && e.arrivals) {
    const v = e.arrivals.rel_ms[i], out = (e.arrivals.outlier || []).includes(i);
    const arr = v === null ? "no pick" : `arrival +${v.toFixed(1)} ms${out ? " (outlier, not in fit)" : ""}`;
    return `amplitude ${eventValue(e, i, t).toFixed(1)} ${S.unit}\n${arr}`;
  }
  if (e && e.data) return `amplitude ${eventValue(e, i, t).toFixed(1)} ${S.unit}`;
  const h = S.hours[Math.floor(t / 3600)], q = h && h[(Math.floor(t) % 3600) * S.nst + i];
  if (!q) return "";
  return S.norm === "abs" ? `log10 RMS ${logOf(q).toFixed(2)} (${S.unit})` : `${(20 * (logOf(q) - S.logK - s.median)).toFixed(1)} dB re. daily median`;
}

// ---------- colorbar & labels ----------
function modeIsEvent() { const e = activeEvent(S.t); return e && e.data ? e : null; }
function drawColorbar() {
  const cv = $("colorbar"), g = cv.getContext("2d"), w = cv.width, e = modeIsEvent();
  const arr = arrivalsOn(e);
  for (let x = 0; x < w; x++) {
    const n = x / (w - 1), c = arr ? seqRGB(n) : e ? sigRGB(n * 2 - 1) : rmsRGB(n);
    g.fillStyle = `rgb(${c.map((v) => Math.round(v * 255))})`; g.fillRect(x, 0, 1, cv.height);
  }
  const band = S.meta ? `${S.meta.filter.frequency[0]}–${S.meta.filter.frequency[1]} Hz bandpass` : "";
  $("arr-row").hidden = !(e && e.arrivals);
  $("arr-text").textContent = e && e.arrivals ? arrivalText(e) : "";
  if (arr) {
    const mx = arrivalMax(e);
    $("cb-lo").textContent = "0"; $("cb-mid").textContent = (mx / 2).toFixed(0); $("cb-hi").textContent = mx.toFixed(0);
    $("cb-title").textContent = `Arrival time relative to first pick (ms), E${e.i + 1}. Gray = no pick / no data. First pick ${fmt(e.t0 + e.arrivals.first)} (UTC+8).`;
    $("mode-line").textContent = `MODE: event E${e.i + 1} arrival times`;
  } else if (e) {
    const v = e.vmax * S.k / S.gain;
    $("cb-lo").textContent = (-v).toPrecision(3); $("cb-mid").textContent = "0"; $("cb-hi").textContent = v.toPrecision(3);
    $("cb-title").textContent = `Filtered amplitude (${S.unit}), ${band}, ${e.fs} Hz samples. Clip = 99th pct of |amplitude| in this event ÷ ${S.gain}`;
    $("mode-line").textContent = `MODE: event waveform E${e.i + 1}`;
  } else {
    const [lo, hi] = S.range[S.norm];
    $("cb-lo").textContent = lo; $("cb-mid").textContent = ((lo + hi) / 2).toFixed(1); $("cb-hi").textContent = hi;
    $("cb-title").textContent = S.norm === "abs"
      ? `log10 RMS amplitude (${S.unit}), ${band}, 1 s window`
      : `RMS / station daily median (dB), ${band}, 1 s window`;
    $("mode-line").textContent = "MODE: 1 s RMS";
  }
}

const angDiff = (a, b) => ((a - b + 540) % 360) - 180;
function arrivalText(e) {
  const a = e.arrivals;
  if (a.baz === undefined) return `E${e.i + 1} plane-wave fit: not enough picks (${a.n_picked}).`;
  const lines = [
    `E${e.i + 1} plane-wave fit to first-arrival picks`,
    `wave comes FROM ${a.baz.toFixed(0)}° ± ${a.baz_err.toFixed(0)}° (clockwise from N)`,
    `  ± from picks ${a.baz_err_pick.toFixed(0)}°, from ±${a.pos_sigma} m positions ${a.baz_err_position.toFixed(0)}°`,
    `apparent velocity ${a.v_app.toFixed(0)} ± ${a.v_app_err.toFixed(0)} m/s`,
    a.incidence !== undefined && a.incidence !== null ? `incidence ${a.incidence.toFixed(0)}° from vertical (hole picks)` : null,
    `fit residual ${a.residual_ms.toFixed(1)} ms rms, ${a.n_used}/${a.n_picked} picks used`,
    `fiber is mostly a NNW–SSE line: directions across it are weakly resolved`,
  ];
  if (e.source) lines.push(`DEMO truth: ${e.source.azimuth.toFixed(0)}° → error ${angDiff(a.baz, e.source.azimuth).toFixed(1)}°`);
  return lines.filter((l) => l !== null).join("\n");
}

// direction arrow on the ground plane: from the array centre toward where the wave comes from, with ±1σ wedge
let arrow = null, arrowFor = undefined;
function updateArrow(e) {
  if (arrow) { world.remove(arrow); arrow = null; }
  arrowFor = e;
  if (!e || !e.arrivals || e.arrivals.baz === undefined) return;
  const a = e.arrivals, c = home.c, L = home.r * 0.3, y = 2;
  const dirOf = (deg) => { const b = deg * Math.PI / 180; return new THREE.Vector3(Math.sin(b), 0, -Math.cos(b)); };
  const o = new THREE.Vector3(c.x, y, c.z);
  arrow = new THREE.Group();
  arrow.add(new THREE.ArrowHelper(dirOf(a.baz), o, L, 0x000000, L * 0.12, L * 0.06));
  if (a.baz_err < 90) for (const s of [-1, 1]) {
    const g = new THREE.BufferGeometry().setFromPoints([o, o.clone().add(dirOf(a.baz + s * a.baz_err).multiplyScalar(L))]);
    const ln = new THREE.Line(g, new THREE.LineDashedMaterial({ color: 0x000000, dashSize: 15, gapSize: 12 }));
    ln.computeLineDistances(); arrow.add(ln);
  }
  const tip = o.clone().add(dirOf(a.baz).multiplyScalar(L * 1.08));
  arrow.add(label(`wave from ${a.baz.toFixed(0)}°±${a.baz_err.toFixed(0)}° (dashed ±1σ)`, tip, 0.024));
  const lb2 = label(`v_app ${a.v_app.toFixed(0)}±${a.v_app_err.toFixed(0)} m/s`, tip, 0.024);
  lb2.center.set(0.5, 1.6); arrow.add(lb2);
  if (e.source) { // demo only: the true source direction, gray, to judge the estimate
    const g = new THREE.BufferGeometry().setFromPoints([o, o.clone().add(dirOf(e.source.azimuth).multiplyScalar(L * 0.8))]);
    arrow.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x999999 })));
    const lt = label(`demo truth ${e.source.azimuth.toFixed(0)}°`, o.clone().add(dirOf(e.source.azimuth).multiplyScalar(L * 0.85)), 0.02);
    arrow.add(lt);
  }
  world.add(arrow);
}

// ---------- station waveform panel (±30 s) ----------
function drawWave() {
  const cv = $("wave"), g = cv.getContext("2d"), W = cv.width, H = cv.height, dpr = devicePixelRatio;
  g.clearRect(0, 0, W, H);
  const i = S.sel;
  if (i === null || i === undefined) { $("station").textContent = "click a station on the fiber"; $("wave-title").textContent = ""; return; }
  $("station").textContent = stationText(i) + "\n" + valueText(i, S.t);
  if (!S.meta.stations[i].has_data) { $("wave-title").textContent = ""; return; }
  const e = modeIsEvent(), xs = [], ys = [];
  if (e) {
    const a = Math.max(0, Math.floor((S.t - 30 - e.t0) * e.fs)), b = Math.min(e.n, Math.ceil((S.t + 30 - e.t0) * e.fs));
    for (let k = a; k < b; k++) { xs.push(e.t0 + k / e.fs - S.t); ys.push(e.data[k * S.nst + i] * e.scale[i] / e.full * S.k); }
    $("wave-title").textContent = `filtered waveform (${S.unit}), ${S.meta.filter.frequency.join("–")} Hz, ±30 s around cursor`;
  } else {
    for (let d = -30; d <= 30; d++) {
      const t = Math.floor(S.t) + d, h = S.hours[Math.floor(t / 3600)];
      const q = h && t >= 0 && h[(t % 3600) * S.nst + i];
      if (q) { xs.push(d); ys.push(S.norm === "abs" ? logOf(q) : 20 * (logOf(q) - S.logK - S.meta.stations[i].median)); }
    }
    $("wave-title").textContent = S.norm === "abs" ? `log10 RMS (${S.unit}), 1 s window, ±30 s around cursor` : "RMS re. daily median (dB), 1 s window, ±30 s";
  }
  if (!ys.length) return;
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (e) { hi = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -hi; }
  if (hi === lo) { hi += 1; lo -= 1; }
  const px = (x) => (x + 30) / 60 * W, py = (y) => H - 14 * dpr - (y - lo) / (hi - lo) * (H - 28 * dpr);
  g.strokeStyle = "#bbb"; g.lineWidth = 1; g.beginPath(); g.moveTo(px(0), 0); g.lineTo(px(0), H); g.stroke();
  g.strokeStyle = "#000"; g.beginPath();
  xs.forEach((x, k) => k ? g.lineTo(px(x), py(ys[k])) : g.moveTo(px(x), py(ys[k]))); g.stroke();
  g.fillStyle = "#000"; g.font = `${11 * dpr}px Courier New`;
  g.fillText(hi.toPrecision(3), 2, 11 * dpr); g.fillText(lo.toPrecision(3), 2, H - 2 * dpr);
  g.fillText("cursor", px(0) + 3 * dpr, 11 * dpr);
  if (e && e.arrivals && e.arrivals.rel_ms[i] !== null) { // this station's first-arrival pick
    const x = px(e.t0 + e.arrivals.first + e.arrivals.rel_ms[i] / 1000 - S.t);
    if (x >= 0 && x <= W) {
      g.setLineDash([4 * dpr, 3 * dpr]); g.beginPath(); g.moveTo(x, 14 * dpr); g.lineTo(x, H); g.stroke(); g.setLineDash([]);
      g.fillText("pick", x + 3 * dpr, 24 * dpr);
    }
  }
}

// ---------- timeline + 2D waterfall (shared x axis: whole day) ----------
const xOf = (t, W) => MARGIN_L * devicePixelRatio + t / S.meta.seconds * (W - (MARGIN_L + MARGIN_R) * devicePixelRatio);
const tOf = (px, W) => (px * devicePixelRatio - MARGIN_L * devicePixelRatio) / (W - (MARGIN_L + MARGIN_R) * devicePixelRatio) * S.meta.seconds;
let wfImage = null, maxCurve = null;

function buildWaterfallImage() {
  const n = S.nst, ov = S.overview, img = new ImageData(1440, n);
  maxCurve = new Float32Array(1440);
  for (let m = 0; m < 1440; m++) {
    let mx = 0;
    for (let s = 0; s < n; s++) {
      const q = ov[m * n + s], o = (s * 1440 + m) * 4;
      let c = NO_DATA;
      if (q && S.meta.stations[s].has_data) { c = rmsRGB(rmsNorm(q, s)); if (q > mx) mx = q; }
      img.data[o] = c[0] * 255; img.data[o + 1] = c[1] * 255; img.data[o + 2] = c[2] * 255; img.data[o + 3] = 255;
    }
    maxCurve[m] = mx;
  }
  wfImage = document.createElement("canvas"); wfImage.width = 1440; wfImage.height = n;
  wfImage.getContext("2d").putImageData(img, 0, 0);
  $("wf-title").textContent = `2D waterfall: stations in fiber order (rows) × time (UTC+8). Per-minute max of the 1 s RMS, same color scale as above. Click to jump.`;
}

function timeTicks(g, W, H, y) {
  const dpr = devicePixelRatio;
  g.fillStyle = "#000"; g.strokeStyle = "#000"; g.font = `${11 * dpr}px Courier New`;
  for (let h = 0; h <= 24; h += 2) {
    const x = xOf(h * 3600, W);
    g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + 4 * dpr); g.stroke();
    if (h < 24) g.fillText(fmt(h * 3600, TZ_H, false).slice(0, 5), x + 2, y + 13 * dpr);
  }
}

function drawTimeline() {
  const cv = $("timeline"), g = cv.getContext("2d"), W = cv.width, H = cv.height, dpr = devicePixelRatio;
  g.clearRect(0, 0, W, H);
  const top = 4 * dpr, bot = H - 18 * dpr;
  for (const e of S.events) { // event windows
    g.fillStyle = "#e0e0e0"; const x0 = xOf(e.t0, W), x1 = xOf(e.t0 + e.n / e.fs, W);
    g.fillRect(x0, top, Math.max(2, x1 - x0), bot - top);
    g.fillStyle = "#000"; g.font = `${10 * dpr}px Courier New`; g.fillText(`E${e.i + 1}`, x1 + 2, top + 10 * dpr);
  }
  if (maxCurve) {
    g.strokeStyle = "#000"; g.lineWidth = 1; g.beginPath();
    for (let m = 0; m < 1440; m++) { const x = xOf(m * 60 + 30, W), y = bot - maxCurve[m] / 255 * (bot - top); m ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
  }
  g.strokeStyle = "#000"; g.beginPath(); g.moveTo(xOf(0, W), bot); g.lineTo(xOf(S.meta.seconds, W), bot); g.stroke();
  timeTicks(g, W, H, bot);
  g.fillStyle = "#000"; g.font = `${11 * dpr}px Courier New`;
  g.fillText("max RMS", 4, top + 12 * dpr); g.fillText("(all stations)", 4, top + 26 * dpr); g.fillText("UTC+8", 4, bot + 13 * dpr);
  const x = xOf(S.t, W); g.fillStyle = "#d00"; g.fillRect(x - dpr, 0, 2 * dpr, H);
}

function drawWaterfall() {
  const cv = $("waterfall"), g = cv.getContext("2d"), W = cv.width, H = cv.height, dpr = devicePixelRatio;
  g.clearRect(0, 0, W, H);
  if (!wfImage) return;
  const x0 = xOf(0, W), x1 = xOf(S.meta.seconds, W);
  g.imageSmoothingEnabled = false; g.drawImage(wfImage, x0, 0, x1 - x0, H);
  // section boundaries + names in the left margin
  g.font = `${10 * dpr}px Courier New`; g.fillStyle = "#000"; g.strokeStyle = "#000";
  let prev = null, lastY = -1e9;
  S.meta.stations.forEach((s, i) => {
    if (s.section === prev) return; prev = s.section;
    const y = i / S.nst * H; g.beginPath(); g.moveTo(x0 - 6 * dpr, y); g.lineTo(x0, y); g.stroke();
    if (y - lastY < 10 * dpr) return; // too short to label without overlap
    g.fillText(s.section.slice(0, 18), 2, y + 9 * dpr); lastY = y;
  });
  const x = xOf(S.t, W); g.fillStyle = "#000"; g.fillRect(x - dpr / 2, 0, dpr, H);
  if (S.sel !== null && S.sel !== undefined) { const y = (S.sel + 0.5) / S.nst * H; g.fillRect(x0, y - dpr / 2, x1 - x0, dpr); }
}

function dragTime(cv, alsoStation) {
  const at = (e) => {
    const r = cv.getBoundingClientRect(); setT(tOf(e.clientX - r.left, cv.width));
    if (alsoStation) { const s = Math.floor((e.clientY - r.top) / r.height * S.nst); if (s >= 0 && s < S.nst) S.sel = s; }
  };
  cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); at(e); });
  cv.addEventListener("pointermove", (e) => { if (cv.hasPointerCapture(e.pointerId)) at(e); });
}
dragTime($("timeline"), false);
dragTime($("waterfall"), true);

// ---------- controls ----------
$("play").onclick = () => { S.playing = !S.playing; $("play").textContent = S.playing ? "pause" : "play"; $("play").classList.toggle("on", S.playing); };
$("speed").onchange = (e) => S.speed = +e.target.value;
document.querySelectorAll("[data-step]").forEach((b) => b.onclick = () => setT(S.t + +b.dataset.step));
$("prev-ev").onclick = () => { const e = [...S.events].reverse().find((e) => e.origin < S.t - 21); if (e) setT(e.origin - 20); };
$("next-ev").onclick = () => { const e = S.events.find((e) => e.origin > S.t + 20.5); if (e) setT(e.origin - 20); };
$("goto").onchange = (e) => {
  const m = e.target.value.trim().match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?$/);
  if (!m) return;
  const local = +m[1] * 3600 + +m[2] * 60 + +(m[3] || 0);
  setT(((local - TZ_H * 3600 - (dayMs / 1000) % 86400) % 86400 + 86400) % 86400);
};
document.querySelectorAll("input[name=norm]").forEach((r) => r.onchange = () => { S.norm = r.value; syncSettings(); });
function syncSettings() {
  const [lo, hi] = S.range[S.norm];
  $("lo").value = lo; $("hi").value = hi; $("range-unit").textContent = S.norm === "abs" ? `log10 ${S.unit}` : "dB";
  buildWaterfallImage(); drawColorbar(); S.dirty = true;
}
const setRange = () => { S.range[S.norm] = [+$("lo").value, +$("hi").value]; syncSettings(); };
$("lo").onchange = setRange; $("hi").onchange = setRange;
$("range-reset").onclick = () => { S.range[S.norm] = S.norm === "abs" ? S.meta.rms.default_range.map((v) => +(v + S.logK).toFixed(2)) : [0, 30]; syncSettings(); };
$("gain").oninput = (e) => { S.gain = +e.target.value; $("gain-v").textContent = `×${S.gain.toFixed(1)}`; drawColorbar(); S.dirty = true; };
$("cmap").onchange = (e) => { S.cmap = e.target.value; syncSettings(); };
$("arr-toggle").onchange = (e) => { S.arrivals = e.target.checked; drawColorbar(); S.dirty = true; };
$("exag").onchange = (e) => { S.exag = +e.target.value; buildScene(); S.dirty = true; };
$("day").onchange = (e) => { const u = new URL(location); u.searchParams.set("day", e.target.value); location = u; };
window.addEventListener("keydown", (e) => { if (e.target.tagName !== "INPUT" && e.code === "Space") { e.preventDefault(); $("play").click(); } });

// ---------- main loop ----------
let last = performance.now(), lastEvent = undefined;
function frame(now) {
  const dt = (now - last) / 1000; last = now;
  if (S.meta && points) {
    if (S.playing) { setT(S.t + dt * S.speed); if (S.t >= S.meta.seconds - 0.01) $("play").click(); }
    const e = activeEvent(S.t);
    if (e) ensureEvent(e);
    const ev = modeIsEvent();
    if (ev !== lastEvent) { lastEvent = ev; drawColorbar(); }
    if (ev !== arrowFor) updateArrow(ev);
    if (S.dirty) {
      const col = points.geometry.attributes.color;
      for (let i = 0; i < S.nst; i++) col.array.set(stationRGB(i, S.t), i * 3);
      col.needsUpdate = true;
      const has = S.sel !== null && S.sel !== undefined;
      if (markerFor !== (has ? S.sel : null)) {
        if (markerLabel) world.remove(markerLabel);
        markerLabel = null; markerFor = has ? S.sel : null;
        if (has) {
          const s = S.meta.stations[S.sel];
          marker.position.fromArray(points.geometry.attributes.position.array, S.sel * 3);
          markerLabel = label(`◀ #${S.sel} ch ${s.ch0}–${s.ch1}`, marker.position, 0.026);
          markerLabel.center.set(-0.08, 0.5); markerLabel.renderOrder = 21; world.add(markerLabel);
        }
      }
      marker.visible = has;
      $("clock").textContent = `${fmt(S.t)} (UTC+8)`; $("clock").title = `${fmt(S.t, 0)} UTC`;
      drawTimeline(); drawWaterfall(); drawWave();
      S.dirty = false;
    }
    controls.update(); renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
}

(async () => {
  const days = await (await fetch("data/index.json")).json();
  const want = new URL(location).searchParams.get("day");
  const day = days.includes(want) ? want : days.filter((d) => d !== "demo").at(-1) || days[0];
  $("day").innerHTML = days.map((d) => `<option ${d === day ? "selected" : ""}>${d}</option>`).join("");
  resize();
  await loadDay(day).catch((err) => { $("status").textContent = `error: ${err.message}`; throw err; });
})();
requestAnimationFrame(frame);
