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
  evmode: "envelope", envscale: "snr", smooth: 0.1, trail: 0.2,
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
// sample position u = (t - t0)·fs is fractional: values are linearly interpolated between the 50 Hz samples,
// so slow-motion playback moves smoothly instead of stepping every 20 ms.
function lerpAt(arr, e, s, u) {
  const i = Math.max(0, Math.min(e.n - 1, Math.floor(u))), j = Math.min(e.n - 1, i + 1), f = Math.max(0, Math.min(1, u - i));
  return arr[i * S.nst + s] * (1 - f) + arr[j * S.nst + s] * f;
}
function eventValue(e, s, t) { // counts (or ε/s), null if not loaded
  if (!e.data) return null;
  return lerpAt(e.data, e, s, (t - e.t0) * e.fs) * e.scale[s] / e.full * S.k;
}

// ---------- envelope (computed in the browser from the int16 waveform) ----------
// env = sqrt(centred moving mean of x² over S.smooth seconds), in counts. Computed only for the event under the
// cursor (Float32, n × n_st ≈ 40 MB for a 600 s window of 336 stations) in one O(n) pass, instead of shipping a
// second file per event. Color scale either in counts (median .. 99.9th pct of all stations) or, per station, as
// SNR in dB = 20·log10(env / its pre-event noise) over SNR_DB (÷ clip on the top), so loud surface stations and
// quiet borehole stations light up alike (a linear scale is set by the strong borehole arrivals and leaves the
// surface white). arrival = first sample where the station mean of env / (station noise median) exceeds ARRIVAL_RATIO.
const ARRIVAL_RATIO = 2.5, NOISE_S = 200, SNR_DB = [6, 26];
function ensureEnvelope(e) {
  if (!e.data) return null;
  if (e.env && e.envSmooth === S.smooth) return e.env;
  for (const o of S.events) if (o !== e) o.env = null; // keep only one in memory
  const n = e.n, ns = S.nst, w = Math.max(1, Math.round(S.smooth * e.fs)), h = Math.floor(w / 2);
  const env = new Float32Array(n * ns), sq = new Float64Array(n + 1);
  for (let s = 0; s < ns; s++) {
    const c = e.scale[s] / e.full;
    if (!c) continue;
    for (let i = 0; i < n; i++) { const x = e.data[i * ns + s] * c; sq[i + 1] = sq[i] + x * x; }
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, Math.min(n - w, i - h)), b = a + w;
      env[i * ns + s] = Math.sqrt(Math.max(0, (sq[b] - sq[a]) / w));
    }
  }
  // statistics from every 7th / 31st sample (enough for percentiles, keeps this fast).
  // noise[s] = median envelope of station s in the first NOISE_S seconds of the window (before the origin).
  const noise = new Float64Array(ns), nNoise = Math.min(n, Math.round(NOISE_S * e.fs));
  for (let s = 0; s < ns; s++) {
    if (!e.scale[s]) continue;
    const v = []; for (let i = 0; i < nNoise; i += 7) v.push(env[i * ns + s]);
    v.sort((a, b) => a - b); noise[s] = v[v.length >> 1] || 1;
  }
  const all = [];
  for (let i = 0; i < n; i += 31) for (let s = 0; s < ns; s++) if (e.scale[s]) all.push(env[i * ns + s]);
  all.sort((a, b) => a - b);
  const pct = (p) => all[Math.min(all.length - 1, Math.floor(p / 100 * all.length))];
  let arrival = null;
  for (let i = 0; i < n && arrival === null; i++) {
    let m = 0, k = 0;
    for (let s = 0; s < ns; s++) if (e.scale[s]) { m += env[i * ns + s] / noise[s]; k++; }
    if (k && m / k > ARRIVAL_RATIO) arrival = e.t0 + i / e.fs;
  }
  Object.assign(e, { env, noise, envSmooth: S.smooth, arrival, envCounts: [pct(50), pct(99.9)] });
  setTimeout(drawColorbar);
  return env;
}
const envValue = (e, s, t) => lerpAt(e.env, e, s, (t - e.t0) * e.fs) * S.k; // counts (or ε/s)
function envLoHi(e) { // color range in the units of S.envscale (counts / ε/s, or dB re. station noise)
  if (S.envscale === "snr") return [SNR_DB[0], SNR_DB[0] + (SNR_DB[1] - SNR_DB[0]) / S.gain];
  const [lo, top] = e.envCounts;
  return [lo * S.k, Math.max(lo * 1.5, top / S.gain) * S.k];
}
function envNorm(e, s, t) { // [0,1] over the color range, with an optional fading trail (peak hold over S.trail s)
  const [lo, hi] = envLoHi(e), snr = S.envscale === "snr", div = e.noise[s] * S.k;
  const n = (v) => ((snr ? 20 * Math.log10(Math.max(1e-9, v / div)) : v) - lo) / (hi - lo);
  let best = n(envValue(e, s, t));
  if (S.trail > 0) {
    const dt = 1 / e.fs;
    for (let a = dt; a < S.trail; a += dt) best = Math.max(best, n(envValue(e, s, t - a)) * (1 - a / S.trail));
  }
  return Math.max(0, Math.min(1, best));
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
  if (e && e.data && S.evmode === "envelope" && ensureEnvelope(e)) return rmsRGB(envNorm(e, s, t));
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
  stationDistances();
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
let lineMats = [], world = null, points = null, marker = null, markerLabel = null, markerFor = null, home = null;

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
  world = new THREE.Group(); scene.add(world); lineMats = [];
  const m = S.meta, box = new THREE.Box3();

  for (const ln of m.lines) {
    const g = new THREE.BufferGeometry().setFromPoints(ln.points.map((p) => toV(...p)));
    const mat = ln.approx ? new THREE.LineDashedMaterial({ color: 0x000000, dashSize: 12, gapSize: 10 }) : new THREE.LineBasicMaterial({ color: 0x000000 });
    lineMats.push(mat); // depthTest is switched off in envelope mode only (see frame), so the fiber stays visible over white stations
    const line = new THREE.Line(g, mat); line.renderOrder = 3; if (ln.approx) line.computeLineDistances();
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
  if (e && e.data) {
    let txt = `amplitude ${eventValue(e, i, t).toFixed(1)} ${S.unit}`;
    if (e.env && S.evmode === "envelope") txt += `\nenvelope ${envValue(e, i, t).toFixed(1)} ${S.unit} (SNR ${(20 * Math.log10(envValue(e, i, t) / (e.noise[i] * S.k))).toFixed(1)} dB)`;
    if (e.arrivals) {
      const v = e.arrivals.rel_ms[i], out = (e.arrivals.outlier || []).includes(i);
      txt += "\n" + (v === null ? "no pick" : `arrival +${v.toFixed(1)} ms${out ? " (outlier, not in fit)" : ""}`);
    }
    return txt;
  }
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
    const n = x / (w - 1), c = arr ? seqRGB(n) : e && S.evmode !== "envelope" ? sigRGB(n * 2 - 1) : rmsRGB(n);
    g.fillStyle = `rgb(${c.map((v) => Math.round(v * 255))})`; g.fillRect(x, 0, 1, cv.height);
  }
  const band = S.meta ? `${S.meta.filter.frequency[0]}–${S.meta.filter.frequency[1]} Hz bandpass` : "";
  $("arr-row").hidden = !(e && e.arrivals);
  renderArrivals($("arr-text"), e);
  if (arr) {
    const mx = arrivalMax(e);
    $("cb-lo").textContent = "0"; $("cb-mid").textContent = (mx / 2).toFixed(0); $("cb-hi").textContent = mx.toFixed(0);
    $("cb-title").textContent = `Arrival time relative to first pick (ms), E${e.i + 1}. Gray = no pick / no data. First pick ${fmt(e.t0 + e.arrivals.first)} (UTC+8).`;
    $("mode-line").textContent = `MODE: event E${e.i + 1} arrival times`;
  } else if (e && S.evmode === "envelope" && e.env) {
    const [lo, hi] = envLoHi(e), snr = S.envscale === "snr";
    const f = (v) => snr ? v.toFixed(1) : v.toPrecision(3);
    $("cb-lo").textContent = f(lo); $("cb-mid").textContent = f((lo + hi) / 2); $("cb-hi").textContent = f(hi);
    $("cb-title").textContent = (snr
      ? `Envelope SNR (dB) = 20·log10(envelope / station noise), ${band}, smoothing ${S.smooth} s. `
        + `Noise = each station's median envelope in the first ${NOISE_S} s of the window. Range ${SNR_DB[0]}–${SNR_DB[1]} dB, top ÷ ${S.gain}. `
      : `Envelope of filtered amplitude (${S.unit}), ${band}, smoothing ${S.smooth} s. Low end = median envelope of this event window, `
        + `high end = 99.9th pct ÷ ${S.gain}. `)
      + `Envelope = sqrt of centred moving mean of x², linear between ${e.fs} Hz samples.`
      + (S.trail ? ` Trail: peak of the last ${S.trail} s, fading linearly to 0.` : "");
    $("mode-line").textContent = `MODE: event envelope E${e.i + 1}`;
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
// plane-wave fit summary as a heading + bullet list (sub-bullets for the ± breakdown)
function arrivalList(e) {
  const a = e.arrivals, title = `E${e.i + 1} plane-wave fit to first-arrival picks`;
  if (a.baz === undefined) return { title, items: [[`not enough picks (${a.n_picked})`]] };
  const items = [
    [`wave comes FROM: ${a.baz.toFixed(0)}° ± ${a.baz_err.toFixed(0)}° (clockwise from N)`,
      [`from picks: ±${a.baz_err_pick.toFixed(0)}°`, `from ±${a.pos_sigma} m channel positions: ±${a.baz_err_position.toFixed(0)}°`]],
    [`apparent velocity: ${a.v_app.toFixed(0)} ± ${a.v_app_err.toFixed(0)} m/s`],
  ];
  if (a.incidence !== undefined && a.incidence !== null) items.push([`incidence: ${a.incidence.toFixed(0)}° from vertical (hole picks)`]);
  items.push([`fit residual: ${a.residual_ms.toFixed(1)} ms rms`]);
  items.push([`picks used: ${a.n_used} / ${a.n_picked}`]);
  items.push([`note: the fiber is mostly a NNW–SSE line, so directions across it are weakly resolved`]);
  if (e.source) items.push([`DEMO truth: ${e.source.azimuth.toFixed(0)}° (error ${angDiff(a.baz, e.source.azimuth).toFixed(1)}°)`]);
  return { title, items };
}
function renderArrivals(el, e) {
  el.replaceChildren();
  if (!e || !e.arrivals) return;
  const { title, items } = arrivalList(e), ul = (rows) => {
    const u = document.createElement("ul");
    for (const [text, sub] of rows) {
      const li = document.createElement("li"); li.textContent = text;
      if (sub) li.append(ul(sub.map((t) => [t])));
      u.append(li);
    }
    return u;
  };
  const h = document.createElement("div"); h.textContent = title;
  el.append(h, ul(items));
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
  if (e && e.env && S.evmode === "envelope") {
    g.strokeStyle = "#888"; g.beginPath();
    xs.forEach((x, k) => { const y = py(envValue(e, i, x + S.t)); k ? g.lineTo(px(x), y) : g.moveTo(px(x), y); }); g.stroke();
    $("wave-title").textContent += `; gray = envelope (smoothing ${S.smooth} s)`;
  }
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
  $("wf-title").textContent = `2D waterfall: stations in fiber order (rows) × time (UTC+8). Per-minute max of the 1 s RMS, same color scale as above. Click or drag to move the crosshair (time + station).`;
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
  const e = modeIsEvent();
  evUI(e);
  if (e) return drawEventView(e);
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

function dragTime(cv) {
  const at = (e) => { const r = cv.getBoundingClientRect(); setT(tOf(e.clientX - r.left, cv.width)); };
  cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); at(e); });
  cv.addEventListener("pointermove", (e) => { if (cv.hasPointerCapture(e.pointerId)) at(e); });
}
dragTime($("timeline"));

// ---------- event zoom waterfall (replaces the day waterfall while the cursor is inside an event window) ----------
// Rows are either stations in fiber order, or one panel per fiber section with y = distance along the section (m).
// In section mode every panel shares one m/px scale, so equal slopes mean equal apparent velocities.
const EV = { span: 3, rows: "sections", norm: "clip", L: null, Lkey: "", anchor: null, dragEv: false };
const EV_TOP = 4, EV_BOT = 30; // css px above / below the image (bottom: time axis)
let SECS = []; // fiber sections that carry stations: {name, idx (station ids sorted by dist), len, hole, lo, hi}

// distance of each station along its section: holes = depth; others = arc length of the drawn route,
// measured from the section's first channel, except "(back)" legs which are flipped to share their forth leg's axis
function stationDistances() {
  const lines = Object.fromEntries(S.meta.lines.map((ln) => [ln.section, ln.points]));
  const arc = (pts) => { const a = [0]; for (let i = 1; i < pts.length; i++) a.push(a[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2])); return a; };
  const cache = {};
  SECS = [];
  for (const s of S.meta.stations) {
    const hole = /^Hole/.test(s.section), pts = lines[s.section];
    if (hole || !pts) s.dist = s.depth;
    else {
      const A = cache[s.section] ||= arc(pts);
      let best = Infinity, d = 0;
      for (let i = 1; i < pts.length; i++) {
        const [ax, ay] = pts[i - 1], bx = pts[i][0] - ax, by = pts[i][1] - ay, L2 = bx * bx + by * by || 1;
        const u = Math.max(0, Math.min(1, ((s.x - ax) * bx + (s.y - ay) * by) / L2));
        const q = Math.hypot(s.x - ax - u * bx, s.y - ay - u * by);
        if (q < best) { best = q; d = A[i - 1] + u * (A[i] - A[i - 1]); }
      }
      s.dist = /\(back\)/.test(s.section) ? A.at(-1) - d : d;
    }
    let sec = SECS.at(-1);
    if (!sec || sec.name !== s.section) {
      const len = hole ? Math.max(...(pts || [[0, 0, s.depth]]).map((p) => p[2])) : (cache[s.section] || [0]).at(-1);
      SECS.push(sec = { name: s.section, idx: [], len, hole, has: false });
    }
    sec.idx.push(s.id ?? S.meta.stations.indexOf(s)); sec.has ||= s.has_data;
  }
  for (const sec of SECS) {
    sec.idx.sort((a, b) => S.meta.stations[a].dist - S.meta.stations[b].dist);
    sec.len = Math.max(sec.len, S.meta.stations[sec.idx.at(-1)].dist);
    const lo = S.meta.stations[sec.idx[0]], hi = S.meta.stations[sec.idx.at(-1)];
    sec.lo = lo; sec.hi = hi;
  }
  EV.Lkey = "";
}
const secOf = (name) => SECS.find((s) => s.name === name);
const nearestIn = (sec, d) => { let b = sec.idx[0], bd = Infinity; for (const i of sec.idx) { const q = Math.abs(S.meta.stations[i].dist - d); if (q < bd) { bd = q; b = i; } } return b; };

// row layout in device px: rows[y] = station id or -1, panels = [{sec, y0, y1, mpp}]
function evLayout(H) {
  const dpr = devicePixelRatio, key = `${H}|${EV.rows}|${dpr}`;
  if (EV.L && EV.Lkey === key) return EV.L;
  const top = Math.round(EV_TOP * dpr), bot = Math.round(H - EV_BOT * dpr), ph = bot - top;
  const rows = new Int32Array(ph).fill(-1), panels = [];
  if (EV.rows === "fiber") {
    for (let y = 0; y < ph; y++) rows[y] = Math.min(S.nst - 1, Math.floor(y / ph * S.nst));
    for (const sec of SECS) {
      const ids = [...sec.idx].sort((a, b) => a - b);
      panels.push({ sec, y0: ids[0] / S.nst * ph, y1: (ids.at(-1) + 1) / S.nst * ph, mpp: null });
    }
  } else {
    const show = SECS.filter((s) => s.has), gap = 6 * dpr, minH = 16 * dpr;
    let avail = ph - gap * (show.length - 1), total = show.reduce((a, s) => a + s.len, 0);
    let mpp = total / avail;
    const small = show.filter((s) => s.len / mpp < minH);
    if (small.length) mpp = (total - small.reduce((a, s) => a + s.len, 0)) / (avail - small.length * minH);
    let y = 0;
    for (const sec of show) {
      const h = sec.len / mpp < minH ? minH : sec.len / mpp, p = { sec, y0: y, y1: y + h, mpp: sec.len / h };
      const spacing = sec.len / Math.max(1, sec.idx.length - 1);
      for (let r = Math.floor(y); r < Math.min(ph, Math.floor(y + h)); r++) {
        const d = (r - y + 0.5) * p.mpp, i = nearestIn(sec, d);
        if (Math.abs(S.meta.stations[i].dist - d) <= Math.max(spacing, p.mpp)) rows[r] = i;
      }
      panels.push(p); y += h + gap;
    }
  }
  EV.L = { top, bot, ph, rows, panels }; EV.Lkey = key;
  return EV.L;
}

// x mapping inside the event view
// window centre: the cursor, except while the user positions the crosshair by clicking/dragging in this view
// (EV.anchor keeps the window still so the picture does not slide under the pointer). Cleared on play or when
// the cursor leaves the window.
const evC = () => {
  if (EV.anchor !== null && (S.playing || Math.abs(S.t - EV.anchor) > EV.span)) EV.anchor = null;
  return EV.anchor ?? S.t;
};
const evX = (t, W) => MARGIN_L * devicePixelRatio + (t - (evC() - EV.span)) / (2 * EV.span) * (W - (MARGIN_L + MARGIN_R) * devicePixelRatio);
const evT = (px, W) => evC() - EV.span + (px - MARGIN_L * devicePixelRatio) / (W - (MARGIN_L + MARGIN_R) * devicePixelRatio) * 2 * EV.span;
// canvas y (device px) → {sec, dist, st}; null in a gap
function evLoc(y) {
  const L = EV.L; if (!L) return null;
  const r = Math.floor(y - L.top); if (r < 0 || r >= L.ph) return null;
  const st = L.rows[r];
  if (EV.rows === "fiber") return st < 0 ? null : { sec: S.meta.stations[st].section, dist: S.meta.stations[st].dist, st };
  const p = L.panels.find((p) => r >= p.y0 && r < p.y1); if (!p) return null;
  const dist = (r - p.y0 + 0.5) * p.mpp;
  return { sec: p.sec.name, dist, st: st >= 0 ? st : nearestIn(p.sec, dist) };
}
function evY(sec, dist) { // inverse of evLoc, device px
  const L = EV.L; if (!L) return null;
  if (EV.rows === "fiber") { const s = secOf(sec); return s ? L.top + (nearestIn(s, dist) + 0.5) / S.nst * L.ph : null; }
  const p = L.panels.find((p) => p.sec.name === sec); return p ? L.top + p.y0 + dist / p.mpp : null;
}

let LUT = null, lutFor = "";
function evLUT() { // 256 colors over n in [-1, 1]
  if (lutFor === S.cmap) return LUT;
  LUT = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) LUT.set(sigRGB(i / 255 * 2 - 1).map((v) => Math.round(v * 255)), i * 3);
  lutFor = S.cmap; return LUT;
}

function drawEventView(e) {
  const cv = $("waterfall"), g = cv.getContext("2d"), W = cv.width, H = cv.height, dpr = devicePixelRatio;
  g.clearRect(0, 0, W, H);
  const L = evLayout(H), x0 = Math.round(MARGIN_L * dpr), x1 = Math.round(W - MARGIN_R * dpr), pw = x1 - x0;
  if (pw < 10 || L.ph < 10) return;
  const nst = S.nst, ta = evC() - EV.span;
  const colK = new Int32Array(pw);
  let kmin = Infinity, kmax = -Infinity;
  for (let x = 0; x < pw; x++) {
    const k = Math.floor((ta + (x + 0.5) / pw * 2 * EV.span - e.t0) * e.fs);
    colK[x] = k >= 0 && k < e.n ? k : -1;
    if (colK[x] >= 0) { kmin = Math.min(kmin, k); kmax = Math.max(kmax, k); }
  }
  // per-station factor: int16 → n in [-1, 1]
  const f = new Float32Array(nst);
  for (let s = 0; s < nst; s++) {
    if (EV.norm === "trace") {
      let m = 0; for (let k = kmin; k <= kmax; k++) m = Math.max(m, Math.abs(e.data[k * nst + s]));
      f[s] = m ? S.gain / m : 0;
    } else f[s] = e.scale[s] / e.full * S.gain / e.vmax;
  }
  const lut = evLUT(), img = g.createImageData(pw, L.ph), D = img.data;
  const gapC = [255, 255, 255], noC = NO_DATA.map((v) => v * 255), outC = NOT_LOADED.map((v) => v * 255);
  for (let r = 0; r < L.ph; r++) {
    const s = L.rows[r], has = s >= 0 && S.meta.stations[s].has_data;
    for (let x = 0; x < pw; x++) {
      const o = (r * pw + x) * 4, k = colK[x];
      let c;
      if (s < 0) c = gapC; else if (!has) c = noC; else if (k < 0) c = outC;
      else {
        const n = Math.max(-1, Math.min(1, e.data[k * nst + s] * f[s]));
        const li = Math.round((n + 1) / 2 * 255) * 3; D[o] = lut[li]; D[o + 1] = lut[li + 1]; D[o + 2] = lut[li + 2]; D[o + 3] = 255; continue;
      }
      D[o] = c[0]; D[o + 1] = c[1]; D[o + 2] = c[2]; D[o + 3] = 255;
    }
  }
  g.putImageData(img, x0, L.top);

  // frames, section labels, distance ticks
  g.font = `${10 * dpr}px Courier New`; g.fillStyle = "#000"; g.strokeStyle = "#000"; g.lineWidth = 1;
  let lastY = -1e9;
  for (const p of L.panels) {
    const y0 = L.top + p.y0, y1 = L.top + p.y1;
    if (EV.rows === "sections") {
      g.strokeRect(x0 + 0.5, Math.round(y0) + 0.5, pw - 1, Math.round(y1 - y0) - 1);
      const short = p.sec.name.replace("(forth)", "fwd").replace("(back)", "back");
      if (y1 - y0 >= 11 * dpr) g.fillText(short.slice(0, 15), 2, y0 + 9 * dpr);
      const ticks = (y1 - y0 >= 30 * dpr) ? [0, p.sec.len] : [];
      // intermediate ticks every 100/200/500 m when there is room
      const step = [100, 200, 500, 1000].find((s) => s / p.mpp >= 22 * dpr) || 0;
      if (step && y1 - y0 >= 30 * dpr) for (let d = step; d < p.sec.len - step / 2; d += step) ticks.push(d);
      for (const d of ticks) {
        const y = y0 + d / p.mpp; g.beginPath(); g.moveTo(x0 - 4 * dpr, y); g.lineTo(x0, y); g.stroke();
        const t = `${Math.round(d)}`, w = g.measureText(t).width;
        if (y - y0 > 14 * dpr || d === 0 && y1 - y0 < 11 * dpr) g.fillText(t, x0 - 5 * dpr - w, Math.min(y + 4 * dpr, y1));
      }
    } else {
      g.beginPath(); g.moveTo(x0 - 6 * dpr, y0); g.lineTo(x0, y0); g.stroke();
      if (y0 - lastY >= 10 * dpr) { g.fillText(p.sec.name.slice(0, 18), 2, y0 + 9 * dpr); lastY = y0; }
    }
  }
  if (EV.rows === "fiber") g.strokeRect(x0 + 0.5, L.top + 0.5, pw - 1, L.ph - 1);

  // time axis (UTC+8)
  const step = { 1: 0.2, 3: 0.5, 10: 2, 30: 5 }[EV.span] || EV.span / 5, yb = L.bot;
  g.textBaseline = "top"; let lastX = -1e9;
  for (let t = Math.ceil(ta / step) * step; t <= ta + 2 * EV.span + 1e-6; t += step) {
    const tt = Math.round(t * 10) / 10, x = evX(tt, W);
    g.beginPath(); g.moveTo(x, yb); g.lineTo(x, yb + 4 * dpr); g.stroke();
    const lab = fmt(tt, TZ_H, false) + (step < 1 ? "." + (Math.round(tt * 10) % 10) : ""), w = g.measureText(lab).width;
    if (x - w / 2 > Math.max(lastX + 6 * dpr, 72 * dpr) && x + w / 2 < W) { g.fillText(lab, x - w / 2, yb + 5 * dpr); lastX = x + w / 2; }
  }
  g.fillText("time UTC+8", 2, yb + 5 * dpr);
  if (EV.rows === "sections") g.fillText("y: dist. (m)", 2, yb + 16 * dpr);
  g.textBaseline = "alphabetic";

  // cursor line
  const xc = evX(S.t, W); g.fillStyle = "#000"; g.fillRect(Math.round(xc), L.top, Math.max(1, Math.round(dpr)), L.ph);
  // selected station: ticks on both sides of the image
  if (S.sel !== null && S.sel !== undefined) {
    const s = S.meta.stations[S.sel], y = evY(s.section, s.dist);
    if (y !== null) {
      g.fillRect(x0, Math.round(y), x1 - x0, Math.max(1, Math.round(dpr))); // crosshair: horizontal line at the selected station
      g.beginPath(); g.moveTo(x0 - 1, y); g.lineTo(x0 - 1 - 7 * dpr, y - 4 * dpr); g.lineTo(x0 - 1 - 7 * dpr, y + 4 * dpr); g.closePath(); g.fill();
      g.beginPath(); g.moveTo(x1 + 1, y); g.lineTo(x1 + 1 + 7 * dpr, y - 4 * dpr); g.lineTo(x1 + 1 + 7 * dpr, y + 4 * dpr); g.closePath(); g.fill();
    }
  }
}

function evUI(e) {
  $("ev-state").textContent = e ? `event view E${e.i + 1}` : "day view";
  if (e) {
    const rows = EV.rows === "sections"
      ? "Rows: one panel per fiber section; y = distance along the section (m) from its first channel (back legs flipped onto their forth leg; holes: depth). Every panel uses the same m/px, so equal slope = equal apparent velocity."
      : "Rows: stations in fiber order (as in the day view).";
    $("wf-title").textContent = `Event E${e.i + 1} zoom: filtered amplitude (${S.unit}), ${S.meta.filter.frequency.join("–")} Hz, ${e.fs} Hz samples, ±${EV.span} s around the cursor (vertical line), color = colorbar above${EV.norm === "trace" ? ", but each row scaled to its own max |amplitude| in this window" : ""}. ${rows} Click or drag: move the crosshair (time + station); the window stays put until you play or leave it. ←/→ = ∓0.1 s.`;
  } else $("wf-title").textContent = `2D waterfall: stations in fiber order (rows) × time (UTC+8). Per-minute max of the 1 s RMS, same color scale as above. Click or drag to move the crosshair (time + station). Inside an event window (gray on the timeline) this becomes the event zoom view.`;
}

{
  const cv = $("waterfall");
  const pos = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * devicePixelRatio, (e.clientY - r.top) * devicePixelRatio, r]; };
  const dayAt = (e) => {
    const r = cv.getBoundingClientRect(); setT(tOf(e.clientX - r.left, cv.width));
    const s = Math.floor((e.clientY - r.top) / r.height * S.nst); if (s >= 0 && s < S.nst) S.sel = s;
  };
  cv.addEventListener("pointerdown", (e) => {
    if (!S.meta) return;
    cv.setPointerCapture(e.pointerId);
    if (!modeIsEvent()) { EV.dragDay = true; return dayAt(e); }
    const [px, py] = pos(e), loc = evLoc(py);
    if (!loc || px < MARGIN_L * devicePixelRatio) return;
    EV.anchor = evC(); EV.dragEv = true;
    S.sel = loc.st; setT(evT(px, cv.width));
  });
  cv.addEventListener("pointermove", (e) => {
    if (!S.meta) return;
    const ev = modeIsEvent();
    if (!ev) { if (EV.dragDay && cv.hasPointerCapture(e.pointerId)) dayAt(e); $("wf-hover").textContent = " "; return; }
    const [px, py] = pos(e), loc = evLoc(py), t = evT(px, cv.width);
    if (EV.dragEv && cv.hasPointerCapture(e.pointerId)) { // drag the crosshair: time within the frozen window + station
      const a = evC(); setT(Math.max(a - EV.span, Math.min(a + EV.span, t)));
      if (loc) S.sel = loc.st;
    }
    if (!loc || px < MARGIN_L * devicePixelRatio) { $("wf-hover").textContent = " "; return; }
    const s = S.meta.stations[loc.st], k = Math.floor((t - ev.t0) * ev.fs);
    const amp = s.has_data && k >= 0 && k < ev.n ? `${(ev.data[k * S.nst + loc.st] * ev.scale[loc.st] / ev.full * S.k).toFixed(1)} ${S.unit}` : "no data";
    const ms = Math.round(((t % 1) + 1) % 1 * 1000);
    $("wf-hover").textContent = `${fmt(t, TZ_H, false)}.${String(ms).padStart(3, "0")} UTC+8 · #${loc.st} ${s.section} ch ${s.ch0}–${s.ch1} · ${loc.dist.toFixed(0)} m${secOf(s.section).hole ? " depth" : " along section"} · ${amp}`;
  });
  cv.addEventListener("pointerup", () => {
    EV.dragDay = false; EV.dragEv = false;
  });
  cv.addEventListener("pointerleave", () => $("wf-hover").textContent = " ");
}
$("ev-span").onchange = (e) => { EV.span = +e.target.value; S.dirty = true; };
$("ev-rows").onchange = (e) => { EV.rows = e.target.value; EV.Lkey = ""; S.dirty = true; };
$("ev-norm").onchange = (e) => { EV.norm = e.target.value; S.dirty = true; };
$("wf-h").onchange = (e) => { document.documentElement.style.setProperty("--wf-h", `${e.target.value}px`); EV.Lkey = ""; resize(); };

// ---------- controls ----------
$("play").onclick = () => { S.playing = !S.playing; $("play").textContent = S.playing ? "pause" : "play"; $("play").classList.toggle("on", S.playing); };
$("speed").onchange = (e) => S.speed = +e.target.value;
const setSpeed = (v) => { S.speed = v; $("speed").value = String(v); };
// jump to 1 s before the first strong arrival of the event under (or next after) the cursor and play at 0.05×
$("replay").onclick = () => {
  const e = activeEvent(S.t) || S.events.find((e) => e.t0 > S.t) || S.events[0];
  if (!e || !e.data || !ensureEnvelope(e)) { $("status").textContent = "event data not loaded yet"; return; }
  setT((e.arrival ?? e.origin) - 1); setSpeed(0.05);
  if (!S.playing) $("play").click();
};
document.querySelectorAll("input[name=evmode]").forEach((r) => r.onchange = () => { S.evmode = r.value; drawColorbar(); S.dirty = true; });
document.querySelectorAll("input[name=envscale]").forEach((r) => r.onchange = () => { S.envscale = r.value; drawColorbar(); S.dirty = true; });
$("smooth").onchange =(e) => { S.smooth = +e.target.value; S.dirty = true; };
$("trail").onchange = (e) => { S.trail = +e.target.value; drawColorbar(); S.dirty = true; };
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
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || !S.meta) return;
  if (e.code === "Space") { e.preventDefault(); $("play").click(); }
  else if (e.code === "ArrowLeft" || e.code === "ArrowRight") { e.preventDefault(); setT(S.t + (e.code === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 1 : 0.1)); }
});

// ---------- main loop ----------
let last = performance.now(), lastEvent = undefined;
function frame(now) {
  const dt = (now - last) / 1000; last = now;
  if (S.meta && points) {
    if (S.playing) { setT(S.t + dt * S.speed); if (S.t >= S.meta.seconds - 0.01) $("play").click(); }
    const e = activeEvent(S.t);
    if (e) ensureEvent(e);
    const ev = modeIsEvent();
    if (ev !== lastEvent) { lastEvent = ev; if (ev && S.evmode === "envelope") ensureEnvelope(ev); drawColorbar(); }
    if (ev !== arrowFor) updateArrow(ev);
    if (S.dirty) {
      const top = !!(ev && S.evmode === "envelope" && !arrivalsOn(ev));
      for (const m of lineMats) m.depthTest = !top;
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
