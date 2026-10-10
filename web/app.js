// UI for the dfmesh browser prototype: code editor, distance field plot,
// drawing tools and mesh display. Python runs in worker.js.
"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const codeEl = $("code");
const logEl = $("log");

const state = {
  field: null, // {view, nx, ny, values, min, max, image}
  mesh: null, // {points, triangles, stats}
  meshStale: false,
  // world -> screen: sx = (x - cx) * scale + w/2, sy = h/2 - (y - cy) * scale
  cam: { cx: 0, cy: 0, scale: 100 },
  tool: "pan",
  op: "+",
  draft: null, // shape being drawn
  pointer: null, // last pointer position in world coordinates
  history: [],
  ready: false,
  running: null, // id of a running mesh job
};

// ---------------------------------------------------------------- worker

let worker = null;
let nextId = 1;
const latest = { field: 0, mesh: 0 };

function workerUrl() {
  // Allow overriding the Pyodide location and dfmesh source path, e.g.
  // index.html?pyodide=/node_modules/pyodide/ for a local copy.
  const p = new URLSearchParams(location.search);
  const q = new URLSearchParams();
  for (const k of ["pyodide", "src"]) if (p.get(k)) q.set(k, p.get(k));
  const s = q.toString();
  return "worker.js" + (s ? "?" + s : "");
}

function startWorker() {
  state.ready = false;
  setStatus("Starting Python…", "loading");
  $("mesh").disabled = true;
  worker = new Worker(workerUrl());
  worker.onmessage = (e) => onWorkerMessage(e.data);
  worker.onerror = (e) => {
    setStatus("Python worker failed", "error");
    log(`Worker error: ${e.message}`, true);
  };
}

function onWorkerMessage(msg) {
  if (msg.type === "status") {
    setStatus(msg.text, "loading");
  } else if (msg.type === "ready") {
    state.ready = true;
    const v = msg.versions;
    setStatus(`Ready · Pyodide ${v.pyodide} · numpy ${v.numpy} · scipy ${v.scipy}`, "ready");
    log(`Python ${v.python}, numpy ${v.numpy}, scipy ${v.scipy}, dfmesh ${v.dfmesh} (Pyodide ${v.pyodide})`);
    $("mesh").disabled = false;
    requestField();
    if (!state.mesh) requestMesh();
  } else if (msg.type === "fatal") {
    setStatus("Could not start Python", "error");
    log(`Start-up failed: ${msg.error}`, true);
    showHint(`Could not start Python:\n${msg.error}`, true);
  } else if (msg.type === "result") {
    if (msg.id !== latest[msg.cmd]) return; // stale
    const res = JSON.parse(msg.payload);
    if (msg.cmd === "field") onField(res);
    else onMesh(res);
  }
}

function send(cmd, extra = {}) {
  const id = nextId++;
  latest[cmd] = id;
  worker.postMessage({ id, cmd, code: codeEl.value, ...extra });
  return id;
}

function requestField() {
  if (!state.ready) return;
  send("field", { resolution: 220, kind: $("field-kind").value });
}

function requestMesh() {
  if (!state.ready || state.running) return;
  state.running = send("mesh");
  setStatus("Meshing…", "busy");
  $("mesh").disabled = true;
  $("stop").hidden = false;
  log("Generating mesh…");
}

function stopMesh() {
  // Pyodide cannot be interrupted from here, so restart the worker.
  worker.terminate();
  state.running = null;
  $("stop").hidden = true;
  log("Stopped. Restarting Python…", true);
  startWorker();
}

// ---------------------------------------------------------------- results

function onField(res) {
  if (!res.ok) {
    if (/must define a variable named `geo`/.test(res.error)) {
      state.field = null;
      showHint("Draw a shape with the tools above, or define <code>geo</code> in the code panel.");
      $("legend").hidden = true;
      draw();
    } else {
      showHint(res.traceback || res.error, true);
    }
    return;
  }
  hideHint();
  if (res.log) log(res.log.trimEnd());
  const first = !state.field;
  const f = res;
  let min = Infinity, max = -Infinity;
  for (const v of f.values) {
    if (v === null) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  f.min = min;
  f.max = max;
  f.kind = res.kind;
  f.image = fieldImage(f);
  f.contours = contours(f);
  state.field = f;
  $("legend").hidden = false;
  $("legend-min").textContent = fmt(min);
  $("legend-max").textContent = fmt(max);
  $("legend").classList.toggle("sequential", f.kind === "h");
  $("legend-label").textContent =
    f.kind === "h" ? "target edge size h" : "signed distance (negative inside)";
  if (first) fitView();
  draw();
}

function onMesh(res) {
  state.running = null;
  $("stop").hidden = true;
  $("mesh").disabled = !state.ready;
  if (res.log) log(res.log.trimEnd());
  if (!res.ok) {
    setStatus("Meshing failed", "error");
    log(res.traceback || res.error, true);
    return;
  }
  setStatus(`Meshed in ${res.elapsed.toFixed(2)} s`, "ready");
  state.mesh = res;
  state.meshStale = false;
  $("download").disabled = false;
  const s = res.stats;
  $("stats").innerHTML =
    `<dt>Nodes</dt><dd>${s.nodes}</dd><dt>Triangles</dt><dd>${s.triangles}</dd>` +
    `<dt>Min quality</dt><dd>${s.q_min.toFixed(3)}</dd><dt>Mean quality</dt><dd>${s.q_mean.toFixed(3)}</dd>`;
  log(`Mesh: ${s.nodes} nodes, ${s.triangles} triangles, ${res.elapsed.toFixed(2)} s`);
  if (!state.field) fitView(res.view);
  draw();
}

// ---------------------------------------------------------------- field rendering

const STOPS = [
  [-1, [59, 76, 192]],
  [-0.5, [141, 176, 254]],
  [0, [242, 242, 242]],
  [0.5, [244, 154, 123]],
  [1, [180, 4, 38]],
];

function colormap(t) {
  t = Math.max(-1, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i];
    if (t <= t1) {
      const [t0, c0] = STOPS[i - 1];
      const a = (t - t0) / (t1 - t0);
      return c0.map((c, k) => c + a * (c1[k] - c));
    }
  }
  return STOPS[STOPS.length - 1][1];
}

// sequential map for the edge size field
const SEQ = [
  [0, [255, 247, 188]],
  [0.5, [65, 182, 196]],
  [1, [37, 52, 148]],
];

function interp(stops, t) {
  t = Math.max(stops[0][0], Math.min(stops[stops.length - 1][0], t));
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i];
    if (t <= t1) {
      const [t0, c0] = stops[i - 1];
      const a = (t - t0) / (t1 - t0);
      return c0.map((c, k) => c + a * (c1[k] - c));
    }
  }
  return stops[stops.length - 1][1];
}

function fieldImage(f) {
  const off = document.createElement("canvas");
  off.width = f.nx;
  off.height = f.ny;
  const octx = off.getContext("2d");
  const img = octx.createImageData(f.nx, f.ny);
  const scale = Math.max(Math.abs(f.min), Math.abs(f.max)) || 1;
  for (let j = 0; j < f.ny; j++) {
    for (let i = 0; i < f.nx; i++) {
      const v = f.values[j * f.nx + i];
      // image row 0 is the top, i.e. y = y1
      const o = ((f.ny - 1 - j) * f.nx + i) * 4;
      if (v === null) {
        img.data[o + 3] = 0;
        continue;
      }
      const c = f.kind === "h"
        ? interp(SEQ, (v - f.min) / (f.max - f.min || 1))
        : colormap(v / scale);
      img.data[o] = c[0];
      img.data[o + 1] = c[1];
      img.data[o + 2] = c[2];
      img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  return off;
}

function niceStep(x) {
  const e = Math.pow(10, Math.floor(Math.log10(x)));
  const m = x / e;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * e;
}

// Marching squares; returns [{level, segs: Float64Array of x0,y0,x1,y1,...}]
function contours(f) {
  const range = f.kind === "h" ? f.max - f.min : Math.max(Math.abs(f.min), Math.abs(f.max));
  if (!isFinite(range) || range === 0) return [];
  const step = niceStep(range / 6);
  const levels = [];
  for (let l = Math.ceil(f.min / step) * step; l <= f.max; l += step) {
    levels.push(Math.abs(l) < step * 1e-9 ? 0 : l);
  }
  if (f.kind !== "h" && !levels.includes(0) && f.min < 0 && f.max > 0) levels.push(0);
  const [x0, x1, y0, y1] = f.view;
  const dx = (x1 - x0) / (f.nx - 1);
  const dy = (y1 - y0) / (f.ny - 1);
  const V = f.values;
  const out = [];
  for (const level of levels) {
    const segs = [];
    for (let j = 0; j < f.ny - 1; j++) {
      for (let i = 0; i < f.nx - 1; i++) {
        const a = V[j * f.nx + i], b = V[j * f.nx + i + 1];
        const c = V[(j + 1) * f.nx + i + 1], d = V[(j + 1) * f.nx + i];
        if (a === null || b === null || c === null || d === null) continue;
        const pts = [];
        // edges: a-b (bottom), b-c (right), d-c (top), a-d (left)
        const edge = (va, vb, xa, ya, xb, yb) => {
          if ((va < level) !== (vb < level)) {
            const t = (level - va) / (vb - va);
            pts.push(xa + t * (xb - xa), ya + t * (yb - ya));
          }
        };
        const X = x0 + i * dx, Y = y0 + j * dy;
        edge(a, b, X, Y, X + dx, Y);
        edge(b, c, X + dx, Y, X + dx, Y + dy);
        edge(d, c, X, Y + dy, X + dx, Y + dy);
        edge(a, d, X, Y, X, Y + dy);
        if (pts.length === 4) segs.push(...pts);
        else if (pts.length === 8) segs.push(...pts); // saddle: two segments
      }
    }
    out.push({ level, segs });
  }
  return out;
}

// ---------------------------------------------------------------- view

function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));
  draw();
}

function size() {
  const dpr = window.devicePixelRatio || 1;
  return { w: canvas.width / dpr, h: canvas.height / dpr, dpr };
}

function toScreen(x, y) {
  const { w, h } = size();
  const c = state.cam;
  return [(x - c.cx) * c.scale + w / 2, h / 2 - (y - c.cy) * c.scale];
}

function toWorld(sx, sy) {
  const { w, h } = size();
  const c = state.cam;
  return [(sx - w / 2) / c.scale + c.cx, (h / 2 - sy) / c.scale + c.cy];
}

function fitView(view) {
  view = view || (state.field && state.field.view) || (state.mesh && state.mesh.view) || [-1, 1, -1, 1];
  const [x0, x1, y0, y1] = view;
  const { w, h } = size();
  // leave room for the toolbar at the top
  const top = document.querySelector(".toolbar").getBoundingClientRect().height + 16, pad = 16;
  state.cam.scale = Math.min((w - 2 * pad) / (x1 - x0), (h - top - 2 * pad) / (y1 - y0));
  state.cam.cx = (x0 + x1) / 2;
  state.cam.cy = (y0 + y1) / 2 + (top / 2) / state.cam.scale;
  draw();
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function draw() {
  const { w, h, dpr } = size();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  drawGrid(w, h);
  const f = state.field;
  if (f && $("show-field").checked) {
    const [x0, x1, y0, y1] = f.view;
    const [sx0, sy1] = toScreen(x0, y1);
    const [sx1, sy0] = toScreen(x1, y0);
    // pixel centres sit on the grid nodes, so extend by half a cell
    const hx = (sx1 - sx0) / (f.nx - 1) / 2, hy = (sy0 - sy1) / (f.ny - 1) / 2;
    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = 0.9;
    ctx.drawImage(f.image, sx0 - hx, sy1 - hy, sx1 - sx0 + 2 * hx, sy0 - sy1 + 2 * hy);
    ctx.globalAlpha = 1;
  }
  if (f && $("show-contours").checked) {
    for (const { level, segs } of f.contours) {
      ctx.beginPath();
      for (let k = 0; k < segs.length; k += 4) {
        const [a, b] = toScreen(segs[k], segs[k + 1]);
        const [c, d] = toScreen(segs[k + 2], segs[k + 3]);
        ctx.moveTo(a, b);
        ctx.lineTo(c, d);
      }
      if (level === 0 && f.kind !== "h") {
        ctx.strokeStyle = "#111";
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = "rgba(40,40,40,0.35)";
        ctx.lineWidth = 0.8;
      }
      ctx.stroke();
    }
  }
  const m = state.mesh;
  if (m && $("show-mesh").checked) {
    const P = m.points.map(([x, y]) => toScreen(x, y));
    ctx.beginPath();
    for (const [a, b, c] of m.triangles) {
      ctx.moveTo(P[a][0], P[a][1]);
      ctx.lineTo(P[b][0], P[b][1]);
      ctx.lineTo(P[c][0], P[c][1]);
      ctx.closePath();
    }
    ctx.strokeStyle = state.meshStale ? "rgba(20,20,20,0.3)" : "rgba(10,20,40,0.85)";
    ctx.lineWidth = 0.7;
    ctx.setLineDash(state.meshStale ? [3, 3] : []);
    ctx.stroke();
    ctx.setLineDash([]);
    if (state.meshStale) {
      ctx.fillStyle = cssVar("--muted");
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("mesh is out of date: Generate mesh to update", 12, h - 40);
    }
  }
  drawDraft();
}

function drawGrid(w, h) {
  const [x0, y1] = toWorld(0, 0);
  const [x1, y0] = toWorld(w, h);
  const step = niceStep(80 / state.cam.scale); // roughly 80 px between lines
  ctx.lineWidth = 1;
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = cssVar("--muted");
  for (let k = Math.ceil(x0 / step); k * step <= x1; k++) {
    const x = k * step;
    const [sx] = toScreen(x, 0);
    ctx.strokeStyle = Math.abs(x) < step / 2 ? "rgba(128,128,128,0.55)" : "rgba(128,128,128,0.18)";
    ctx.beginPath();
    ctx.moveTo(Math.round(sx) + 0.5, 0);
    ctx.lineTo(Math.round(sx) + 0.5, h);
    ctx.stroke();
    ctx.fillText(fmt(x, step), sx + 3, h - 4);
  }
  for (let k = Math.ceil(y0 / step); k * step <= y1; k++) {
    const y = k * step;
    const [, sy] = toScreen(0, y);
    ctx.strokeStyle = Math.abs(y) < step / 2 ? "rgba(128,128,128,0.55)" : "rgba(128,128,128,0.18)";
    ctx.beginPath();
    ctx.moveTo(0, Math.round(sy) + 0.5);
    ctx.lineTo(w, Math.round(sy) + 0.5);
    ctx.stroke();
    ctx.fillText(fmt(y, step), w - 40, sy - 3);
  }
}

function drawDraft() {
  const d = state.draft;
  if (!d) return;
  ctx.strokeStyle = state.op === "-" ? "#b42318" : state.op === "&" ? "#7a3cc2" : "#1a7f37";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  if (d.type === "circle") {
    const [sx, sy] = toScreen(d.x0, d.y0);
    ctx.arc(sx, sy, d.r * state.cam.scale, 0, 2 * Math.PI);
  } else if (d.type === "rect") {
    const [a, b] = toScreen(d.x0, d.y0);
    const [c, e] = toScreen(d.x1, d.y1);
    ctx.rect(Math.min(a, c), Math.min(b, e), Math.abs(c - a), Math.abs(e - b));
  } else if (d.type === "polygon") {
    const pts = d.points.concat(state.pointer ? [state.pointer] : []);
    pts.forEach(([x, y], i) => {
      const [sx, sy] = toScreen(x, y);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    });
  }
  ctx.stroke();
  ctx.setLineDash([]);
  if (d.type === "polygon") {
    ctx.fillStyle = "#fff";
    for (const [x, y] of d.points) {
      const [sx, sy] = toScreen(x, y);
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }
}

// ---------------------------------------------------------------- interaction

function snapStep() {
  // round drawn coordinates to roughly a pixel-sized decimal step
  return Math.pow(10, Math.floor(Math.log10(4 / state.cam.scale)));
}
function snap(v) {
  const s = snapStep();
  return Math.round(v / s) * s;
}
function num(v) {
  const s = snapStep();
  const digits = Math.max(1, -Math.floor(Math.log10(s)));
  return Number(v).toFixed(digits);
}
function fmt(v, step) {
  if (v === 0 || (step && Math.abs(v) < step * 1e-6)) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(1);
  return String(Number(v.toPrecision(3)));
}

function eventWorld(e) {
  const r = canvas.getBoundingClientRect();
  return toWorld(e.clientX - r.left, e.clientY - r.top);
}

let drag = null;

canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  const [x, y] = eventWorld(e);
  if (state.tool === "pan" || e.button === 1 || e.button === 2) {
    drag = { type: "pan", sx: e.clientX, sy: e.clientY, cx: state.cam.cx, cy: state.cam.cy };
    canvas.classList.add("dragging");
  } else if (state.tool === "circle") {
    state.draft = { type: "circle", x0: snap(x), y0: snap(y), r: 0 };
    drag = { type: "draw" };
  } else if (state.tool === "rect") {
    state.draft = { type: "rect", x0: snap(x), y0: snap(y), x1: snap(x), y1: snap(y) };
    drag = { type: "draw" };
  } else if (state.tool === "polygon") {
    const d = state.draft && state.draft.type === "polygon" ? state.draft : (state.draft = { type: "polygon", points: [] });
    const p = [snap(x), snap(y)];
    if (d.points.length >= 3) {
      const [fx, fy] = toScreen(...d.points[0]);
      const [px, py] = toScreen(...p);
      if (Math.hypot(fx - px, fy - py) < 8) {
        finishPolygon();
        return;
      }
    }
    d.points.push(p);
    draw();
  }
});

canvas.addEventListener("pointermove", (e) => {
  const [x, y] = eventWorld(e);
  state.pointer = [snap(x), snap(y)];
  updateReadout(x, y);
  if (!drag) {
    if (state.draft && state.draft.type === "polygon") draw();
    return;
  }
  if (drag.type === "pan") {
    state.cam.cx = drag.cx - (e.clientX - drag.sx) / state.cam.scale;
    state.cam.cy = drag.cy + (e.clientY - drag.sy) / state.cam.scale;
  } else if (state.draft.type === "circle") {
    state.draft.r = snap(Math.hypot(x - state.draft.x0, y - state.draft.y0));
  } else if (state.draft.type === "rect") {
    state.draft.x1 = snap(x);
    state.draft.y1 = snap(y);
  }
  draw();
});

canvas.addEventListener("pointerup", () => {
  if (!drag) return;
  canvas.classList.remove("dragging");
  if (drag.type === "draw") {
    const d = state.draft;
    state.draft = null;
    if (d.type === "circle" && d.r > 0) {
      addShape(`df.Circle([${num(d.x0)}, ${num(d.y0)}], ${num(d.r)})`);
    } else if (d.type === "rect" && d.x0 !== d.x1 && d.y0 !== d.y1) {
      const [a, b] = [Math.min(d.x0, d.x1), Math.max(d.x0, d.x1)];
      const [c, e] = [Math.min(d.y0, d.y1), Math.max(d.y0, d.y1)];
      addShape(`df.Rectangle(${num(a)}, ${num(b)}, ${num(c)}, ${num(e)})`);
    }
    draw();
  }
  drag = null;
});

canvas.addEventListener("dblclick", () => {
  if (state.draft && state.draft.type === "polygon") finishPolygon();
});

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener("pointerleave", () => {
  $("readout").textContent = "";
});

canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    const [x, y] = toWorld(sx, sy);
    const k = Math.exp(-e.deltaY * 0.0015);
    state.cam.scale *= k;
    // keep the point under the cursor fixed
    const [x2, y2] = toWorld(sx, sy);
    state.cam.cx += x - x2;
    state.cam.cy += y - y2;
    draw();
  },
  { passive: false },
);

function finishPolygon() {
  const d = state.draft;
  state.draft = null;
  // drop a duplicated last point from the double click
  const pts = d.points.filter((p, i) => i === 0 || p[0] !== d.points[i - 1][0] || p[1] !== d.points[i - 1][1]);
  if (pts.length >= 3) {
    const list = pts.map(([x, y]) => `[${num(x)}, ${num(y)}]`).join(", ");
    addShape(`df.Polygon([${list}])`);
  }
  draw();
}

function updateReadout(x, y) {
  let text = `x ${num(x)}  y ${num(y)}`;
  const f = state.field;
  if (f) {
    const [x0, x1, y0, y1] = f.view;
    const i = Math.round(((x - x0) / (x1 - x0)) * (f.nx - 1));
    const j = Math.round(((y - y0) / (y1 - y0)) * (f.ny - 1));
    if (i >= 0 && i < f.nx && j >= 0 && j < f.ny) {
      const v = f.values[j * f.nx + i];
      if (v !== null) text += `  ${f.kind === "h" ? "h" : "d"} ${v.toFixed(3)}`;
    }
  }
  $("readout").textContent = text;
}

// Append a drawn shape to the code as a set operation on `geo`.
function addShape(expr) {
  const code = codeEl.value;
  const hasGeo = /^\s*geo\s*=/m.test(code);
  const line = hasGeo ? `geo = geo ${state.op} ${expr}` : `geo = ${expr}`;
  setCode(code.replace(/\s*$/, "\n") + `\n# drawn\n${line}\n`);
  requestField();
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.draft) {
    state.draft = null;
    draw();
  }
  if (e.key === "Enter" && state.draft && state.draft.type === "polygon") finishPolygon();
});

function selectTool(tool) {
  state.tool = tool;
  state.draft = null;
  document.querySelectorAll("[data-tool]").forEach((b) => {
    b.classList.toggle("active", b.dataset.tool === tool);
    b.setAttribute("aria-checked", b.dataset.tool === tool);
  });
  canvas.className = tool === "pan" ? "tool-pan" : "tool-draw";
  draw();
}
document.querySelectorAll("[data-tool]").forEach((b) => b.addEventListener("click", () => selectTool(b.dataset.tool)));
document.querySelectorAll("[data-op]").forEach((b) =>
  b.addEventListener("click", () => {
    state.op = b.dataset.op;
    document.querySelectorAll("[data-op]").forEach((c) => {
      c.classList.toggle("active", c === b);
      c.setAttribute("aria-checked", c === b);
    });
  }),
);
for (const id of ["show-field", "show-contours", "show-mesh"]) $(id).addEventListener("change", draw);
$("field-kind").addEventListener("change", requestField);
$("fit").addEventListener("click", () => fitView());

// ---------------------------------------------------------------- code editor

function setCode(code, { pushHistory = true } = {}) {
  if (pushHistory) state.history.push(codeEl.value);
  codeEl.value = code;
  codeChanged();
}

let fieldTimer = null;
function codeChanged() {
  if (state.mesh) state.meshStale = true;
  clearTimeout(fieldTimer);
  fieldTimer = setTimeout(requestField, 400);
  draw();
}

codeEl.addEventListener("input", codeChanged);
codeEl.addEventListener("keydown", (e) => {
  if (e.key === "Tab" && !e.shiftKey) {
    e.preventDefault();
    document.execCommand("insertText", false, "    ");
  } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    requestMesh();
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && e.target !== codeEl) requestMesh();
});

$("undo").addEventListener("click", () => {
  if (state.history.length) setCode(state.history.pop(), { pushHistory: false });
});

function loadExample(i) {
  const ex = EXAMPLES[i];
  state.mesh = null;
  state.meshStale = false;
  state.field = null; // refit the view to the new geometry
  $("stats").innerHTML = "";
  $("download").disabled = true;
  setCode(ex.code);
  requestField();
  if (state.ready) requestMesh();
}

EXAMPLES.forEach((ex, i) => {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = ex.name;
  $("example").appendChild(o);
});
$("example").addEventListener("change", (e) => loadExample(+e.target.value));

$("mesh").addEventListener("click", requestMesh);
$("stop").addEventListener("click", stopMesh);

$("download").addEventListener("click", () => {
  const m = state.mesh;
  if (!m) return;
  const blob = new Blob([JSON.stringify({ p: m.points, t: m.triangles }, null, 0)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "mesh.json";
  a.click();
  URL.revokeObjectURL(a.href);
});

function standaloneScript() {
  return `import math

import numpy as np
import dfmesh as df

${codeEl.value.trimEnd()}

try:
    h
except NameError:
    h = 0.1
try:
    options
except NameError:
    options = {}

p, t = df.triangulate(geo, h, **options)  # p: (2, N) points, t: (3, M) triangles
print(p.shape, t.shape)
`;
}

$("copy-script").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(standaloneScript());
    log("Copied a standalone Python script to the clipboard.");
  } catch {
    log("Clipboard is not available; select the code panel and copy it instead.", true);
  }
});

// ---------------------------------------------------------------- splitter

(() => {
  const split = $("splitter");
  const side = document.querySelector(".side");
  let start = null;
  split.addEventListener("pointerdown", (e) => {
    start = { x: e.clientX, w: side.getBoundingClientRect().width };
    split.setPointerCapture(e.pointerId);
    split.classList.add("dragging");
  });
  split.addEventListener("pointermove", (e) => {
    if (!start) return;
    side.style.width = `${start.w - (e.clientX - start.x)}px`;
    resize();
  });
  split.addEventListener("pointerup", () => {
    start = null;
    split.classList.remove("dragging");
  });
})();

// ---------------------------------------------------------------- misc

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = `status ${kind}`;
}

function log(text, isError = false) {
  if (!text) return;
  const span = document.createElement("span");
  if (isError) span.className = "err";
  span.textContent = text + "\n";
  logEl.appendChild(span);
  logEl.scrollTop = logEl.scrollHeight;
}

function showHint(html, isError = false) {
  const el = $("hint");
  el.hidden = false;
  el.classList.toggle("error", isError);
  if (isError) el.textContent = html;
  else el.innerHTML = html;
}
function hideHint() {
  $("hint").hidden = true;
}

new ResizeObserver(resize).observe(canvas);
selectTool("pan");
startWorker();
loadExample(0);
