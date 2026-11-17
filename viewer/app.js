// Planner UI. All coordinates are scanner RAS of the T1 (mm) unless named
// otherwise. The fMRI map is never resampled ahead of time: each pixel's world
// point is carried into the map's voxel grid through the registration and the
// map's own matrix (scene.overlay.world_to_voxel).
import * as G from "./geometry.js";

const RES = 288; // canvas pixels per side
const state = {
  scene: null, t1: null, ov: null,
  cursor: [0, 0, 0], center: [0, 0, 0], mmPerPx: 0.9,
  target: null, entry: null, depth: 1.0,
  thr: 3.1, ovOn: true, radiological: true, hover: null,
};
const views = {};
const $ = (id) => document.getElementById(id);
const fmt = (p, d = 1) => (p ? p.map((x) => x.toFixed(d).padStart(7)).join(" ") : "–");

async function loadArray(url, dtype) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  let buf = await r.arrayBuffer();
  if (url.endsWith(".gz")) {
    const ds = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    buf = await new Response(ds).arrayBuffer();
  }
  return dtype === "uint8" ? new Uint8Array(buf) : new Float32Array(buf);
}

async function init() {
  const base = new URLSearchParams(location.search).get("scene") || "scene/";
  const scene = await (await fetch(base + "scene.json")).json();
  state.scene = scene;
  state.t1 = { ...scene.t1, data: await loadArray(base + scene.t1.file, scene.t1.dtype), inv: G.inv4(scene.t1.affine) };
  if (scene.overlay) {
    state.ov = { ...scene.overlay, data: await loadArray(base + scene.overlay.file, "float32") };
    $("rOvLabel").textContent = scene.overlay.short_label || "t";
  }
  state.center = scene.head_center_ras.slice();
  state.cursor = (scene.peak_ras || scene.head_center_ras).slice();
  $("sceneId").textContent = scene.title;
  for (const fig of document.querySelectorAll(".view")) {
    const canvas = fig.querySelector("canvas");
    canvas.width = canvas.height = RES;
    const v = { name: fig.dataset.view, fig, canvas, ctx: canvas.getContext("2d"), img: null };
    v.img = v.ctx.createImageData(RES, RES);
    views[v.name] = v;
    attachView(v);
  }
  wireControls();
  render();
}

// ---- plane for each view ---------------------------------------------------
function plane(v) {
  if (v.name === "axial" || v.name === "coronal" || v.name === "sagittal") {
    const b = G.planeBasis(v.name, state.radiological);
    const off = G.dot(G.sub(state.cursor, state.center), b.n);
    return { basis: b, center: G.add(state.center, G.scale(b.n, off)), mmPerPx: state.mmPerPx };
  }
  if (!state.target || !state.entry || G.norm(G.sub(state.target, state.entry)) < 1) return null;
  const tb = G.trajectoryBasis(state.entry, state.target);
  const len = G.norm(G.sub(state.target, state.entry));
  if (v.name === "along") {
    const mid = G.scale(G.add(state.entry, state.target), 0.5);
    return { basis: tb.along, center: mid, mmPerPx: Math.max(120, len * 1.5) / RES };
  }
  const tip = G.add(state.entry, G.scale(G.sub(state.target, state.entry), state.depth));
  return { basis: tb.probe, center: tip, mmPerPx: 60 / RES };
}

// ---- drawing ---------------------------------------------------------------
function render() {
  for (const v of Object.values(views)) drawView(v);
  updateReadouts();
}

function drawView(v) {
  const pl = plane(v);
  const { ctx, img } = v;
  if (!pl) {
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, RES, RES);
    ctx.fillStyle = "#8a94a3"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Set a target and an entry point", RES / 2, RES / 2);
    v.plane = null;
    return;
  }
  v.plane = pl;
  const { basis, center, mmPerPx } = pl;
  // world(px, py) = o + px*U + py*V, all affine, so voxel coordinates step linearly too
  const o = G.pixelToWorld(0, 0, RES, RES, center, basis, mmPerPx);
  const U = G.scale(basis.u, mmPerPx), V = G.scale(basis.v, -mmPerPx);
  const t1 = state.t1, A = t1.inv;
  const lin = (M, p) => G.apply(M, p);
  const linDir = (M, d) => [M[0][0] * d[0] + M[0][1] * d[1] + M[0][2] * d[2], M[1][0] * d[0] + M[1][1] * d[1] + M[1][2] * d[2], M[2][0] * d[0] + M[2][1] * d[1] + M[2][2] * d[2]];
  const t0 = lin(A, o), tu = linDir(A, U), tv = linDir(A, V);
  const ov = state.ovOn && state.ov ? state.ov : null;
  let q0, qu, qv;
  if (ov) { const W = ov.world_to_voxel; q0 = lin(W, o); qu = linDir(W, U); qv = linDir(W, V); }
  const d = img.data, thr = state.thr;
  for (let py = 0; py < RES; py++) {
    for (let px = 0; px < RES; px++) {
      const i = t0[0] + px * tu[0] + py * tv[0], j = t0[1] + px * tu[1] + py * tv[1], k = t0[2] + px * tu[2] + py * tv[2];
      let g = G.trilinear(t1.data, t1.shape, i, j, k);
      if (Number.isNaN(g)) g = 0;
      let r = g, gg = g, b = g;
      if (ov) {
        const t = G.trilinear(ov.data, ov.shape, q0[0] + px * qu[0] + py * qv[0], q0[1] + px * qu[1] + py * qv[1], q0[2] + px * qu[2] + py * qv[2]);
        if (t > thr) {
          const f = Math.min(1, (t - thr) / 5);
          const a = 0.75;
          r = (1 - a) * g + a * 255; gg = (1 - a) * g + a * (60 + 195 * f); b = (1 - a) * g + a * 20 * f;
        }
      }
      const p = 4 * (py * RES + px);
      d[p] = r; d[p + 1] = gg; d[p + 2] = b; d[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  drawAnnotations(v, pl);
}

const LETTERS = [["R", "L"], ["A", "P"], ["S", "I"]];
function letter(vec) {
  let ax = 0;
  for (let a = 1; a < 3; a++) if (Math.abs(vec[a]) > Math.abs(vec[ax])) ax = a;
  const oblique = Math.abs(vec[ax]) < 0.95 ? "~" : "";
  return oblique + LETTERS[ax][vec[ax] >= 0 ? 0 : 1];
}

function drawAnnotations(v, pl) {
  const { ctx } = v;
  const { basis, center, mmPerPx } = pl;
  const toPx = (p) => G.worldToPixel(p, RES, RES, center, basis, mmPerPx);
  // cursor crosshair
  const [cx, cy, cdist] = toPx(state.cursor);
  if (Math.abs(cdist) < 1.5 * mmPerPx + 0.5) {
    ctx.strokeStyle = "rgba(79,179,255,0.55)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, cy - 6); ctx.moveTo(cx, cy + 6); ctx.lineTo(cx, RES);
    ctx.moveTo(0, cy); ctx.lineTo(cx - 6, cy); ctx.moveTo(cx + 6, cy); ctx.lineTo(RES, cy); ctx.stroke();
  }
  // trajectory: projected line, solid where it is within 2 mm of this plane
  if (state.target && state.entry) {
    const a = toPx(state.entry), b = toPx(state.target);
    ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]);
    if (v.name === "probe") {
      ctx.strokeStyle = "#39d98a"; ctx.beginPath(); ctx.arc(RES / 2, RES / 2, 1 / mmPerPx, 0, 2 * Math.PI); ctx.stroke(); // 1 mm radius
    }
  }
  for (const [p, col, lab] of [[state.entry, "#4fd1ff", "E"], [state.target, "#39d98a", "T"]]) {
    if (!p) continue;
    const [x, y, dist] = toPx(p);
    const near = Math.abs(dist) <= 2;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = near ? 2 : 1;
    ctx.globalAlpha = near ? 1 : 0.35;
    ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.stroke();
    ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.fillText(lab, x + 7, y - 6);
    ctx.globalAlpha = 1; ctx.lineWidth = 1;
  }
  // orientation letters
  ctx.fillStyle = "#ffb547"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(letter(basis.v), RES / 2, 14);
  ctx.fillText(letter(G.scale(basis.v, -1)), RES / 2, RES - 6);
  ctx.textAlign = "left"; ctx.fillText(letter(G.scale(basis.u, -1)), 4, RES / 2);
  ctx.textAlign = "right"; ctx.fillText(letter(basis.u), RES - 4, RES / 2);
  // 10 mm scale bar
  const L = 10 / mmPerPx;
  ctx.strokeStyle = "#d9dee5"; ctx.beginPath(); ctx.moveTo(RES - 10 - L, RES - 10); ctx.lineTo(RES - 10, RES - 10); ctx.stroke();
  ctx.font = "10px sans-serif"; ctx.textAlign = "right"; ctx.fillStyle = "#d9dee5"; ctx.fillText("10 mm", RES - 10, RES - 14);
}

// ---- head surface ------------------------------------------------------------
// The T1 is windowed to 0-255 for display; scalp and skin sit well above this.
const HEAD_LEVEL = 25;
function t1At(p) {
  const q = G.apply(state.t1.inv, p);
  const v = G.trilinear(state.t1.data, state.t1.shape, q[0], q[1], q[2]);
  return Number.isNaN(v) ? 0 : v;
}
// Walk from the target outward along the planned line; the scalp is the last
// point before at least 5 mm of continuous air.
function scalpAlong(target, dirOut) {
  let last = null, air = 0;
  for (let s = 0; s <= 200; s += 0.5) {
    const p = G.add(target, G.scale(dirOut, s));
    if (t1At(p) >= HEAD_LEVEL) { last = p; air = 0; } else if (last && (air += 0.5) >= 5) break;
  }
  return last;
}
function planWarnings() {
  const w = [];
  if (state.entry) {
    if (t1At(state.entry) < HEAD_LEVEL) w.push("Entry is outside the head.");
    if (state.target) {
      const d = G.unit(G.sub(state.entry, state.target));
      const sc = scalpAlong(state.target, d);
      if (sc) {
        const gap = G.dot(G.sub(state.entry, sc), d);
        if (Math.abs(gap) > 3) w.push(`Entry is ${Math.abs(gap).toFixed(0)} mm ${gap > 0 ? "outside" : "inside"} the scalp along this line.`);
      }
    }
  }
  return w;
}
function toMni(p) { return state.scene.t1_to_mni ? G.apply(state.scene.t1_to_mni, p) : null; }

// ---- readouts ----------------------------------------------------------------
function sampleOverlay(p) {
  if (!state.ov) return NaN;
  const q = G.apply(state.ov.world_to_voxel, p);
  return G.trilinear(state.ov.data, state.ov.shape, q[0], q[1], q[2]);
}

function updateReadouts() {
  const p = state.cursor;
  $("rRas").textContent = fmt(p);
  $("rLps").textContent = fmt(G.rasToLps(p));
  const ijk = G.apply(state.t1.inv, p);
  $("rVox").textContent = ijk.map((x) => Math.round(x)).join(", ");
  const g = G.trilinear(state.t1.data, state.t1.shape, ijk[0], ijk[1], ijk[2]);
  $("rT1").textContent = Number.isNaN(g) ? "outside" : g.toFixed(0) + " (display units)";
  const t = sampleOverlay(p);
  $("rOv").textContent = Number.isNaN(t) ? "outside map" : t.toFixed(2);
  $("rMni").textContent = fmt(toMni(p), 0);
  $("pMni").textContent = state.target ? fmt(toMni(state.target), 0) : "–";
  const warns = planWarnings();
  $("planWarn").hidden = warns.length === 0;
  $("planWarn").textContent = warns.join(" ");
  $("snapEntry").disabled = !(state.target && state.entry);
  $("pTarget").textContent = fmt(state.target);
  $("pEntry").textContent = fmt(state.entry);
  const ok = state.target && state.entry && G.norm(G.sub(state.target, state.entry)) >= 1;
  $("pLen").textContent = ok ? G.norm(G.sub(state.target, state.entry)).toFixed(1) + " mm" : "–";
  const tt = state.target ? sampleOverlay(state.target) : NaN;
  $("pT").textContent = state.target ? (Number.isNaN(tt) ? "outside map" : tt.toFixed(2)) : "–";
  $("export").disabled = !state.target;
  $("probeDepth").textContent = ok ? `${(state.depth * G.norm(G.sub(state.target, state.entry))).toFixed(0)} mm from entry` : "";
}

// ---- interaction -------------------------------------------------------------
function eventPx(v, e) {
  const r = v.canvas.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * RES, ((e.clientY - r.top) / r.height) * RES];
}

function attachView(v) {
  let drag = null;
  v.canvas.addEventListener("pointerdown", (e) => {
    if (!v.plane) return;
    const [px, py] = eventPx(v, e);
    if (e.shiftKey && ["axial", "coronal", "sagittal"].includes(v.name)) { drag = { px, py, c: state.center.slice() }; return; }
    state.cursor = G.pixelToWorld(px, py, RES, RES, v.plane.center, v.plane.basis, v.plane.mmPerPx);
    render();
  });
  v.canvas.addEventListener("pointermove", (e) => {
    if (!drag || !v.plane) return;
    const [px, py] = eventPx(v, e);
    const b = v.plane.basis;
    state.center = G.sub(drag.c, G.add(G.scale(b.u, (px - drag.px) * state.mmPerPx), G.scale(b.v, -(py - drag.py) * state.mmPerPx)));
    render();
  });
  window.addEventListener("pointerup", () => (drag = null));
  v.canvas.addEventListener("pointerenter", () => { state.hover = v; v.fig.classList.add("active"); });
  v.canvas.addEventListener("pointerleave", () => { if (state.hover === v) state.hover = null; v.fig.classList.remove("active"); });
  v.canvas.addEventListener("wheel", (e) => {
    if (!v.plane) return;
    e.preventDefault();
    if (e.ctrlKey) {
      state.mmPerPx = Math.min(2, Math.max(0.2, state.mmPerPx * (e.deltaY > 0 ? 1.1 : 1 / 1.1)));
    } else if (v.name === "probe") {
      state.depth = Math.min(1.2, Math.max(0, state.depth + (e.deltaY > 0 ? 0.01 : -0.01)));
      $("depth").value = Math.round(state.depth * 100); $("depthOut").textContent = $("depth").value + "%";
    } else {
      state.cursor = G.add(state.cursor, G.scale(v.plane.basis.n, e.deltaY > 0 ? -1 : 1));
    }
    render();
  }, { passive: false });
}

function setPoint(which) {
  state[which] = state.cursor.slice();
  if (which === "target" && !state.entry) $("modeHint").textContent = "Target set. Now move the cursor to an entry point on the scalp and press E.";
  else if (state.target && state.entry) $("modeHint").textContent = "Plan ready. Check the trajectory views, then export.";
  render();
}

function wireControls() {
  $("thr").addEventListener("input", (e) => { state.thr = +e.target.value; $("thrOut").textContent = state.thr.toFixed(1); render(); });
  $("ovOn").addEventListener("change", (e) => { state.ovOn = e.target.checked; render(); });
  $("radio").addEventListener("change", (e) => { state.radiological = e.target.checked; render(); });
  $("depth").addEventListener("input", (e) => { state.depth = +e.target.value / 100; $("depthOut").textContent = e.target.value + "%"; render(); });
  $("toPeak").addEventListener("click", () => { if (state.scene.peak_ras) { state.cursor = state.scene.peak_ras.slice(); render(); } });
  $("setTarget").addEventListener("click", () => setPoint("target"));
  $("setEntry").addEventListener("click", () => setPoint("entry"));
  $("export").addEventListener("click", exportPlan);
  $("snapEntry").addEventListener("click", () => {
    const sc = scalpAlong(state.target, G.unit(G.sub(state.entry, state.target)));
    if (sc) { state.entry = sc; state.cursor = sc.slice(); render(); }
  });
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
    const k = e.key.toLowerCase();
    if (k === "t") setPoint("target");
    else if (k === "e") setPoint("entry");
    else if (state.hover && state.hover.plane && ["arrowleft", "arrowright", "arrowup", "arrowdown", "pageup", "pagedown"].includes(k)) {
      e.preventDefault();
      const b = state.hover.plane.basis;
      const step = { arrowleft: G.scale(b.u, -1), arrowright: b.u, arrowup: b.v, arrowdown: G.scale(b.v, -1), pageup: b.n, pagedown: G.scale(b.n, -1) }[k];
      state.cursor = G.add(state.cursor, step);
      render();
    }
  });
}

function buildPlan() {
  const s = state.scene;
  const P = (p) => p && { ras_mm: p.map((x) => +x.toFixed(3)), lps_mm: G.rasToLps(p).map((x) => +x.toFixed(3)),
    t1_voxel: G.apply(state.t1.inv, p).map((x) => +x.toFixed(3)) };
  const plan = {
    schema: "scan-to-target/plan/1",
    created: new Date().toISOString(),
    title: s.title,
    frame: "scanner RAS of the T1 (mm); lps_mm is the same point in DICOM patient coordinates",
    target: P(state.target),
    entry: P(state.entry),
    trajectory: null,
    map_t_at_target: state.target ? +sampleOverlay(state.target).toFixed(3) : null,
    target_mni_approx_mm: state.target && toMni(state.target) ? toMni(state.target).map((x) => +x.toFixed(1)) : null,
    warnings: planWarnings(),
    map_threshold_shown: state.thr,
    inputs: s.inputs,
    notes: $("notes").value,
  };
  if (state.target && state.entry) {
    const d = G.sub(state.target, state.entry);
    plan.trajectory = { direction_ras: G.unit(d).map((x) => +x.toFixed(6)), length_mm: +G.norm(d).toFixed(3) };
  }
  return plan;
}

async function exportPlan() {
  const plan = buildPlan();
  const msg = $("exportMsg");
  try {
    const r = await fetch("api/plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(plan) });
    if (!r.ok) throw new Error(await r.text());
    const res = await r.json();
    msg.textContent = `Saved ${res.saved_as}. Checked by the server: target re-read at ${res.check.target_voxel_error.toExponential(1)} voxel.`;
    if (res.robot) showRobot(res.robot);
  } catch (err) {
    const blob = new Blob([JSON.stringify(plan, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "plan.json"; a.click();
    msg.textContent = "No planning server here, so the plan was downloaded instead.";
  }
}

function showRobot(rb) {
  $("robotH").hidden = false; const el = $("robot"); el.hidden = false;
  el.innerHTML = `<table class="readout">
    <tr><th>Target, robot mm</th><td>${fmt(rb.target_robot_mm, 2)}</td></tr>
    <tr><th>Entry, robot mm</th><td>${fmt(rb.entry_robot_mm, 2)}</td></tr>
    <tr><th>Fiducials</th><td>${rb.n_fiducials}, FRE ${rb.fre_mm.toFixed(2)} mm</td></tr>
    <tr><th>Predicted TRE</th><td>${rb.predicted_tre_rms_mm.toFixed(2)} mm RMS at target</td></tr>
  </table><p class="hint">${rb.note}</p>`;
}

init().catch((e) => { $("sceneId").textContent = "failed to load scene: " + e.message; console.error(e); });
