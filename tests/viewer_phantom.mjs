// Left-right / orientation phantom for the viewer's real rendering code.
// Truth comes from the display conventions, not from the code under test:
//   radiological axial/coronal: patient LEFT on screen RIGHT; neurological: patient LEFT on screen LEFT;
//   axial: anterior at the top; coronal and sagittal: superior at the top; sagittal: anterior on screen LEFT.
import * as G from "../s2t/viewer/geometry.js";

let fail = 0;
const check = (ok, msg) => { if (!ok) { fail++; console.log("FAIL", msg); } else console.log("ok  ", msg); };

// Voxel grid with flipped first axis and a 7 degree tilt, like an LAS EPI header.
const N = [80, 90, 70], vs = 2.0, c = 0.12187, s = 0.99255; // cos/sin of ~7 deg
const A = [[-vs, 0, 0, 79], [0, vs * s, -vs * c, -90], [0, vs * c, vs * s, -60], [0, 0, 0, 1]];
const inv = G.inv4(A);
const data = new Float32Array(N[0] * N[1] * N[2]);
const put = (p, val, r = 4) => { // bright ball of radius r mm at world point p (RAS)
  for (let i = 0; i < N[0]; i++) for (let j = 0; j < N[1]; j++) for (let k = 0; k < N[2]; k++) {
    const w = G.apply(A, [i, j, k]);
    if (G.norm(G.sub(w, p)) <= r) data[(i * N[1] + j) * N[2] + k] = val;
  }
};
const LEFT = [-40, 0, 0], ANT = [0, 40, 0], SUP = [0, 0, 40];
put(LEFT, 250); put(ANT, 150); put(SUP, 100);
const t1 = { data, shape: N, inv };
const RES = 160, mm = 1.0;

function brightest(view, radio, center, val) {
  const pl = { basis: G.planeBasis(view, radio), center, mmPerPx: mm };
  const img = G.renderPlane(new Uint8ClampedArray(RES * RES * 4), RES, pl, t1, null, 0);
  let sx = 0, sy = 0, n = 0;
  for (let y = 0; y < RES; y++) for (let x = 0; x < RES; x++) {
    if (Math.abs(img[4 * (y * RES + x)] - val) < 1) { sx += x + 0.5; sy += y + 0.5; n++; }
  }
  return { x: sx / n, y: sy / n, n, pl };
}
const C = RES / 2;
for (const radio of [true, false]) {
  const tag = radio ? "radiological" : "neurological";
  let b = brightest("axial", radio, [0, 0, 0], 250);
  check(b.n > 0 && (radio ? b.x > C + 30 : b.x < C - 30), `${tag} axial: patient-left marker on screen ${radio ? "right" : "left"} (x=${b.x.toFixed(1)})`);
  const L = G.edgeLetters(b.pl.basis);
  check(L[3] === (radio ? "L" : "R") && L[2] === (radio ? "R" : "L") && L[0] === "A", `${tag} axial letters ${L}`);
  b = brightest("axial", radio, [0, 0, 0], 150);
  check(b.y < C - 30, `${tag} axial: anterior marker at top (y=${b.y.toFixed(1)})`);
  b = brightest("coronal", radio, [0, 0, 0], 250);
  check(radio ? b.x > C + 30 : b.x < C - 30, `${tag} coronal: patient-left marker on screen ${radio ? "right" : "left"}`);
  b = brightest("coronal", radio, [0, 0, 0], 100);
  check(b.y < C - 30, `${tag} coronal: superior marker at top`);
}
let b = brightest("sagittal", true, [0, 0, 0], 150);
check(b.x < C - 30, `sagittal: anterior marker on screen left (x=${b.x.toFixed(1)})`);
check(G.edgeLetters(b.pl.basis).join("") === "SIAP", `sagittal letters ${G.edgeLetters(b.pl.basis)}`);

// Click consistency: a dot's drawn centre is where worldToPixel puts it, and clicking
// that pixel position returns the dot's world point (sub-pixel, no half-pixel bias).
data.fill(0);
const vox = [37, 41, 33];
data[(vox[0] * N[1] + vox[1]) * N[2] + vox[2]] = 255;
const dot = G.apply(A, vox);
const pl = { basis: G.planeBasis("axial", true), center: [dot[0] + 3.3, dot[1] - 7.6, dot[2]], mmPerPx: 0.5 };
const img = G.renderPlane(new Uint8ClampedArray(RES * RES * 4), RES, pl, t1, null, 0);
let sx = 0, sy = 0, sw = 0;
for (let y = 0; y < RES; y++) for (let x = 0; x < RES; x++) { const w = img[4 * (y * RES + x)]; sx += w * (x + 0.5); sy += w * (y + 0.5); sw += w; }
const [ex, ey] = G.worldToPixel(dot, RES, RES, pl.center, pl.basis, pl.mmPerPx);
check(sw > 0 && Math.abs(sx / sw - ex) < 0.05 && Math.abs(sy / sw - ey) < 0.05,
  `single voxel drawn centred at (${(sx / sw).toFixed(3)}, ${(sy / sw).toFixed(3)}), predicted (${ex.toFixed(3)}, ${ey.toFixed(3)})`);
const back = G.pixelToWorld(ex, ey, RES, RES, pl.center, pl.basis, pl.mmPerPx);
check(G.norm(G.sub(back, dot)) < 1e-9, "click at that position returns the voxel's world point");
process.exit(fail ? 1 : 0);
