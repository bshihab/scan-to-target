// Coordinate math for the viewer. Pure functions, no DOM, so the same file
// runs in the browser and under node for the B5 cross-check against Python.
//
// Matrices are 4x4, row-major nested arrays: M[row][col]. They act on column
// vectors, as in the Python code: world = M * [i, j, k, 1]^T.
// Volumes are flat typed arrays in C order: index = (i * J + j) * K + k.

export function matmul(A, B) {
  const C = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += A[r][k] * B[k][c];
      C[r][c] = s;
    }
  return C;
}

// Inverse by Gauss-Jordan elimination with partial pivoting.
export function inv4(M) {
  const a = M.map((row, r) => [...row, ...[0, 1, 2, 3].map((c) => (c === r ? 1 : 0))]);
  for (let col = 0; col < 4; col++) {
    let piv = col;
    for (let r = col + 1; r < 4; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    if (Math.abs(a[piv][col]) < 1e-12) throw new Error("singular matrix");
    [a[col], a[piv]] = [a[piv], a[col]];
    const p = a[col][col];
    for (let c = 0; c < 8; c++) a[col][c] /= p;
    for (let r = 0; r < 4; r++) {
      if (r === col) continue;
      const f = a[r][col];
      if (f !== 0) for (let c = 0; c < 8; c++) a[r][c] -= f * a[col][c];
    }
  }
  return a.map((row) => row.slice(4));
}

export function apply(M, p) {
  return [
    M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + M[0][3],
    M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + M[1][3],
    M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + M[2][3],
  ];
}

// Scanner RAS <-> DICOM patient LPS: negate x and y.
export const rasToLps = (p) => [-p[0], -p[1], p[2]];
export const lpsToRas = rasToLps;

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.sqrt(dot(a, a));
export const unit = (a) => scale(a, 1 / norm(a));

// Trilinear interpolation at continuous voxel index (i, j, k). Outside -> NaN.
export function trilinear(data, shape, i, j, k) {
  const [I, J, K] = shape;
  if (i < 0 || j < 0 || k < 0 || i > I - 1 || j > J - 1 || k > K - 1) return NaN;
  const i0 = Math.min(Math.floor(i), I - 2 < 0 ? 0 : I - 2);
  const j0 = Math.min(Math.floor(j), J - 2 < 0 ? 0 : J - 2);
  const k0 = Math.min(Math.floor(k), K - 2 < 0 ? 0 : K - 2);
  const di = i - i0, dj = j - j0, dk = k - k0;
  const at = (a, b, c) => data[(a * J + b) * K + c];
  const i1 = Math.min(i0 + 1, I - 1), j1 = Math.min(j0 + 1, J - 1), k1 = Math.min(k0 + 1, K - 1);
  const c00 = at(i0, j0, k0) * (1 - di) + at(i1, j0, k0) * di;
  const c10 = at(i0, j1, k0) * (1 - di) + at(i1, j1, k0) * di;
  const c01 = at(i0, j0, k1) * (1 - di) + at(i1, j0, k1) * di;
  const c11 = at(i0, j1, k1) * (1 - di) + at(i1, j1, k1) * di;
  const c0 = c00 * (1 - dj) + c10 * dj;
  const c1 = c01 * (1 - dj) + c11 * dj;
  return c0 * (1 - dk) + c1 * dk;
}

// Display planes in scanner RAS. u points to screen right, v to screen up.
// Radiological convention (patient's right on screen left) is the clinical
// default; neurological flips the left-right axis.
export function planeBasis(view, radiological = true) {
  const lr = radiological ? [-1, 0, 0] : [1, 0, 0];
  if (view === "axial") return { u: lr, v: [0, 1, 0], n: [0, 0, 1] };
  if (view === "coronal") return { u: lr, v: [0, 0, 1], n: [0, 1, 0] };
  if (view === "sagittal") return { u: [0, -1, 0], v: [0, 0, 1], n: [1, 0, 0] };
  throw new Error("unknown view " + view);
}

// Planes defined by a trajectory from entry to target.
// "along": contains the trajectory; its in-plane perpendicular is chosen as
// close to superior (or anterior if the path is vertical) as possible.
// "probe": perpendicular to the trajectory, what the tip would see.
export function trajectoryBasis(entry, target) {
  const d = unit(sub(target, entry));
  let ref = Math.abs(d[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  const side = unit(cross(d, ref));
  const up = cross(side, d);
  return { along: { u: d, v: up, n: side }, probe: { u: side, v: up, n: d }, dir: d };
}

// World point for display pixel (px, py) of a W x H canvas whose centre maps
// to `center` with `mmPerPx` spacing.
export function pixelToWorld(px, py, W, H, center, basis, mmPerPx) {
  const x = (px - W / 2) * mmPerPx;
  const y = (H / 2 - py) * mmPerPx;
  return add(center, add(scale(basis.u, x), scale(basis.v, y)));
}

export function worldToPixel(p, W, H, center, basis, mmPerPx) {
  const r = sub(p, center);
  return [W / 2 + dot(r, basis.u) / mmPerPx, H / 2 - dot(r, basis.v) / mmPerPx, dot(r, basis.n)];
}

// Angle in degrees between a trajectory direction and a surface normal.
export function angleDeg(a, b) {
  const c = Math.max(-1, Math.min(1, dot(unit(a), unit(b))));
  return (Math.acos(c) * 180) / Math.PI;
}
