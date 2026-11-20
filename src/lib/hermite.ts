// Bicubic Hermite interpolation on a regular grid from node values and exact node gradients.
// Rows of every table are the y index, columns the x index. E and G are value and exact gradient of the
// SAME C1 interpolant, so G = grad E (needed e.g. by line searches).
export type HermiteGrid = {
  xs: number[];
  ys: number[];
  f: number[][];
  fx: number[][];
  fy: number[][];
};

export function createHermite({ xs, ys, f, fx, fy }: HermiteGrid) {
  const n = xs.length, m = ys.length;
  const hx = (xs[n - 1] - xs[0]) / (n - 1), hy = (ys[m - 1] - ys[0]) / (m - 1);
  // cross derivative: mean of d(fx)/dy and d(fy)/dx, central differences (one-sided on the boundary)
  const dy = (F: number[][], j: number, i: number) =>
    j === 0 ? (F[1][i] - F[0][i]) / hy : j === m - 1 ? (F[m - 1][i] - F[m - 2][i]) / hy : (F[j + 1][i] - F[j - 1][i]) / (2 * hy);
  const dx = (F: number[][], j: number, i: number) =>
    i === 0 ? (F[j][1] - F[j][0]) / hx : i === n - 1 ? (F[j][n - 1] - F[j][n - 2]) / hx : (F[j][i + 1] - F[j][i - 1]) / (2 * hx);
  const fxy = f.map((_, j) => f[j].map((__, i) => 0.5 * (dy(fx, j, i) + dx(fy, j, i))));

  // Hermite basis on [0,1]: values and derivatives; index [h00, h10, h01, h11]
  const basis = (t: number): [number[], number[]] => {
    const t2 = t * t, t3 = t2 * t;
    return [
      [2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2],
      [6 * t2 - 6 * t, 3 * t2 - 4 * t + 1, -6 * t2 + 6 * t, 3 * t2 - 2 * t],
    ];
  };

  // value and gradient (d/dx, d/dy) at a point; positions outside the grid are clamped onto it
  function eval2(x: number, y: number): { e: number; gx: number; gy: number } {
    const px = Math.min(Math.max((x - xs[0]) / hx, 0), n - 1), py = Math.min(Math.max((y - ys[0]) / hy, 0), m - 1);
    const i = Math.min(Math.floor(px), n - 2), j = Math.min(Math.floor(py), m - 2);
    const [bx, dbx] = basis(px - i), [by, dby] = basis(py - j);
    let e = 0, gx = 0, gy = 0;
    for (let b = 0; b < 2; b++) {
      for (let a = 0; a < 2; a++) {
        // basis indices for node offset a (value, slope): a=0 -> (0,1), a=1 -> (2,3)
        const v = [f[j + b][i + a], fx[j + b][i + a] * hx, fy[j + b][i + a] * hy, fxy[j + b][i + a] * hx * hy];
        const X = [bx[2 * a], bx[2 * a + 1]], dX = [dbx[2 * a], dbx[2 * a + 1]];
        const Y = [by[2 * b], by[2 * b + 1]], dY = [dby[2 * b], dby[2 * b + 1]];
        // value term f: X0 Y0; fx*hx: X1 Y0; fy*hy: X0 Y1; fxy*hx*hy: X1 Y1
        e += v[0] * X[0] * Y[0] + v[1] * X[1] * Y[0] + v[2] * X[0] * Y[1] + v[3] * X[1] * Y[1];
        gx += (v[0] * dX[0] * Y[0] + v[1] * dX[1] * Y[0] + v[2] * dX[0] * Y[1] + v[3] * dX[1] * Y[1]) / hx;
        gy += (v[0] * X[0] * dY[0] + v[1] * X[1] * dY[0] + v[2] * X[0] * dY[1] + v[3] * X[1] * dY[1]) / hy;
      }
    }
    return { e, gx, gy };
  }
  return {
    E: (x: number, y: number) => eval2(x, y).e,
    G: (x: number, y: number): [number, number] => {
      const r = eval2(x, y);
      return [r.gx, r.gy];
    },
  };
}
