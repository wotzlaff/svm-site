// Small helpers shared by the interactive explainers (canvas plots, theme colours, solver loading).

export function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Any CSS colour (the build minifies "#ffffff" to "#fff") -> [r, g, b]. */
export function rgb(color: string): [number, number, number] {
  const c = document.createElement("canvas").getContext("2d")!;
  c.fillStyle = "#000";
  c.fillStyle = color;
  const h = String(c.fillStyle).replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
}

export function mix(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [0, 1, 2].map((k) => Math.round(a[k] * (1 - t) + b[k] * t)) as [number, number, number];
}

export function fmt(x: number, digits = 3): string {
  if (x === 0) return "0";
  return Math.abs(x) >= 0.01 && Math.abs(x) < 1000 ? x.toPrecision(digits) : x.toExponential(1);
}

/** Canvas whose backing store is `scale` times larger than its CSS size (crisp on high-DPI screens). */
export function setupCanvas(canvas: HTMLCanvasElement, w: number, h: number, scale = 2) {
  canvas.width = w * scale;
  canvas.height = h * scale;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  return ctx;
}

/** Line segments of the level set {v = level} of values on an n x m grid (marching squares).
 *  `at(i, j)` returns the value at column i, row j; the result is in grid coordinates. */
export function contour(
  at: (i: number, j: number) => number,
  n: number,
  m: number,
  level: number,
): [number, number, number, number][] {
  const segs: [number, number, number, number][] = [];
  const t = (a: number, b: number) => (level - a) / (b - a);
  for (let j = 0; j < m - 1; j++) {
    for (let i = 0; i < n - 1; i++) {
      const v = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
      const pts: [number, number][] = [];
      const edges: [number, number, number, number, number, number][] = [
        [0, 1, i, j, 1, 0],
        [1, 2, i + 1, j, 0, 1],
        [3, 2, i, j + 1, 1, 0],
        [0, 3, i, j, 0, 1],
      ];
      for (const [a, b, x0, y0, dx, dy] of edges) {
        if (v[a] > level !== v[b] > level) {
          const s = t(v[a], v[b]);
          pts.push([x0 + dx * s, y0 + dy * s]);
        }
      }
      if (pts.length >= 2) segs.push([...pts[0], ...pts[1]] as [number, number, number, number]);
      if (pts.length === 4) segs.push([...pts[2], ...pts[3]] as [number, number, number, number]);
    }
  }
  return segs;
}

export type Solver = {
  memory: WebAssembly.Memory;
  alloc(n: number): number;
  dealloc(ptr: number, n: number): void;
  train(x: number, n: number, dim: number, y: number, kind: number, lam: number, gamma: number,
        smoothing: number, epsilon: number, tol: number): bigint;
  decision(p: number, m: number, out: number): number;
  support(out: number): bigint;
  stat(which: number): number;
};

export async function loadSolver(url: string): Promise<Solver> {
  const { instance } = await WebAssembly.instantiateStreaming(fetch(url), {}).catch(async () =>
    WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), {}),
  );
  return instance.exports as unknown as Solver;
}

/** Calls `f` on every theme change (toggle button or system preference). */
export function onTheme(f: () => void) {
  window.addEventListener("themechange", f);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", f);
}
