// Synthetic data and metrics of the live ARD explainer (no DOM dependencies).

export type ArdTask = "classification" | "regression";

export type ArdData = {
  task: ArdTask;
  seed: number;
  dim: number;
  /** Number of leading informative features; the remaining ones are pure noise. */
  informative: number;
  /** Standardised training features (row-major) and targets (labels +-1, or y standardised for regression). */
  x: Float64Array;
  y: Float64Array;
  /** Test features (standardised with the training statistics) and targets in original units. */
  testX: Float64Array;
  testY: Float64Array;
  /** Regression: mean and standard deviation of the training targets (1, 0 for classification). */
  yMean: number;
  ySd: number;
};

/** Seeded 32-bit generator (mulberry32). */
function rng(seed: number) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return { uniform, normal };
}

export const CIRCLE_RADIUS = Math.sqrt(2 / Math.PI); // the disc of this radius covers half of [-1, 1]^2
export const LABEL_NOISE = 0.08;
export const REGRESSION_NOISE = 0.5;

/** Raw samples: informative features first, then `noise` uniform noise features. */
function sample(task: ArdTask, n: number, noise: number, r: ReturnType<typeof rng>) {
  const informative = task === "classification" ? 2 : 3;
  const dim = informative + noise;
  const x = new Float64Array(n * dim), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dim; d++) {
      const u = r.uniform();
      x[i * dim + d] = task === "classification" ? 2 * u - 1 : u;
    }
    const a = x[i * dim], b = x[i * dim + 1];
    if (task === "classification") {
      const inside = Math.hypot(a, b) < CIRCLE_RADIUS;
      const flip = r.uniform() < LABEL_NOISE;
      y[i] = inside !== flip ? 1 : -1;
    } else {
      const c = x[i * dim + 2];
      y[i] = 10 * Math.sin(Math.PI * a * b) + 20 * (c - 0.5) ** 2 + REGRESSION_NOISE * r.normal();
    }
  }
  return { x, y, dim, informative };
}

/** Training set (standardised) and an independent test set; deterministic for a given seed. */
export function makeArdData(task: ArdTask, noise: number, n: number, nTest: number, seed: number): ArdData {
  const r = rng(seed * 7919 + 17);
  const tr = sample(task, n, noise, r);
  const te = sample(task, nTest, noise, r);
  const { dim } = tr;
  for (let d = 0; d < dim; d++) {
    let m = 0, v = 0;
    for (let i = 0; i < n; i++) m += tr.x[i * dim + d];
    m /= n;
    for (let i = 0; i < n; i++) v += (tr.x[i * dim + d] - m) ** 2;
    const sd = Math.sqrt(v / n) || 1;
    for (let i = 0; i < n; i++) tr.x[i * dim + d] = (tr.x[i * dim + d] - m) / sd;
    for (let i = 0; i < nTest; i++) te.x[i * dim + d] = (te.x[i * dim + d] - m) / sd;
  }
  let yMean = 0, ySd = 1;
  const y = Float64Array.from(tr.y);
  if (task === "regression") {
    yMean = y.reduce((s, v) => s + v, 0) / n;
    ySd = Math.sqrt(y.reduce((s, v) => s + (v - yMean) ** 2, 0) / n) || 1;
    for (let i = 0; i < n; i++) y[i] = (y[i] - yMean) / ySd;
  }
  return { task, seed, dim, informative: tr.informative, x: tr.x, y, testX: te.x, testY: te.y, yMean, ySd };
}

/** Test metric from decision values: accuracy (classification) or RMSE in original units (regression). */
export function testMetric(data: ArdData, decision: ArrayLike<number>): number {
  const m = data.testY.length;
  let s = 0;
  for (let i = 0; i < m; i++) {
    if (data.task === "classification") s += (decision[i] > 0 ? 1 : -1) === data.testY[i] ? 1 : 0;
    else s += (decision[i] * data.ySd + data.yMean - data.testY[i]) ** 2;
  }
  return data.task === "classification" ? s / m : Math.sqrt(s / m);
}
