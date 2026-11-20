// Solver core shared by the Web Worker (svm-worker.ts) and the main-thread fallback (svm-client.ts).
// It has no DOM dependencies. New commands are added to `commands` and to the `SvmCommands` type.

export type SvmKind = "classification" | "regression" | "logistic";

export type SvmTrainArgs = {
  /** Row-major samples, `n * dim` values. */
  x: Float64Array;
  /** Targets (labels +-1 for classification), `n` values. */
  y: Float64Array;
  dim: number;
  /**
   * "classification" (hinge loss), "regression" (epsilon-SVR) or "logistic" (kernel logistic regression,
   * labels +-1, dense Newton). For "logistic" the decision function is the logit f (P(+1) = 1 / (1 + e^-f)),
   * `smoothing` and `epsilon` are ignored, `tol` is on the optimality norm (use about 1e-8) and
   * `steps`/`gap` are Newton iterations / the final optimality norm.
   */
  kind: SvmKind;
  lambda: number;
  gamma: number;
  smoothing: number;
  epsilon: number;
  tol: number;
  /** Row-major evaluation points (`m * dim` values); the decision function on them is returned. */
  grid: Float64Array;
};

export type SvmTrainResult = {
  /** Number of SMO steps; negative if the solver rejected the input (the arrays are then empty). */
  steps: number;
  /** Support vector indices as numbered by the solver (regression: two copies per sample). */
  support: Float64Array;
  bias: number;
  gap: number;
  dual: number;
  violation: number;
  /** Wall time of the training call in seconds. */
  seconds: number;
  decisionOnGrid: Float64Array;
  decisionOnTrain: Float64Array;
};

export type SvmCvArgs = {
  x: Float64Array;
  y: Float64Array;
  dim: number;
  kind: SvmKind;
  /** Number of folds (assignment: `makeFolds`, deterministic). */
  folds: number;
  /** Natural logarithms of lambda and gamma. */
  logLambda: number;
  logGamma: number;
  /** Training loss smoothing; the gradient is only exact for a positive value. */
  smoothing: number;
  epsilon: number;
  /** SMO tolerance of the fold problems. */
  tol: number;
};
export type SvmCvResult = { value: number; gradLogLambda: number; gradLogGamma: number; seconds: number };

export type SvmTuneArgs = Omit<SvmCvArgs, "logLambda" | "logGamma"> & {
  /** Start point and box bounds in natural logs: [log lambda, log gamma]. */
  initial: [number, number];
  lower: [number, number];
  upper: [number, number];
  maxIterations: number;
  /** Stops when the projected gradient (w.r.t. the natural logs) is below this value. */
  gradientTolerance: number;
  /** Hard budget of objective evaluations; the best point seen so far is returned (status "evaluation-limit"). */
  maxEvaluations: number;
  /** SMO tolerance of the training runs that produce the displayed decision function. */
  trainTol: number;
  grid: Float64Array;
};
/** Streamed once per objective evaluation (accepted or trial point). */
export type SvmTuneProgress = {
  evaluation: number;
  logLambda: number;
  logGamma: number;
  value: number;
  gradient: [number, number];
  /** Model trained at this point: decision function on the grid, support vector indices. */
  decisionOnGrid: Float64Array;
  support: Float64Array;
};
export type SvmTuneResult = {
  logLambda: number;
  logGamma: number;
  value: number;
  iterations: number;
  evaluations: number;
  status: string;
  seconds: number;
  /** Model trained at the optimum. */
  final: SvmTrainResult;
};

/** ARD tuning: one RBF width per feature, L-BFGS-B over (log lambda, log gamma_1..gamma_dim). */
export type SvmTuneArdArgs = Omit<SvmTuneArgs, "initial" | "lower" | "upper" | "grid" | "trainTol"> & {
  /** Start point and box bounds in natural logs: [log lambda, log gamma_1, ..., log gamma_dim]. */
  initial: Float64Array;
  lower: Float64Array;
  upper: Float64Array;
};
/** Streamed once per objective evaluation. */
export type SvmTuneArdProgress = {
  evaluation: number;
  logLambda: number;
  logGammas: Float64Array;
  value: number;
  /** [d/dlog lambda, d/dlog gamma_1, ...] */
  gradient: Float64Array;
};
export type SvmTuneArdResult = {
  logLambda: number;
  logGammas: Float64Array;
  value: number;
  iterations: number;
  evaluations: number;
  status: string;
  seconds: number;
};

/** Trains with one RBF width per feature and evaluates the decision function on `test`. */
export type SvmFitArdArgs = {
  x: Float64Array;
  y: Float64Array;
  dim: number;
  kind: SvmKind;
  lambda: number;
  /** `dim` values (all equal = ordinary isotropic RBF kernel). */
  gammas: Float64Array;
  smoothing: number;
  epsilon: number;
  tol: number;
  test: Float64Array;
};
export type SvmFitArdResult = { steps: number; decisionOnTest: Float64Array; seconds: number };

export type SvmTraceArgs = SvmTrainArgs & {
  /** Upper bound of the number of frames (including the initial and the final one). */
  maxFrames: number;
};
export type SvmTraceResult = {
  steps: number;
  frames: number;
  /** Per frame: SMO step, duality gap, violation (NaN for frame 0), bias. */
  frameStep: Float64Array;
  frameGap: Float64Array;
  frameViolation: Float64Array;
  frameBias: Float64Array;
  /** `frames * n` expansion coefficients (non-zero = support vector at that moment). */
  coef: Float64Array;
  /** `frames * m` decision values on the grid (single precision to keep the transfer small). */
  decision: Float32Array;
  gap: number;
  seconds: number;
};

/** Command table: name -> [arguments, result]. Extend here. */
export type SvmCommands = {
  init: [{ wasmUrl: string; optimizerUrl: string }, null];
  train: [SvmTrainArgs, SvmTrainResult];
  cv: [SvmCvArgs, SvmCvResult];
  tune: [SvmTuneArgs, SvmTuneResult];
  trace: [SvmTraceArgs, SvmTraceResult];
  tuneArd: [SvmTuneArdArgs, SvmTuneArdResult];
  fitArd: [SvmFitArdArgs, SvmFitArdResult];
};
/** Progress payload type per command (commands without an entry never stream). */
export type SvmProgress = { tune: SvmTuneProgress; tuneArd: SvmTuneArdProgress };
export type SvmCmd = keyof SvmCommands;

type Exports = {
  memory: WebAssembly.Memory;
  alloc(n: number): number;
  dealloc(ptr: number, n: number): void;
  train(x: number, n: number, dim: number, y: number, kind: number, lam: number, gamma: number,
        smoothing: number, epsilon: number, tol: number): bigint;
  decision(p: number, m: number, out: number): number;
  support(out: number): bigint;
  stat(which: number): number;
  cv_objective(x: number, n: number, dim: number, y: number, foldOf: number, k: number, kind: number,
               logLambda: number, logGamma: number, smoothing: number, epsilon: number, tol: number,
               out: number): number;
  cv_objective_ard(x: number, n: number, dim: number, y: number, foldOf: number, k: number, kind: number,
                   logLambda: number, logGammas: number, smoothing: number, epsilon: number, tol: number,
                   out: number): number;
  train_trace(x: number, n: number, dim: number, y: number, kind: number, lam: number, gamma: number,
              smoothing: number, epsilon: number, tol: number, period: number, maxFrames: number): bigint;
  trace_len(): bigint;
  trace_frame(i: number, out: number): bigint;
  trace_decision(i: number, points: number, m: number, out: number): number;
};

async function loadWasm(url: string): Promise<Exports> {
  const { instance } = await WebAssembly.instantiateStreaming(fetch(url), {}).catch(async () =>
    WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), {}),
  );
  return instance.exports as unknown as Exports;
}

/** Seeded 32-bit generator (mulberry32). */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Deterministic fold assignment: a seeded shuffle, stratified by class for classification (every
 * class is spread evenly over the folds). Returns the fold id of every sample as `f64`.
 */
export function makeFolds(y: ArrayLike<number>, k: number, stratify: boolean, seed = 12345): Float64Array {
  const n = y.length, rnd = mulberry32(seed);
  const groups = stratify
    ? [[...Array(n).keys()].filter((i) => y[i] > 0), [...Array(n).keys()].filter((i) => y[i] <= 0)]
    : [[...Array(n).keys()]];
  const fold = new Float64Array(n);
  let c = 0;
  for (const g of groups) {
    for (let i = g.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [g[i], g[j]] = [g[j], g[i]];
    }
    for (const idx of g) fold[idx] = c++ % k;
  }
  return fold;
}

/** Allocation scope: buffers are freed by `free()`. */
function scope(e: Exports) {
  const ptrs: [number, number][] = [];
  return {
    buf(len: number) {
      const len1 = Math.max(len, 1);
      const p = e.alloc(len1);
      ptrs.push([p, len1]);
      return p;
    },
    free() {
      for (const [p, len] of ptrs) e.dealloc(p, len);
    },
  };
}

const kindCode = (k: SvmKind) => (k === "regression" ? 1 : k === "logistic" ? 2 : 0);
const stratified = (k: SvmKind) => k !== "regression";

function train(e: Exports, a: SvmTrainArgs): SvmTrainResult {
  const n = a.y.length, m = a.grid.length / a.dim;
  const { buf, free } = scope(e);
  try {
    const px = buf(n * a.dim), py = buf(n), pg = buf(m * a.dim), pd = buf(m), pt = buf(n);
    // memory.buffer may be replaced when the heap grows: create views right before use
    new Float64Array(e.memory.buffer, px, n * a.dim).set(a.x);
    new Float64Array(e.memory.buffer, py, n).set(a.y);
    new Float64Array(e.memory.buffer, pg, m * a.dim).set(a.grid);
    const t0 = performance.now();
    const steps = Number(
      e.train(px, n, a.dim, py, kindCode(a.kind), a.lambda, a.gamma, a.smoothing, a.epsilon, a.tol),
    );
    const seconds = (performance.now() - t0) / 1000;
    if (steps < 0) {
      const none = new Float64Array(0);
      return { steps, support: none, bias: NaN, gap: NaN, dual: NaN, violation: NaN, seconds,
               decisionOnGrid: none, decisionOnTrain: none };
    }
    e.decision(pg, m, pd);
    const decisionOnGrid = new Float64Array(new Float64Array(e.memory.buffer, pd, m));
    e.decision(px, n, pt);
    const decisionOnTrain = new Float64Array(new Float64Array(e.memory.buffer, pt, n));
    const nsv = Number(e.support(0));
    const ps = buf(nsv);
    e.support(ps);
    const support = new Float64Array(new Float64Array(e.memory.buffer, ps, nsv));
    return { steps, support, bias: e.stat(0), gap: e.stat(1), dual: e.stat(2), violation: e.stat(4), seconds,
             decisionOnGrid, decisionOnTrain };
  } finally {
    free();
  }
}

/** Calls `cv_objective` (fold warm starts are kept inside the wasm module while y, folds and kind are unchanged). */
function cv(e: Exports, a: SvmCvArgs, foldOf: Float64Array): SvmCvResult {
  const n = a.y.length;
  const { buf, free } = scope(e);
  try {
    const px = buf(n * a.dim), py = buf(n), pf = buf(n), po = buf(3);
    new Float64Array(e.memory.buffer, px, n * a.dim).set(a.x);
    new Float64Array(e.memory.buffer, py, n).set(a.y);
    new Float64Array(e.memory.buffer, pf, n).set(foldOf);
    const t0 = performance.now();
    const status = e.cv_objective(px, n, a.dim, py, pf, a.folds, kindCode(a.kind),
                                  a.logLambda, a.logGamma, a.smoothing, a.epsilon, a.tol, po);
    const seconds = (performance.now() - t0) / 1000;
    if (status !== 0) throw new Error(`cv_objective failed (status ${status})`);
    const o = new Float64Array(e.memory.buffer, po, 3);
    return { value: o[0], gradLogLambda: o[1], gradLogGamma: o[2], seconds };
  } finally {
    free();
  }
}

async function tune(e: Exports, optimizerUrl: string, a: SvmTuneArgs, emit: Emit): Promise<SvmTuneResult> {
  const { createOptimizer } = await import("./vendor/browser-optimizer/optimizer.mjs");
  const optimizer = await createOptimizer(optimizerUrl);
  const foldOf = makeFolds(a.y, a.folds, stratified(a.kind));
  const trainArgs = (logLambda: number, logGamma: number): SvmTrainArgs => ({
    x: a.x, y: a.y, dim: a.dim, kind: a.kind, lambda: Math.exp(logLambda), gamma: Math.exp(logGamma),
    smoothing: a.smoothing, epsilon: a.epsilon, tol: a.trainTol, grid: a.grid,
  });
  let evaluation = 0;
  let best = { value: Infinity, x: [a.initial[0], a.initial[1]] as [number, number] };
  const t0 = performance.now();
  const budget = new Error("evaluation limit");
  const solverFailure = new Error("solver failure");
  let r: { x: ArrayLike<number>; value: number; iterations: number; evaluations: number; status: string };
  try {
    r = optimizer.minimize({
      initial: a.initial, lower: a.lower, upper: a.upper,
      maxIterations: a.maxIterations, gradientTolerance: a.gradientTolerance,
      evaluate: (theta: Float64Array) => {
        if (evaluation >= a.maxEvaluations) throw budget;
        let c: SvmCvResult;
        try {
          c = cv(e, { ...a, logLambda: theta[0], logGamma: theta[1] }, foldOf);
        } catch (err) {
          // e.g. SMO did not converge at an extreme trial point: stop and keep the best point so far
          if (evaluation > 0) throw solverFailure;
          throw err;
        }
        if (c.value < best.value) best = { value: c.value, x: [theta[0], theta[1]] };
        const m = train(e, trainArgs(theta[0], theta[1]));
        const gradient: [number, number] = [c.gradLogLambda, c.gradLogGamma];
        emit({ evaluation: ++evaluation, logLambda: theta[0], logGamma: theta[1], value: c.value, gradient,
               decisionOnGrid: m.decisionOnGrid, support: m.support } satisfies SvmTuneProgress);
        return { value: c.value, gradient };
      },
    });
  } catch (err) {
    if (err !== budget && err !== solverFailure) throw err;
    r = { x: best.x, value: best.value, iterations: NaN, evaluations: evaluation,
          status: err === budget ? "evaluation-limit" : "solver-failure" };
  }
  const final = train(e, trainArgs(r.x[0], r.x[1]));
  return { logLambda: r.x[0], logGamma: r.x[1], value: r.value, iterations: r.iterations,
           evaluations: r.evaluations, status: r.status, seconds: (performance.now() - t0) / 1000, final };
}

/** Calls `cv_objective_ard`; returns [value, d/dlog lambda, d/dlog gamma_1..dim]. */
function cvArd(e: Exports, a: SvmTuneArdArgs, theta: ArrayLike<number>, foldOf: Float64Array): Float64Array {
  const n = a.y.length, { dim } = a;
  const { buf, free } = scope(e);
  try {
    const px = buf(n * dim), py = buf(n), pf = buf(n), pg = buf(dim), po = buf(2 + dim);
    new Float64Array(e.memory.buffer, px, n * dim).set(a.x);
    new Float64Array(e.memory.buffer, py, n).set(a.y);
    new Float64Array(e.memory.buffer, pf, n).set(foldOf);
    const g = new Float64Array(e.memory.buffer, pg, dim);
    for (let d = 0; d < dim; d++) g[d] = theta[1 + d];
    const status = e.cv_objective_ard(px, n, dim, py, pf, a.folds, kindCode(a.kind), theta[0], pg,
                                      a.smoothing, a.epsilon, a.tol, po);
    if (status !== 0) throw new Error(`cv_objective_ard failed (status ${status})`);
    return new Float64Array(new Float64Array(e.memory.buffer, po, 2 + dim));
  } finally {
    free();
  }
}

async function tuneArd(e: Exports, optimizerUrl: string, a: SvmTuneArdArgs, emit: Emit): Promise<SvmTuneArdResult> {
  const { createOptimizer } = await import("./vendor/browser-optimizer/optimizer.mjs");
  const optimizer = await createOptimizer(optimizerUrl);
  const foldOf = makeFolds(a.y, a.folds, stratified(a.kind));
  let evaluation = 0;
  let best = { value: Infinity, x: Float64Array.from(a.initial) };
  const t0 = performance.now();
  const budget = new Error("evaluation limit");
  const solverFailure = new Error("solver failure");
  let r: { x: ArrayLike<number>; value: number; iterations: number; evaluations: number; status: string };
  try {
    r = optimizer.minimize({
      initial: a.initial, lower: a.lower, upper: a.upper,
      maxIterations: a.maxIterations, gradientTolerance: a.gradientTolerance,
      evaluate: (theta: Float64Array) => {
        if (evaluation >= a.maxEvaluations) throw budget;
        let o: Float64Array;
        try {
          o = cvArd(e, a, theta, foldOf);
        } catch (err) {
          if (evaluation > 0) throw solverFailure;
          throw err;
        }
        if (o[0] < best.value) best = { value: o[0], x: Float64Array.from(theta) };
        const gradient = o.slice(1);
        emit({ evaluation: ++evaluation, logLambda: theta[0], logGammas: Float64Array.from(theta.subarray(1)),
               value: o[0], gradient: gradient.slice() } satisfies SvmTuneArdProgress); // a copy: posting transfers (detaches) the buffer
        return { value: o[0], gradient };
      },
    });
  } catch (err) {
    if (err !== budget && err !== solverFailure) throw err;
    r = { x: best.x, value: best.value, iterations: NaN, evaluations: evaluation,
          status: err === budget ? "evaluation-limit" : "solver-failure" };
  }
  const xs = Float64Array.from(r.x);
  return { logLambda: xs[0], logGammas: xs.slice(1), value: r.value, iterations: r.iterations,
           evaluations: r.evaluations, status: r.status, seconds: (performance.now() - t0) / 1000 };
}

/** Scales feature d by sqrt(gamma_d), so that the RBF kernel with gamma = 1 is the ARD kernel. */
function scaleFeatures(v: Float64Array, dim: number, gammas: Float64Array): Float64Array {
  const out = new Float64Array(v.length);
  const s = Float64Array.from(gammas, Math.sqrt);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * s[i % dim];
  return out;
}

function fitArd(e: Exports, a: SvmFitArdArgs): SvmFitArdResult {
  const r = train(e, {
    x: scaleFeatures(a.x, a.dim, a.gammas), y: a.y, dim: a.dim, kind: a.kind, lambda: a.lambda, gamma: 1,
    smoothing: a.smoothing, epsilon: a.epsilon, tol: a.tol, grid: scaleFeatures(a.test, a.dim, a.gammas),
  });
  if (r.steps < 0) throw new Error(`train failed (status ${r.steps})`);
  return { steps: r.steps, decisionOnTest: r.decisionOnGrid, seconds: r.seconds };
}

function trace(e: Exports, a: SvmTraceArgs): SvmTraceResult {
  const n = a.y.length, m = a.grid.length / a.dim;
  const { buf, free } = scope(e);
  try {
    const px = buf(n * a.dim), py = buf(n), pg = buf(m * a.dim), pd = buf(m);
    new Float64Array(e.memory.buffer, px, n * a.dim).set(a.x);
    new Float64Array(e.memory.buffer, py, n).set(a.y);
    new Float64Array(e.memory.buffer, pg, m * a.dim).set(a.grid);
    const kind = kindCode(a.kind);
    const run = (period: number) =>
      Number(e.train_trace(px, n, a.dim, py, kind, a.lambda, a.gamma, a.smoothing, a.epsilon, a.tol, period, a.maxFrames));
    const t0 = performance.now();
    // first pass with a huge period only measures the number of steps (SMO steps, or Newton iterations: typically 4-10,
    // which gives period 1, i.e. every iteration is a frame)
    let steps = run(1 << 30);
    if (steps < 0) throw new Error(`train_trace failed (status ${steps})`);
    let period = Math.max(1, Math.ceil(steps / (a.maxFrames - 1)));
    steps = run(period);
    if (steps > period * (a.maxFrames - 1)) {
      period = Math.ceil(steps / (a.maxFrames - 1)); // shrinking was off, so the run was longer than measured
      steps = run(period);
    }
    const seconds = (performance.now() - t0) / 1000;
    const frames = Number(e.trace_len());
    const frameStep = new Float64Array(frames), frameGap = new Float64Array(frames);
    const frameViolation = new Float64Array(frames), frameBias = new Float64Array(frames);
    const coef = new Float64Array(frames * n);
    const decision = new Float32Array(frames * m);
    const pf = buf(4 + n);
    for (let i = 0; i < frames; i++) {
      e.trace_frame(i, pf);
      const f = new Float64Array(e.memory.buffer, pf, 4 + n);
      frameStep[i] = f[0]; frameGap[i] = f[1]; frameViolation[i] = f[2]; frameBias[i] = f[3];
      coef.set(f.subarray(4), i * n);
      e.trace_decision(i, pg, m, pd);
      decision.set(new Float64Array(e.memory.buffer, pd, m), i * m);
    }
    return { steps, frames, frameStep, frameGap, frameViolation, frameBias, coef, decision,
             gap: frameGap[frames - 1], seconds };
  } finally {
    free();
  }
}

/** Callback for streamed progress messages of a running command. */
export type Emit = (progress: unknown) => void;

/** A solver instance: `handle(cmd, args, emit)` executes one command (synchronously apart from `init` and `tune`'s lazy import). */
export function createSolverCore() {
  let wasm: Exports | null = null;
  let optimizerUrl = "";
  const need = () => {
    if (!wasm) throw new Error("solver not initialised");
    return wasm;
  };
  return {
    async handle(cmd: string, args: unknown, emit: Emit = () => {}): Promise<unknown> {
      switch (cmd) {
        case "init": {
          const a = args as SvmCommands["init"][0];
          optimizerUrl = a.optimizerUrl;
          wasm = await loadWasm(a.wasmUrl);
          return null;
        }
        case "train":
          return train(need(), args as SvmTrainArgs);
        case "cv": {
          const a = args as SvmCvArgs;
          return cv(need(), a, makeFolds(a.y, a.folds, stratified(a.kind)));
        }
        case "tune":
          return tune(need(), optimizerUrl, args as SvmTuneArgs, emit);
        case "tuneArd":
          return tuneArd(need(), optimizerUrl, args as SvmTuneArdArgs, emit);
        case "fitArd":
          return fitArd(need(), args as SvmFitArdArgs);
        case "trace":
          return trace(need(), args as SvmTraceArgs);
        default:
          throw new Error(`unknown command: ${cmd}`);
      }
    },
  };
}

export { transferables } from "./svm-transfer";
