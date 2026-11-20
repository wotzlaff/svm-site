// Solver core shared by the Web Worker (svm-worker.ts) and the main-thread fallback (svm-client.ts).
// It has no DOM dependencies. New commands are added to `commands` and to the `SvmCommands` type.

export type SvmTrainArgs = {
  /** Row-major samples, `n * dim` values. */
  x: Float64Array;
  /** Targets (labels +-1 for classification), `n` values. */
  y: Float64Array;
  dim: number;
  /** "classification" or "regression" (epsilon-SVR). */
  kind: "classification" | "regression";
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
  kind: "classification" | "regression";
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
};
/** Progress payload type per command (commands without an entry never stream). */
export type SvmProgress = { tune: SvmTuneProgress };
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
      e.train(px, n, a.dim, py, a.kind === "regression" ? 1 : 0, a.lambda, a.gamma, a.smoothing, a.epsilon, a.tol),
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
    const status = e.cv_objective(px, n, a.dim, py, pf, a.folds, a.kind === "regression" ? 1 : 0,
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
  const foldOf = makeFolds(a.y, a.folds, a.kind === "classification");
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

function trace(e: Exports, a: SvmTraceArgs): SvmTraceResult {
  const n = a.y.length, m = a.grid.length / a.dim;
  const { buf, free } = scope(e);
  try {
    const px = buf(n * a.dim), py = buf(n), pg = buf(m * a.dim), pd = buf(m);
    new Float64Array(e.memory.buffer, px, n * a.dim).set(a.x);
    new Float64Array(e.memory.buffer, py, n).set(a.y);
    new Float64Array(e.memory.buffer, pg, m * a.dim).set(a.grid);
    const kind = a.kind === "regression" ? 1 : 0;
    const run = (period: number) =>
      Number(e.train_trace(px, n, a.dim, py, kind, a.lambda, a.gamma, a.smoothing, a.epsilon, a.tol, period, a.maxFrames));
    const t0 = performance.now();
    // first pass with period 1 and a tiny frame budget only measures the number of steps
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
          return cv(need(), a, makeFolds(a.y, a.folds, a.kind === "classification"));
        }
        case "tune":
          return tune(need(), optimizerUrl, args as SvmTuneArgs, emit);
        case "trace":
          return trace(need(), args as SvmTraceArgs);
        default:
          throw new Error(`unknown command: ${cmd}`);
      }
    },
  };
}

export { transferables } from "./svm-transfer";
