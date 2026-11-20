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

/** Command table: name -> [arguments, result]. Extend here (e.g. `cv_objective`, `train_trace`). */
export type SvmCommands = {
  init: [{ wasmUrl: string }, null];
  train: [SvmTrainArgs, SvmTrainResult];
};
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
};

async function loadWasm(url: string): Promise<Exports> {
  const { instance } = await WebAssembly.instantiateStreaming(fetch(url), {}).catch(async () =>
    WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), {}),
  );
  return instance.exports as unknown as Exports;
}

function train(e: Exports, a: SvmTrainArgs): SvmTrainResult {
  const n = a.y.length, m = a.grid.length / a.dim;
  const ptrs: [number, number][] = [];
  const buf = (len: number) => {
    const len1 = Math.max(len, 1);
    const p = e.alloc(len1);
    ptrs.push([p, len1]);
    return p;
  };
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
    for (const [p, len] of ptrs) e.dealloc(p, len);
  }
}

/** A solver instance: `handle(cmd, args)` executes one command (synchronously apart from `init`). */
export function createSolverCore() {
  let wasm: Exports | null = null;
  const need = () => {
    if (!wasm) throw new Error("solver not initialised");
    return wasm;
  };
  return {
    async handle(cmd: string, args: unknown): Promise<unknown> {
      switch (cmd) {
        case "init":
          wasm = await loadWasm((args as SvmCommands["init"][0]).wasmUrl);
          return null;
        case "train":
          return train(need(), args as SvmTrainArgs);
        default:
          throw new Error(`unknown command: ${cmd}`);
      }
    },
  };
}

/** Collects the ArrayBuffers of typed arrays at the top level of `v` (for zero-copy transfer). */
export function transferables(v: unknown): ArrayBuffer[] {
  if (!v || typeof v !== "object") return [];
  const seen = new Set<ArrayBuffer>();
  for (const x of Object.values(v)) {
    if (ArrayBuffer.isView(x) && x.buffer instanceof ArrayBuffer && x.byteOffset === 0 && x.byteLength === x.buffer.byteLength) {
      seen.add(x.buffer);
    }
  }
  return [...seen];
}
