// Promise API for the solver worker, with request coalescing and a main-thread fallback.
import { transferables } from "./svm-transfer";
import {
  type SvmCmd, type SvmCommands, type SvmProgress, type SvmTrainArgs, type SvmTrainResult,
  type SvmTuneArgs, type SvmTuneProgress, type SvmTuneResult, type SvmTraceArgs, type SvmTraceResult,
  type SvmCvArgs, type SvmCvResult, type SvmKind, type SvmTuneArdArgs, type SvmTuneArdProgress,
  type SvmTuneArdResult, type SvmFitArdArgs, type SvmFitArdResult,
} from "./svm-core";
export type {
  SvmTrainArgs, SvmTrainResult, SvmTuneArgs, SvmTuneProgress, SvmTuneResult, SvmTraceArgs, SvmTraceResult,
  SvmCvArgs, SvmCvResult, SvmKind, SvmTuneArdArgs, SvmTuneArdProgress, SvmTuneArdResult, SvmFitArdArgs, SvmFitArdResult,
};

/** Rejection reason of a request that was replaced by a newer one of the same command. */
export class SvmSupersededError extends Error {
  constructor() {
    super("superseded by a newer request");
    this.name = "SvmSupersededError";
  }
}

/** Rejection reason of requests that were abandoned by `SvmClient.cancel()`. */
export class SvmCancelledError extends Error {
  constructor() {
    super("cancelled");
    this.name = "SvmCancelledError";
  }
}

/** The worker itself failed (could not be created or loaded), as opposed to a solver error. */
class WorkerFailure extends Error {}

type Progress = ((p: unknown) => void) | undefined;
type Backend = {
  run(cmd: string, args: unknown, transfer: Transferable[], onProgress: Progress): Promise<unknown>;
  /** Stops the backend for good; pending requests are rejected with `SvmCancelledError`. */
  terminate(): void;
};

function workerBackend(wasmUrl: string, optimizerUrl: string): Backend {
  const worker = new Worker(new URL("./svm-worker.ts", import.meta.url), { type: "module" });
  const waiting = new Map<number, { resolve(v: unknown): void; reject(e: unknown): void; onProgress: Progress }>();
  let nextId = 1;
  let initialised = false;
  const failAll = (msg: string) => {
    // before init nothing has been transferred, so the caller may fall back to the main thread;
    // afterwards the arguments are detached and the failure is reported as an ordinary error
    for (const w of waiting.values()) w.reject(initialised ? new Error(msg) : new WorkerFailure(msg));
    waiting.clear();
  };
  worker.onmessage = (ev: MessageEvent) => {
    const { id, ok, result, error, progress } = ev.data;
    const w = waiting.get(id);
    if (!w) return;
    if ("progress" in ev.data) return w.onProgress?.(progress);
    waiting.delete(id);
    ok ? w.resolve(result) : w.reject(new Error(error));
  };
  worker.onerror = (ev) => {
    ev.preventDefault();
    failAll(ev.message || "worker error");
  };
  worker.onmessageerror = () => failAll("worker message error");
  const send = (cmd: string, args: unknown, transfer: Transferable[], onProgress?: Progress) =>
    new Promise<unknown>((resolve, reject) => {
      const id = nextId++;
      waiting.set(id, { resolve, reject, onProgress });
      worker.postMessage({ id, cmd, args }, transfer);
    });
  const ready = send("init", { wasmUrl, optimizerUrl }, []).then(() => {
    initialised = true;
  });
  ready.catch(() => {}); // surfaced through the first real request
  return {
    run: async (cmd, args, transfer, onProgress) => (await ready, send(cmd, args, transfer, onProgress)),
    terminate() {
      worker.terminate();
      for (const w of waiting.values()) w.reject(new SvmCancelledError());
      waiting.clear();
    },
  };
}

function localBackend(wasmUrl: string, optimizerUrl: string): Backend {
  const ready = import("./svm-core").then(async ({ createSolverCore }) => {
    const core = createSolverCore();
    await core.handle("init", { wasmUrl, optimizerUrl });
    return core;
  });
  // yield to the event loop first so that a stale queued request can still be dropped
  return {
    run: async (cmd, args, _transfer, onProgress) =>
      (await new Promise((r) => setTimeout(r)), (await ready).handle(cmd, args, onProgress)),
    terminate() {}, // a synchronous main-thread run cannot be interrupted
  };
}

type Request = {
  cmd: SvmCmd;
  args: unknown;
  transfer: Transferable[];
  onProgress?: (p: unknown) => void;
  resolve(v: unknown): void;
  reject(e: unknown): void;
  superseded: boolean;
};

/**
 * Runs commands one at a time. Per command, only the latest request matters: a request still
 * queued is rejected with `SvmSupersededError` when a newer one arrives; a request already running
 * finishes in the worker, but its result is discarded (rejected the same way) if a newer one exists.
 */
export class SvmClient {
  private backend: Backend | null = null;
  private inflight: Request | null = null;
  private queued = new Map<SvmCmd, Request>();
  /** "worker" or "main" (fallback); known after the first request has been started. */
  mode: "worker" | "main" | null = null;

  private optimizerUrl: string;

  constructor(private wasmUrl: string, private useWorker = true) {
    this.wasmUrl = new URL(wasmUrl, location.href).href;
    this.optimizerUrl = new URL("svm_browser_optimizer_bg.wasm", this.wasmUrl).href;
  }

  private ensureBackend(): Backend {
    if (!this.backend) {
      try {
        if (!this.useWorker || typeof Worker === "undefined") throw new Error("no worker");
        this.backend = workerBackend(this.wasmUrl, this.optimizerUrl);
        this.mode = "worker";
      } catch {
        this.backend = localBackend(this.wasmUrl, this.optimizerUrl);
        this.mode = "main";
      }
    }
    return this.backend;
  }

  private call<C extends SvmCmd>(
    cmd: C, args: SvmCommands[C][0], transfer: Transferable[], onProgress?: (p: unknown) => void,
  ): Promise<SvmCommands[C][1]> {
    return new Promise((resolve, reject) => {
      const old = this.queued.get(cmd);
      if (old) old.reject(new SvmSupersededError());
      if (this.inflight?.cmd === cmd) this.inflight.superseded = true;
      this.queued.set(cmd, { cmd, args, transfer, onProgress, resolve: resolve as (v: unknown) => void, reject, superseded: false });
      this.pump();
    });
  }

  private async pump() {
    if (this.inflight) return;
    const first = this.queued.entries().next();
    if (first.done) return;
    const [cmd, req] = first.value;
    this.queued.delete(cmd);
    this.inflight = req;
    try {
      let result: unknown;
      try {
        result = await this.ensureBackend().run(req.cmd, req.args, req.transfer, (p) => {
          if (!req.superseded) req.onProgress?.(p);
        });
      } catch (err) {
        if (!(err instanceof WorkerFailure) || this.mode === "main") throw err;
        // the worker could not be initialised: run the same code on the main thread (args were not transferred)
        this.backend = localBackend(this.wasmUrl, this.optimizerUrl);
        this.mode = "main";
        result = await this.backend.run(req.cmd, req.args, [], req.onProgress);
      }
      if (req.superseded) req.reject(new SvmSupersededError());
      else req.resolve(result);
    } catch (err) {
      req.reject(req.superseded ? new SvmSupersededError() : err);
    } finally {
      this.inflight = null;
      this.pump();
    }
  }

  /**
   * Abandons all pending and running requests. The worker is terminated (a running `tune` is a
   * synchronous loop that cannot be interrupted otherwise) and recreated lazily by the next
   * request, which re-initialises the wasm module. The rejected requests get `SvmCancelledError`.
   * On the main-thread fallback a running command cannot be interrupted; only queued ones are dropped.
   */
  cancel() {
    for (const q of this.queued.values()) q.reject(new SvmCancelledError());
    this.queued.clear();
    if (this.mode === "worker" && this.backend) {
      this.backend.terminate();
      this.backend = null;
    } else if (this.inflight) {
      this.inflight.superseded = true; // discard the result of the uninterruptible run
    }
  }

  /** Trains a model and evaluates it on `grid` and on the training points. */
  train(a: SvmTrainArgs): Promise<SvmTrainResult> {
    // the typed arrays are moved to the worker, so pass buffers that the caller does not reuse
    return this.call("train", a, transferables(a));
  }

  /** One cross-validation evaluation (error and hypergradient). */
  cv(a: SvmCvArgs): Promise<SvmCvResult> {
    return this.call("cv", a, transferables(a));
  }

  /** L-BFGS-B over (log lambda, log gamma); `onProgress` receives every objective evaluation. */
  tune(a: SvmTuneArgs, onProgress: (p: SvmProgress["tune"]) => void): Promise<SvmTuneResult> {
    return this.call("tune", a, transferables(a), onProgress as (p: unknown) => void);
  }

  /** L-BFGS-B over (log lambda, log gamma_1..gamma_dim) (ARD); `onProgress` receives every objective evaluation. */
  tuneArd(a: SvmTuneArdArgs, onProgress: (p: SvmProgress["tuneArd"]) => void): Promise<SvmTuneArdResult> {
    return this.call("tuneArd", a, transferables(a), onProgress as (p: unknown) => void);
  }

  /** Trains with per-feature kernel widths and returns the decision values on `test`. */
  fitArd(a: SvmFitArdArgs): Promise<SvmFitArdResult> {
    return this.call("fitArd", a, transferables(a));
  }

  /** Records an SMO run (at most `maxFrames` frames) with the decision function of every frame. */
  trace(a: SvmTraceArgs): Promise<SvmTraceResult> {
    return this.call("trace", a, transferables(a));
  }
}

const clients = new Map<string, SvmClient>();
/** One client (one worker) per wasm URL and page. */
export function getSvmClient(wasmUrl: string): SvmClient {
  let c = clients.get(wasmUrl);
  if (!c) clients.set(wasmUrl, (c = new SvmClient(wasmUrl, !new URLSearchParams(location.search).has("noworker"))));
  return c;
}
