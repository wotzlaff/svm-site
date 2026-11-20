// Promise API for the solver worker, with request coalescing and a main-thread fallback.
import { transferables, type SvmCmd, type SvmCommands, type SvmTrainArgs, type SvmTrainResult } from "./svm-core";
export type { SvmTrainArgs, SvmTrainResult };

/** Rejection reason of a request that was replaced by a newer one of the same command. */
export class SvmSupersededError extends Error {
  constructor() {
    super("superseded by a newer request");
    this.name = "SvmSupersededError";
  }
}

/** The worker itself failed (could not be created or loaded), as opposed to a solver error. */
class WorkerFailure extends Error {}

type Backend = { run(cmd: string, args: unknown, transfer: Transferable[]): Promise<unknown> };

function workerBackend(wasmUrl: string): Backend {
  const worker = new Worker(new URL("./svm-worker.ts", import.meta.url), { type: "module" });
  const waiting = new Map<number, { resolve(v: unknown): void; reject(e: unknown): void }>();
  let nextId = 1;
  let initialised = false;
  const failAll = (msg: string) => {
    // before init nothing has been transferred, so the caller may fall back to the main thread;
    // afterwards the arguments are detached and the failure is reported as an ordinary error
    for (const w of waiting.values()) w.reject(initialised ? new Error(msg) : new WorkerFailure(msg));
    waiting.clear();
  };
  worker.onmessage = (ev: MessageEvent) => {
    const { id, ok, result, error } = ev.data;
    const w = waiting.get(id);
    waiting.delete(id);
    if (w) ok ? w.resolve(result) : w.reject(new Error(error));
  };
  worker.onerror = (ev) => {
    ev.preventDefault();
    failAll(ev.message || "worker error");
  };
  worker.onmessageerror = () => failAll("worker message error");
  const send = (cmd: string, args: unknown, transfer: Transferable[]) =>
    new Promise<unknown>((resolve, reject) => {
      const id = nextId++;
      waiting.set(id, { resolve, reject });
      worker.postMessage({ id, cmd, args }, transfer);
    });
  const ready = send("init", { wasmUrl }, []).then(() => {
    initialised = true;
  });
  ready.catch(() => {}); // surfaced through the first real request
  return { run: async (cmd, args, transfer) => (await ready, send(cmd, args, transfer)) };
}

function localBackend(wasmUrl: string): Backend {
  const ready = import("./svm-core").then(async ({ createSolverCore }) => {
    const core = createSolverCore();
    await core.handle("init", { wasmUrl });
    return core;
  });
  // yield to the event loop first so that a stale queued request can still be dropped
  return { run: async (cmd, args) => (await new Promise((r) => setTimeout(r)), (await ready).handle(cmd, args)) };
}

type Request = {
  cmd: SvmCmd;
  args: unknown;
  transfer: Transferable[];
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

  constructor(private wasmUrl: string, private useWorker = true) {
    this.wasmUrl = new URL(wasmUrl, location.href).href;
  }

  private ensureBackend(): Backend {
    if (!this.backend) {
      try {
        if (!this.useWorker || typeof Worker === "undefined") throw new Error("no worker");
        this.backend = workerBackend(this.wasmUrl);
        this.mode = "worker";
      } catch {
        this.backend = localBackend(this.wasmUrl);
        this.mode = "main";
      }
    }
    return this.backend;
  }

  private call<C extends SvmCmd>(cmd: C, args: SvmCommands[C][0], transfer: Transferable[]): Promise<SvmCommands[C][1]> {
    return new Promise((resolve, reject) => {
      const old = this.queued.get(cmd);
      if (old) old.reject(new SvmSupersededError());
      if (this.inflight?.cmd === cmd) this.inflight.superseded = true;
      this.queued.set(cmd, { cmd, args, transfer, resolve: resolve as (v: unknown) => void, reject, superseded: false });
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
        result = await this.ensureBackend().run(req.cmd, req.args, req.transfer);
      } catch (err) {
        if (!(err instanceof WorkerFailure) || this.mode === "main") throw err;
        // the worker could not be initialised: run the same code on the main thread (args were not transferred)
        this.backend = localBackend(this.wasmUrl);
        this.mode = "main";
        result = await this.backend.run(req.cmd, req.args, []);
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

  /** Trains a model and evaluates it on `grid` and on the training points. */
  train(a: SvmTrainArgs): Promise<SvmTrainResult> {
    // the typed arrays are moved to the worker, so pass buffers that the caller does not reuse
    return this.call("train", a, transferables(a));
  }
}

const clients = new Map<string, SvmClient>();
/** One client (one worker) per wasm URL and page. */
export function getSvmClient(wasmUrl: string): SvmClient {
  let c = clients.get(wasmUrl);
  if (!c) clients.set(wasmUrl, (c = new SvmClient(wasmUrl, !new URLSearchParams(location.search).has("noworker"))));
  return c;
}
