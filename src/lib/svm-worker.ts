// Web Worker hosting the WebAssembly solver.
// Protocol: request {id, cmd, args}
//   -> zero or more {id, progress} messages (streaming commands such as `tune`)
//   -> one response {id, ok: true, result} | {id, ok: false, error}.
// The worker is single-threaded: a long command (`tune`) blocks later requests until it is done,
// so the client cancels it by terminating the worker.
import { createSolverCore } from "./svm-core";
import { transferables } from "./svm-transfer";

const core = createSolverCore();
const scope = self as unknown as {
  onmessage: ((ev: MessageEvent) => void) | null;
  postMessage(msg: unknown, transfer?: Transferable[]): void;
};

scope.onmessage = async (ev: MessageEvent<{ id: number; cmd: string; args: unknown }>) => {
  const { id, cmd, args } = ev.data;
  try {
    const result = await core.handle(cmd, args, (progress) => scope.postMessage({ id, progress }, transferables(progress)));
    scope.postMessage({ id, ok: true, result }, transferables(result));
  } catch (err) {
    scope.postMessage({ id, ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
