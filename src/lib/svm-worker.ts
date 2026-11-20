// Web Worker hosting the WebAssembly solver.
// Protocol: request {id, cmd, args} -> response {id, ok: true, result} | {id, ok: false, error}.
import { createSolverCore, transferables } from "./svm-core";

const core = createSolverCore();
const scope = self as unknown as {
  onmessage: ((ev: MessageEvent) => void) | null;
  postMessage(msg: unknown, transfer?: Transferable[]): void;
};

scope.onmessage = async (ev: MessageEvent<{ id: number; cmd: string; args: unknown }>) => {
  const { id, cmd, args } = ev.data;
  try {
    const result = await core.handle(cmd, args);
    scope.postMessage({ id, ok: true, result }, transferables(result));
  } catch (err) {
    scope.postMessage({ id, ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
