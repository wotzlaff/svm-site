// Dependency-free helper shared by the client, the worker and the solver core.

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
