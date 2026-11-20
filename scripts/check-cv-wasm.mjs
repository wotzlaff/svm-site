// Node check (no browser) of the cross-validation exports of public/wasm/smo_solver_wasm.wasm against the
// oracle fixture of smo-wasm-hpo (reference values of optimized-svm), plus one run of the vendored L-BFGS-B.
//   node scripts/check-cv-wasm.mjs [path/to/cv_fixture.json]
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createOptimizer } from "../src/lib/vendor/browser-optimizer/optimizer.mjs";

const fixturePath = process.argv[2] ?? "/home/nico/svm/repos/smo-wasm-hpo/crates/smo-solver-wasm/tests/cv_fixture.json";
const cases = JSON.parse(await readFile(fixturePath, "utf8"));
const { instance } = await WebAssembly.instantiate(
  await readFile(new URL("../public/wasm/smo_solver_wasm.wasm", import.meta.url)), {});
const e = instance.exports;

function cv(c, logLambda, logGamma, tol = 1e-8) {
  const n = c.y.length;
  const bufs = [];
  const put = (arr) => {
    const p = e.alloc(Math.max(arr.length, 1));
    bufs.push([p, Math.max(arr.length, 1)]);
    new Float64Array(e.memory.buffer, p, arr.length).set(arr);
    return p;
  };
  const px = put(c.x), py = put(c.y), pf = put(c.fold_of), po = put(new Float64Array(3));
  try {
    const status = e.cv_objective(px, n, c.dim, py, pf, c.k, c.kind, logLambda, logGamma, c.smoothing, c.epsilon, tol, po);
    assert.equal(status, 0, `cv_objective status ${status}`);
    return Array.from(new Float64Array(e.memory.buffer, po, 3));
  } finally {
    for (const [p, l] of bufs) e.dealloc(p, l);
  }
}

const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-3);
let worst = 0;
for (const c of cases) {
  for (const p of c.points) {
    const [v, gl, gg] = cv(c, p.log_lambda, p.log_gamma);
    const errs = [rel(v, p.value), rel(gl, p.grad_log_lambda), rel(gg, p.grad_log_gamma)];
    worst = Math.max(worst, ...errs);
    console.log(`${c.name.padEnd(20)} (${p.log_lambda}, ${p.log_gamma})  value ${v.toFixed(6)} grad [${gl.toExponential(4)}, ${gg.toExponential(4)}]  rel.err ${Math.max(...errs).toExponential(1)}`);
  }
}
console.log(`max relative error: ${worst.toExponential(2)}`);
assert.ok(worst <= 1e-5, "relative error above 1e-5");

// vendored optimizer on the CV objective of the moons data (delta = 0.3, bounds as the demo sliders)
const optimizer = await createOptimizer(await readFile(new URL("../public/wasm/svm_browser_optimizer_bg.wasm", import.meta.url)));
const moons = cases.find((c) => c.name === "moons-hinge");
const ln10 = Math.LN10;
let evals = 0;
const t0 = performance.now();
const r = optimizer.minimize({
  initial: [-3 * ln10, 1.5 * ln10], lower: [-5 * ln10, -1 * ln10], upper: [0, 3 * ln10],
  maxIterations: 40, gradientTolerance: 1e-3,
  evaluate(x) {
    const [value, gl, gg] = cv(moons, x[0], x[1], 1e-6);
    evals++;
    console.log(`  eval ${String(evals).padStart(2)}  log10 lambda ${(x[0] / ln10).toFixed(3)}  log10 gamma ${(x[1] / ln10).toFixed(3)}  cv ${value.toFixed(4)}`);
    return { value, gradient: [gl, gg] };
  },
});
console.log(`L-BFGS-B: ${r.status}, ${r.iterations} iterations, ${r.evaluations} evaluations, cv ${r.value.toFixed(4)}, log10 (lambda, gamma) = (${(r.x[0] / ln10).toFixed(3)}, ${(r.x[1] / ln10).toFixed(3)}), ${(performance.now() - t0).toFixed(0)} ms`);
assert.ok(Number.isFinite(r.value));
