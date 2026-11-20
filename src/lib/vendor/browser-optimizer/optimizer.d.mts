export type OptimizerProblem = {
  initial: ArrayLike<number>;
  lower: ArrayLike<number>;
  upper: ArrayLike<number>;
  evaluate(x: Float64Array): { value: number; gradient: ArrayLike<number> };
  maxIterations?: number;
  gradientTolerance?: number;
};
export type OptimizerResult = {
  x: Float64Array;
  value: number;
  iterations: number;
  evaluations: number;
  status: "converged" | "iteration-limit" | "stopped";
};
export function createOptimizer(wasm?: string | URL | BufferSource): Promise<{ minimize(p: OptimizerProblem): OptimizerResult }>;
