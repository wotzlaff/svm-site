// Mapping of the thesis chapters to the software. Keep in sync with ROADMAP.md of the workspace.
export type Status = "done" | "partial" | "planned" | "theory";

export const statusLabel: Record<Status, string> = {
  done: "Implemented",
  partial: "Partly",
  planned: "Planned",
  theory: "Theory",
};

export interface Item {
  ref: string;
  topic: string;
  status: Status;
  note: string;
}

export interface Group {
  title: string;
  items: Item[];
}

export const coverage: Group[] = [
  {
    title: "Training problems (Chapters 2–5)",
    items: [
      { ref: "2–3", topic: "General training problem, Fenchel duality, representer theorem", status: "theory",
        note: "Underlies the generic separable-quadratic dual that smo-solver solves; losses are described as loss terms in optimized-svm." },
      { ref: "4.1.1", topic: "Support vector classification (hinge loss)", status: "done",
        note: "Exact hinge (δ = 0) and quadratically smoothed hinge; sample and class weights. smo-solver: solve_classification; optimized-svm: SmoothSVC." },
      { ref: "4.1.2", topic: "Maximal margin classifier", status: "partial",
        note: "Approximated by λ → 0 without smoothing; no dedicated indicator loss." },
      { ref: "4.1.3", topic: "Logistic-regression SVM", status: "planned",
        note: "Needs an additional loss; the Newton method and the hypergradient only require its derivatives." },
      { ref: "4.2.1", topic: "Least-squares SVM", status: "done",
        note: "smo-solver: solve_least_squares; optimized-svm: SquaredLoss (closed-form reference in the tests)." },
      { ref: "4.2.2", topic: "ε-support vector regression", status: "done",
        note: "Two loss terms per sample sharing one kernel row; exact and smoothed." },
      { ref: "4.2.3", topic: "Hard ε-SVR", status: "partial", note: "As for the maximal margin classifier: λ → 0." },
      { ref: "4.4", topic: "Smooth approximations of the maximum function", status: "done",
        note: "Degrees 2, 3 and 6 in optimized-svm (Newton and hypergradient); degree 2 is solved by SMO." },
      { ref: "5.1", topic: "Multi-class classification", status: "partial",
        note: "One-vs-rest (with separate hyperparameters per class in OptimizedSVC). The soft-max formulation of the thesis is not implemented." },
      { ref: "5.2", topic: "ν-support vector machines", status: "done",
        note: "1-norm constraint on the dual variables (max_asum), solved by SMO and Newton; also in the hypergradient." },
      { ref: "5.3", topic: "Subspace SVM", status: "planned", note: "Not implemented." },
    ],
  },
  {
    title: "Solution methods (Chapters 6–8)",
    items: [
      { ref: "6", topic: "Finite-dimensional reformulations", status: "theory",
        note: "The Newton method works on the formulation of Sec. 6.3 / 8.4." },
      { ref: "7.1–7.5, 7.8", topic: "Decomposition framework and convergence", status: "theory", note: "Basis of the SMO implementation." },
      { ref: "7.7", topic: "Working set selection", status: "done",
        note: "Maximal violating pair, second-order rules with and without step bounds, one- and two-sided search." },
      { ref: "7.9", topic: "Implementation of SMO", status: "done",
        note: "Rust, LRU kernel-row cache, permuted active set, AVX2 loops with run-time detection. Speed comparable to LIBSVM." },
      { ref: "7.10", topic: "Shrinking", status: "done", note: "Implemented with the threshold rule of the thesis; the free parameters have not been studied systematically." },
      { ref: "7.6", topic: "Bound-aware violation measure", status: "partial",
        note: "Available in the Rust crate; not yet selectable as stopping criterion." },
      { ref: "8.2–8.3", topic: "Gradient descent and preconditioned gradient descent", status: "planned", note: "Not implemented; Newton's method is used for the primal problem." },
      { ref: "8.4", topic: "Newton's method for the primal problem", status: "done",
        note: "Semismooth Newton with line search; used after a rough SMO solve (the SMO → Newton switch is fixed at tolerance 0.1)." },
    ],
  },
  {
    title: "Hyperparameter optimization (Chapter 9)",
    items: [
      { ref: "9.1–9.2", topic: "Bilevel formulation and implicit-function hypergradient", status: "done",
        note: "One adjoint linear system per fold, independent of the number of kernel scalings; checked against finite differences for all losses." },
      { ref: "9.2", topic: "Regularization and one scaling per feature (ARD)", status: "done", note: "Two-stage optimization with L-BFGS-B, time limits and full history." },
      { ref: "11", topic: "HPO of further parameters (ε, smoothing δ)", status: "planned", note: "Suggested in the outlook of the thesis." },
      { ref: "11", topic: "Cheaper validation than cross-validation (sampling)", status: "planned", note: "Idea from the outlook; fold parallelization is planned as well." },
    ],
  },
  {
    title: "Computational study (Chapter 10)",
    items: [
      { ref: "10.1", topic: "Checkerboard dataset", status: "done", note: "Used in the demo and the CV-error landscape on this site." },
      { ref: "10.2", topic: "SMO study (working sets, optimality measures, shrinking)", status: "partial",
        note: "Figures from the thesis are shown above; the study has not been repeated with the new solver." },
      { ref: "10.3", topic: "Primal methods, Newton vs. SMO", status: "partial",
        note: "Only the switching tolerance was studied (no effect between 0.1 and 2.0, 0.01 is 2.5× slower)." },
      { ref: "10.4.1", topic: "HPO on the artificial dataset", status: "done", note: "Landscape and optimization path on the home page." },
      { ref: "10.4.2", topic: "HPO on real regression data (Table 10.4.1)", status: "partial",
        note: "Reproduced with benchmarks/thesis_regression.py: five of seven data sets agree; mpg and housing are 13–15 % worse in MSE (cause unknown, not the optimizer)." },
    ],
  },
];

export interface Legacy {
  name: string;
  what: string;
  now: string;
}

export const legacy: Legacy[] = [
  { name: "nimsmo (Nim, 2022)", what: "SMO solver with Python bindings",
    now: "Replaced by smo-solver (Rust). All ten defects found in the review are fixed or obsolete; tag legacy-nim keeps the old state." },
  { name: "optimized_svm (Python, 2022)", what: "HPO package",
    now: "Rebuilt as optimized-svm with tests and CI. Thirteen defects fixed, among them a regression HPO that did not terminate; tag legacy-2022." },
  { name: "hpo4svm (2018/19)", what: "Early OpenML experiments",
    now: "Evaluated for the data and the protocol of Table 10.4.1; not ported." },
];
