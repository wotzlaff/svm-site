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
      { ref: "4.1.2", topic: "Maximal margin classifier", status: "done",
        note: "MaximalMarginLoss and loss=\"hard_margin\" in SmoothSVC/OptimizedSVC (optimized-svm 0.6.0): direct constrained QP with feasibility and KKT checks, no finite soft-margin approximation. Active-set hypergradient for kernel scalings; positive λ leaves predictions unchanged. Dense solver for moderate n; interactive hard/soft comparison on Explore." },
      { ref: "4.1.3", topic: "Logistic-regression SVM", status: "done",
        note: "LogisticLoss and loss=\"logistic\" in optimized-svm, with predict_proba. Trained by Newton's method only: no SMO, dense solution, so best for moderate n." },
      { ref: "4.2.1", topic: "Least-squares SVM", status: "done",
        note: "smo-solver: solve_least_squares; optimized-svm: SquaredLoss (closed-form reference in the tests)." },
      { ref: "4.2.2", topic: "ε-support vector regression", status: "done",
        note: "Two loss terms per sample sharing one kernel row; exact and smoothed." },
      { ref: "4.2.3", topic: "Hard ε-SVR", status: "done",
        note: "HardEpsilonInsensitiveLoss and loss=\"hard_epsilon_insensitive\" in SmoothSVR/OptimizedSVR (optimized-svm 0.6.0): minimum RKHS norm subject to |f(xᵢ) − yᵢ| ≤ εᵢ. Per-sample tolerances in the low-level API, scalar ε in estimators; infeasible constraints raise an error. Active-set kernel/ε gradients are local to a stable support set. Examples on Explore." },
      { ref: "4.4", topic: "Smooth approximations of the maximum function", status: "done",
        note: "Degrees 2, 3 and 6 in optimized-svm (Newton and hypergradient); degree 2 is solved by SMO." },
      { ref: "5.1", topic: "Multi-class classification", status: "partial",
        note: "Soft-max formulation (Prop. 5.1) as multi_class=\"softmax\" in SmoothSVC/OptimizedSVC: one joint multinomial model with common hyperparameters, Newton training and exact hypergradient (dense systems, for r·n up to a few thousand; released in optimized-svm 0.5.0; covariance-root SPD Newton/adjoint solves since 0.6.0, without kernel jitter). One-vs-rest with separate hyperparameters per class remains the default. The simplified formulation (Prop. 5.2) is not implemented." },
      { ref: "5.2", topic: "ν-support vector machines", status: "done",
        note: "1-norm constraint on the dual variables (max_asum), solved by SMO and Newton; also in the hypergradient." },
      { ref: "5.3", topic: "Subspace SVM", status: "partial", note: "Experimental: Newton training in the span of m basis points and exact hypergradient (n_basis in the estimators, basis re-selected during the optimization). Speed-up shown for fits with fixed hyperparameters on large regression problems with wide kernels; in complete hyperparameter optimizations no gain without loss of accuracy so far." },
    ],
  },
  {
    title: "Solution methods (Chapters 6–8)",
    items: [
      { ref: "6", topic: "Finite-dimensional reformulations", status: "theory",
        note: "The Newton method works on the formulation of Sec. 6.3 / 8.4." },
      { ref: "7.1–7.5, 7.8", topic: "Decomposition framework and convergence", status: "theory", note: "Basis of the SMO implementation." },
      { ref: "7.7", topic: "Working set selection", status: "done",
        note: "Maximal violating pair, second-order rules with and without step bounds, one- and two-sided search. Optional jointly solved blocks of 2–64 variables (default 2, smo-solver 0.2.0); matching regression copies can reduce zig-zagging at an active norm bound. Benchmarks compare actual accuracy and runtime, including ordinary problems where larger blocks are slower." },
      { ref: "7.9", topic: "Implementation of SMO", status: "done",
        note: "Rust, LRU kernel-row cache, permuted active set, AVX2 loops with run-time detection. Speed comparable to LIBSVM." },
      { ref: "7.10", topic: "Shrinking", status: "done", note: "Implemented with the threshold rule of the thesis; parameter and stopping studies tune the defaults (exponent 1 and a minimum removed fraction of 0.1). See Benchmarks." },
      { ref: "7.6", topic: "Bound-aware violation measure", status: "done",
        note: "Selectable in Rust and Python as stopping=\"bound_aware\" (smo-solver 0.1.2); studies found no advantage over a larger ordinary tolerance." },
      { ref: "8.2–8.3", topic: "Gradient descent and preconditioned gradient descent", status: "done",
        note: "Implemented for the study only (benchmarks/primal_first_order.py in optimized-svm), not in the library: far slower than SMO and Newton's method (preconditioned GD needs about 3·10⁵ steps for distance 10⁻³ where SMO needs about 10⁴; plain GD hardly moves). Newton's method is used for the primal problem." },
      { ref: "8.4", topic: "Newton's method for the primal problem", status: "done",
        note: "Semismooth Newton with line search after rough SMO: tolerance 0.3 or a budget of five outer updates per loss term, followed by Newton refinement. Norm-constraint activity is rechecked after Newton, so a projected warm start cannot determine the final branch incorrectly (corrected in optimized-svm 0.6.0)." },
    ],
  },
  {
    title: "Hyperparameter optimization (Chapter 9)",
    items: [
      { ref: "9.1–9.2", topic: "Bilevel formulation and implicit-function hypergradient", status: "done",
        note: "One adjoint linear system per fold, independent of the number of kernel scalings; checked against finite differences for all losses." },
      { ref: "9.2", topic: "Regularization and one scaling per feature (ARD)", status: "done", note: "Two-stage optimization with L-BFGS-B, time limits and full history." },
      { ref: "11", topic: "HPO of further parameters (ε, smoothing δ)", status: "partial",
        note: "Exact hypergradients w.r.t. ε and δ (loss_params, low-level API). Not used by the estimators: ε is not identifiable below δ, the CV error is multimodal in δ and tuning gains little (study in optimized-svm)." },
      { ref: "11", topic: "Cheaper validation than cross-validation (sampling)", status: "planned", note: "Idea from the outlook; fold parallelization is planned as well." },
    ],
  },
  {
    title: "Computational study (Chapter 10)",
    items: [
      { ref: "10.1", topic: "Checkerboard dataset", status: "done", note: "Used in the demo and the CV-error landscape on this site." },
      { ref: "10.2", topic: "SMO study (working sets, convergence, shrinking, stopping)", status: "done",
        note: "Repeated with the new solver on the checkerboard problem; see Benchmarks. Working set rules (MVP, baMVP, MuDP, MDP, tMDP; steps to a tolerance in four settings), the measures against the violation and their factors c(λ) with new fits, the smoothed hinge, support vectors and free variables over λ, the dependence on the problem size (steps, time, free variables up to n = 50 000), shrinking (speed-up over size and tolerance, the parameters η and c, a minimal removed fraction) and the stopping measure (violation vs. bound-aware vs. duality gap against the actual error)." },
      { ref: "10.3", topic: "Primal methods, Newton vs. SMO", status: "done",
        note: "Repeated with the new solvers; see Benchmarks. First-order methods (gradient descent, preconditioned gradient descent with and without the bias scaling, against SMO), the empirical error bound, Newton's method against SMO and the two-stage training for n up to 40 000 (benchmarks/primal_first_order.py, benchmarks/newton_vs_smo.py in optimized-svm), and the switching point: a measurement study over problem size, λ, kernel width, loss and solver options (benchmarks/smo_newton_switch.py) gave the default rule of tolerance 0.3 plus a step budget." },
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
  { name: "optimized-lssvr (Python, 2016–2023)", what: "LS-SVR with optimized hyperparameters (Fischer et al. 2015)",
    now: "Covered by OptimizedSVR(loss=\"squared\"): same objective and gradient (checked against the old package to 1e-13), plus group-aware CV splitters. Deprecated with 0.0.3 and archived; migration guide in the optimized-svm docs." },
  { name: "gkm (Python, 2023–2024)", what: "Generalized kernel method: kernel GLMs with deviance losses",
    now: "Folded into optimized-svm as KernelGLMRegressor / OptimizedKernelGLMRegressor: binomial, negative binomial, Gamma and Tweedie families with log links, offsets/exposure and a hypergradient-tuned NB dispersion. Deprecated with 0.1.2." },
];
