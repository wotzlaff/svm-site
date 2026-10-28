// Content of the project page, taken from the dissertation (legacy/html/d.md).

export const thesis = {
  title:
    "Duality, Derivative-Based Training Methods and Hyperparameter Optimization for Support Vector Machines",
  author: "Nico Strasdat",
  institution: "TU Dresden",
  year: 2023,
  urn: "urn:nbn:de:bsz:14-qucosa2-874731",
  url: "https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-874731",
  pdf: "https://tud.qucosa.de/api/qucosa%3A87473/attachment/ATT-0/" as string | null,
  bibtex: `@phdthesis{strasdat2023,
  author = {Nico Strasdat},
  title  = {Duality, Derivative-Based Training Methods and
            Hyperparameter Optimization for Support Vector Machines},
  school = {Technische Universit{\\"a}t Dresden},
  type   = {Dissertation},
  year   = {2023},
  url    = {https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-874731},
  note   = {urn:nbn:de:bsz:14-qucosa2-874731}
}`,
};

export const chapters: { title: string; summary: string }[] = [
  { title: "Introduction", summary: "Motivation and structure of the work." },
  {
    title: "Basics of Support Vector Machines",
    summary: "Learning problems, kernels, losses, and a general training problem.",
  },
  {
    title: "Application of Convex Duality Theory",
    summary: "Fenchel dual, optimality conditions, representer theorem, existence and uniqueness.",
  },
  {
    title: "Examples of Classification and Regression Problems",
    summary: "SVC, LS-SVM, ε-SVR and smooth approximations of loss functions.",
  },
  {
    title: "Examples of Non-Standard Training Problems",
    summary: "Multi-class SVMs, generalised ν-SVMs and subspace SVMs.",
  },
  {
    title: "Reformulations of the Primal Training Problem",
    summary: "Finite-dimensional formulations and a system of nonlinear equations.",
  },
  {
    title: "Sequential Minimal Optimization",
    summary:
      "Convergence framework, bound-aware violation measure, working set rules and detection of fixed variables for shrinking.",
  },
  {
    title: "Descent-Based Methods for the Primal Problem",
    summary: "Preconditioned gradient methods and an efficient Newton method.",
  },
  {
    title: "Hyperparameter Optimization",
    summary: "Bilevel formulation and an implicit function approach for exact gradients.",
  },
  {
    title: "Computational Study",
    summary: "SMO measures and shrinking, primal methods, hyperparameter optimization in practice.",
  },
  { title: "Summary and Outlook", summary: "" },
];

/** Table 10.4.1 (validation loss: smoothed absolute error); columns: ε = 0.1, 0.25, 0.5, LS-SVM. */
export const regressionTable = [
  { name: "pyrim", n: "74", mse: [0.00656, 0.00625, 0.00645, 0.00597], mae: [0.0532, 0.0512, 0.0526, 0.0491] },
  { name: "triazines", n: "186", mse: [0.0244, 0.0239, 0.0237, 0.0232], mae: [0.107, 0.109, 0.111, 0.11] },
  { name: "mpg", n: "392", mse: [6.5, 6.58, 7.37, 6.63], mae: [1.81, 1.84, 1.98, 1.82] },
  { name: "housing", n: "506", mse: [10.8, 10.9, 11.9, 9.54], mae: [2.1, 2.17, 2.43, 2.07] },
  { name: "space_ga", n: "1,000*", mse: [0.0133, 0.0127, 0.0128, 0.0127], mae: [0.0849, 0.0845, 0.0854, 0.0845] },
  { name: "abalone", n: "1,000*", mse: [4.72, 4.61, 4.56, 4.53], mae: [1.5, 1.5, 1.52, 1.53] },
  { name: "cpusmall", n: "1,000*", mse: [19.1, 20.0, 32.6, 18.7], mae: [2.7, 2.96, 3.66, 2.7] },
];
