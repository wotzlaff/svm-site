# svm-site

Project website for the dissertation *Duality, Derivative-Based Training Methods and
Hyperparameter Optimization for Support Vector Machines* (Nico Strasdat) and the packages
[`smo-solver`](https://github.com/wotzlaff/smo-solver) and
[`optimized-svm`](https://github.com/wotzlaff/optimized-svm).

Static site built with [Astro](https://astro.build); formulas are rendered at build time with
KaTeX, the interactive demo runs the `smo-solver` solver compiled to WebAssembly.

## Development

```bash
npm install
npm run dev        # http://localhost:4321/svm-site/
npm run build      # static output in dist/
```

## Assets

| path | origin |
|---|---|
| `public/wasm/smo_solver_wasm.wasm` | `npm run wasm` (needs the sibling checkout `../smo-solver` and the Rust target `wasm32-unknown-unknown`) |
| `public/figures/*-{light,dark}.svg`, `hpo-summary.json` | `python scripts/make_figures.py` (needs `optimized-svm` and `matplotlib`) |
| `public/data/*.json` (HPO path, gradient slice, ARD path of the Explore page) | `python scripts/make_explorer_data.py` (needs `optimized-svm`) |
| `public/figures/thesis/*.png` | figures of the dissertation |
| `src/data/thesis.ts` | chapter outline, BibTeX and Table 10.4.1 |

## TODO

- set `site`/`base` in `astro.config.mjs` if a custom domain is used

Deployment: GitHub Actions (`.github/workflows/deploy.yml`) publishes `dist/` to GitHub Pages
on every push to `main` (enable Pages with source "GitHub Actions" in the repository settings).
