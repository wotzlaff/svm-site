// UI helpers shared by the demos' hyperparameter tuning: the inset plots (path in the (log10 lambda,
// log10 gamma) plane and CV loss per evaluation) and a pacer that shows streamed evaluations one by one.
import { css, fmt, onTheme, setupCanvas } from "./plot";

export type TunePoint = { ll: number; lg: number; value: number }; // log10 lambda, log10 gamma, CV error
export type Range = [number, number];

export const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Builds the two inset canvases inside `host` (an empty element); `plane` are the log10 slider ranges. */
export function createTuneInset(host: HTMLElement, plane: { lam: Range; gamma: Range }) {
  const mk = (label: string) => {
    const c = document.createElement("canvas");
    c.setAttribute("role", "img");
    c.setAttribute("aria-label", label);
    host.appendChild(c);
    return c;
  };
  const W = 250, H = 170;
  const cPath = mk("Path of the optimizer in the log lambda, log gamma plane");
  const cErr = mk("Cross-validation loss per evaluation");
  const ctxP = setupCanvas(cPath, W, H), ctxE = setupCanvas(cErr, W, H);
  let hist: TunePoint[] = [];

  const font = () => `10.5px ${css("--sans") || "sans-serif"}`;
  function frame(ctx: CanvasRenderingContext2D, xl: string, yl: string) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = css("--surface");
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = css("--muted");
    ctx.font = font();
    ctx.textAlign = "right";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(xl, W - 8, H - 4);
    ctx.save();
    ctx.translate(11, 6);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "right";
    ctx.fillText(yl, 0, 0);
    ctx.restore();
  }
  const L = 46, R = 12, T = 8, B = 32;

  function draw() {
    // path plane
    frame(ctxP, "log₁₀ λ", "log₁₀ γ");
    const sx = (v: number) => L + ((v - plane.lam[0]) / (plane.lam[1] - plane.lam[0])) * (W - L - R);
    const sy = (v: number) => H - B - ((v - plane.gamma[0]) / (plane.gamma[1] - plane.gamma[0])) * (H - T - B);
    ctxP.strokeStyle = css("--line");
    ctxP.lineWidth = 1;
    ctxP.strokeRect(L, T, W - L - R, H - T - B);
    ctxP.fillStyle = css("--muted");
    ctxP.font = font();
    ctxP.textAlign = "center";
    for (const v of [plane.lam[0], plane.lam[1]]) ctxP.fillText(String(v), sx(v), H - B + 12);
    ctxP.textAlign = "right";
    for (const v of [plane.gamma[0], plane.gamma[1]]) ctxP.fillText(String(v), L - 4, sy(v) + 3);
    if (hist.length) {
      ctxP.strokeStyle = css("--accent");
      ctxP.lineWidth = 1.5;
      ctxP.beginPath();
      hist.forEach((p, i) => ctxP[i ? "lineTo" : "moveTo"](sx(p.ll), sy(p.lg)));
      ctxP.stroke();
      hist.forEach((p, i) => {
        ctxP.fillStyle = i === hist.length - 1 ? css("--accent-2") : css("--accent");
        ctxP.beginPath();
        ctxP.arc(sx(p.ll), sy(p.lg), i === hist.length - 1 ? 4 : 2.2, 0, 2 * Math.PI);
        ctxP.fill();
      });
      ctxP.strokeStyle = css("--ink");
      ctxP.lineWidth = 1.2;
      ctxP.beginPath();
      ctxP.arc(sx(hist[0].ll), sy(hist[0].lg), 4.5, 0, 2 * Math.PI);
      ctxP.stroke();
    }
    // CV loss per evaluation
    frame(ctxE, "evaluation", "CV loss");
    ctxE.strokeStyle = css("--line");
    ctxE.lineWidth = 1;
    ctxE.strokeRect(L, T, W - L - R, H - T - B);
    if (hist.length) {
      const vs = hist.map((p) => p.value);
      let lo = Math.min(...vs), hi = Math.max(...vs);
      if (hi - lo < 1e-9) [lo, hi] = [lo - 0.01, hi + 0.01];
      const nx = Math.max(hist.length - 1, 1);
      const ex = (i: number) => L + (hist.length === 1 ? 0.5 : i / nx) * (W - L - R);
      const ey = (v: number) => H - B - ((v - lo) / (hi - lo)) * (H - T - B - 8) - 4;
      ctxE.fillStyle = css("--muted");
      ctxE.font = font();
      ctxE.textAlign = "right";
      ctxE.fillText(fmt(hi, 2), L - 4, ey(hi) + 3);
      ctxE.fillText(fmt(lo, 2), L - 4, ey(lo) + 3);
      ctxE.textAlign = "center";
      ctxE.fillText("1", ex(0), H - B + 12);
      if (hist.length > 1) ctxE.fillText(String(hist.length), ex(hist.length - 1), H - B + 12);
      ctxE.strokeStyle = css("--accent");
      ctxE.lineWidth = 1.5;
      ctxE.beginPath();
      hist.forEach((p, i) => ctxE[i ? "lineTo" : "moveTo"](ex(i), ey(p.value)));
      ctxE.stroke();
      ctxE.fillStyle = css("--accent-2");
      ctxE.beginPath();
      ctxE.arc(ex(hist.length - 1), ey(hist[hist.length - 1].value), 3.5, 0, 2 * Math.PI);
      ctxE.fill();
    }
  }
  onTheme(draw);
  draw();
  return {
    set(h: TunePoint[]) {
      hist = h;
      draw();
    },
  };
}

/**
 * Shows streamed items one at a time (so that a run that finishes in a few milliseconds is still
 * followable). With reduced motion, only an occasional latest item is shown.
 */
export function createPacer<T>(show: (item: T) => void) {
  const queue: T[] = [];
  let timer = 0;
  let lastShown = -Infinity;
  let waiters: (() => void)[] = [];
  const settle = () => {
    waiters.forEach((w) => w());
    waiters = [];
  };
  const gap = () => (reducedMotion() ? 300 : queue.length > 25 ? 30 : 140);
  function schedule() {
    if (!queue.length) return settle();
    if (timer) return;
    timer = window.setTimeout(step, Math.max(0, lastShown + gap() - performance.now()));
  }
  function step() {
    timer = 0;
    if (reducedMotion() && queue.length > 1) queue.splice(0, queue.length - 1);
    const item = queue.shift();
    if (item !== undefined) {
      lastShown = performance.now();
      show(item);
    }
    schedule();
  }
  return {
    push(item: T) {
      queue.push(item);
      schedule();
    },
    /** Resolves when every queued item has been shown. */
    drain: () => new Promise<void>((res) => (queue.length || timer ? waiters.push(res) : res())),
    clear() {
      queue.length = 0;
      clearTimeout(timer);
      timer = 0;
      settle();
    },
  };
}
