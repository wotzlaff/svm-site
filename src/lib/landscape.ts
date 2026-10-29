// Heat map with contour lines and axes of the CV-error landscape over (log10 lambda, log10 gamma).
import { contour, css, mix, rgb } from "./plot";

export type Margins = { l: number; r: number; t: number; b: number };

export function paintLandscape(
  ctx: CanvasRenderingContext2D,
  LW: number,
  LH: number,
  M: Margins,
  xs: number[],
  ys: number[],
  err: number[][],
) {
  const n = err.length;
  const x0 = xs[0], x1 = xs[xs.length - 1], y0 = ys[0], y1 = ys[ys.length - 1];
  const px = (x: number) => M.l + ((x - x0) / (x1 - x0)) * (LW - M.l - M.r);
  const py = (y: number) => LH - M.b - ((y - y0) / (y1 - y0)) * (LH - M.t - M.b);
  const V = err.map((row) => row.map((e) => Math.log10(e)));
  const flat = V.flat();
  const vmin = Math.min(...flat), vmax = Math.max(...flat);
  const ink = css("--ink"), surface = rgb(css("--surface")), accent = rgb(css("--accent"));
  // the best region gets the most conspicuous colour: dark navy on light, bright blue on dark backgrounds
  const dark = 0.299 * surface[0] + 0.587 * surface[1] + 0.114 * surface[2] < 128;
  const deep = dark ? mix(accent, [255, 255, 255], 0.3) : mix(accent, rgb(ink), 0.55);
  ctx.fillStyle = css("--surface");
  ctx.fillRect(0, 0, LW, LH);
  // heat map: bilinear interpolation of the grid
  const w = Math.round(LW - M.l - M.r), h = Math.round(LH - M.t - M.b);
  const img = ctx.createImageData(w, h);
  const at = (fx: number, fy: number) => {
    const gx = Math.min(Math.max(fx, 0), n - 1.0001), gy = Math.min(Math.max(fy, 0), n - 1.0001);
    const i = Math.floor(gx), j = Math.floor(gy), a = gx - i, b = gy - j;
    return V[j][i] * (1 - a) * (1 - b) + V[j][i + 1] * a * (1 - b) + V[j + 1][i] * (1 - a) * b + V[j + 1][i + 1] * a * b;
  };
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      const v = at((xx / (w - 1)) * (n - 1), ((h - 1 - yy) / (h - 1)) * (n - 1));
      const t = Math.pow((v - vmin) / (vmax - vmin), 0.8);
      const c = t < 0.5 ? mix(deep, accent, t * 2) : mix(accent, surface, (t - 0.5) * 2 * 0.92);
      const o = 4 * (yy * w + xx);
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
  }
  const off = new OffscreenCanvas(w, h);
  off.getContext("2d")!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, M.l, M.t, LW - M.l - M.r, LH - M.t - M.b);
  // contour lines
  ctx.strokeStyle = dark ? css("--surface") : ink;
  ctx.globalAlpha = dark ? 0.4 : 0.28;
  ctx.lineWidth = 0.8;
  for (let k = 1; k < 12; k++) {
    const level = vmin + ((vmax - vmin) * k) / 12;
    ctx.beginPath();
    for (const [a, b, c, d] of contour((i, j) => V[j][i], n, n, level)) {
      ctx.moveTo(px(x0 + (a / (n - 1)) * (x1 - x0)), py(y0 + (b / (n - 1)) * (y1 - y0)));
      ctx.lineTo(px(x0 + (c / (n - 1)) * (x1 - x0)), py(y0 + (d / (n - 1)) * (y1 - y0)));
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // frame and axes
  ctx.strokeStyle = css("--muted");
  ctx.lineWidth = 1;
  ctx.strokeRect(M.l, M.t, LW - M.l - M.r, LH - M.t - M.b);
  ctx.fillStyle = css("--muted");
  ctx.font = "11px " + css("--mono");
  ctx.textAlign = "center";
  for (let t = Math.ceil(x0); t <= x1; t++) {
    ctx.fillText(String(t), px(t), LH - M.b + 15);
    ctx.beginPath(); ctx.moveTo(px(t), LH - M.b); ctx.lineTo(px(t), LH - M.b + 4); ctx.stroke();
  }
  ctx.textAlign = "right";
  for (let t = Math.ceil(y0); t <= y1; t++) {
    ctx.fillText(String(t), M.l - 7, py(t) + 4);
    ctx.beginPath(); ctx.moveTo(M.l - 4, py(t)); ctx.lineTo(M.l, py(t)); ctx.stroke();
  }
  ctx.fillStyle = css("--ink-2");
  ctx.font = "12px " + css("--sans");
  ctx.textAlign = "center";
  ctx.fillText("log₁₀ λ  (regularization)", M.l + (LW - M.l - M.r) / 2, LH - 6);
  ctx.save();
  ctx.translate(13, M.t + (LH - M.t - M.b) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("log₁₀ γ  (kernel width)", 0, 0);
  ctx.restore();
  return { px, py, x0, x1, y0, y1, dark, vmin, vmax };
}
