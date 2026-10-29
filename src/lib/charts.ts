// Small SVG chart renderers for the benchmark page. Every function re-renders the whole SVG for a
// given pixel width, so text keeps its size on narrow screens.

const NS = "http://www.w3.org/2000/svg";

export function el<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
  text?: string,
) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e;
}

export function reset(svg: SVGSVGElement, w: number, h: number) {
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const keep = [svg.querySelector("title"), svg.querySelector("desc")].filter(Boolean) as Element[];
  svg.replaceChildren(...keep);
}

/** Series colours (CSS classes `s0` ... `s3` are defined in the page). */
export const SERIES = ["s0", "s1", "s2", "s3"];

export function legend(svg: SVGSVGElement, names: string[], x: number, y: number, maxWidth: number) {
  let cx = x, cy = y;
  names.forEach((n, i) => {
    const w = 26 + n.length * 6.4;
    if (cx + w > x + maxWidth && cx > x) {
      cx = x;
      cy += 18;
    }
    svg.append(el("rect", { x: cx, y: cy - 9, width: 11, height: 11, rx: 2, class: `fill ${SERIES[i % 4]}` }));
    svg.append(el("text", { x: cx + 16, y: cy, class: "lbl" }, n));
    cx += w;
  });
  return cy;
}

export type BarsSpec = {
  categories: string[];
  compact?: string[]; // short labels used when the groups get narrow
  series: { name: string; values: number[] }[];
  yLabel: string;
  log?: boolean;
  format?: (v: number) => string;
  note?: (c: number) => string; // text above a category (e.g. speed-up)
  hline?: number; // reference line (e.g. 1 for speed-ups)
};

/** Grouped vertical bars with optional logarithmic axis. */
export function groupedBars(svg: SVGSVGElement, W: number, spec: BarsSpec) {
  const fmt = spec.format ?? ((v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2)));
  const legendRows = Math.ceil((spec.series.length * 150) / Math.max(W - 40, 200));
  const narrow = !!spec.compact && (W - 64) / spec.categories.length < 90;
  const cats = narrow ? spec.compact! : spec.categories;
  const maxLines = Math.max(...cats.map((c) => c.split("|").length));
  const M = { l: 52, r: 12, t: 22 + 18 * legendRows, b: 22 + 13 * maxLines + (spec.note ? 18 : 0) };
  const H = Math.round(Math.max(250, Math.min(360, W * 0.62))) + 13 * (maxLines - 2) + (spec.note ? 10 : 0);
  reset(svg, W, H);
  legend(svg, spec.series.map((s) => s.name), M.l, 16, W - M.l - M.r);
  const all = spec.series.flatMap((s) => s.values).filter((v) => v > 0);
  const lo = spec.log ? Math.pow(10, Math.floor(Math.log10(Math.min(...all)))) : 0;
  const hi = spec.log ? Math.pow(10, Math.ceil(Math.log10(Math.max(...all)) + 0.05)) : Math.max(...all) * 1.12;
  const py = (v: number) => {
    const t = spec.log ? (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);
    return H - M.b - t * (H - M.t - M.b);
  };
  // grid
  const ticks: number[] = [];
  if (spec.log) for (let v = lo; v <= hi * 1.0001; v *= 10) ticks.push(v);
  else {
    const step = Math.pow(10, Math.floor(Math.log10(hi / 4)));
    const s = [1, 2, 5, 10].map((m) => m * step).find((m) => hi / m <= 6) ?? step;
    for (let v = 0; v <= hi; v += s) ticks.push(v);
  }
  for (const v of ticks) {
    svg.append(el("line", { x1: M.l, x2: W - M.r, y1: py(v), y2: py(v), class: "grid" }));
    svg.append(el("text", { x: M.l - 6, y: py(v) + 3, class: "tick", "text-anchor": "end" }, fmt(v)));
  }
  svg.append(el("text", { x: 12, y: M.t + (H - M.t - M.b) / 2, class: "lbl", "text-anchor": "middle", transform: `rotate(-90 12 ${M.t + (H - M.t - M.b) / 2})` }, spec.yLabel));
  if (spec.hline !== undefined && spec.hline >= lo && spec.hline <= hi) {
    svg.append(el("line", { x1: M.l, x2: W - M.r, y1: py(spec.hline), y2: py(spec.hline), class: "hline" }));
  }
  const gw = (W - M.l - M.r) / cats.length;
  const bw = Math.min(34, (gw * 0.78) / spec.series.length);
  cats.forEach((c, i) => {
    const x0 = M.l + i * gw + (gw - bw * spec.series.length) / 2;
    spec.series.forEach((s, j) => {
      const v = s.values[i];
      if (!(v > 0)) return;
      svg.append(el("rect", { x: x0 + j * bw, y: py(v), width: bw - 2, height: Math.max(H - M.b - py(v), 1), class: `fill ${SERIES[j % 4]}` }));
      if (bw >= 18) svg.append(el("text", { x: x0 + j * bw + (bw - 2) / 2, y: py(v) - 4, class: "val", "text-anchor": "middle" }, fmt(v)));
    });
    // wrap category labels at the first space
    const parts = c.split("|");
    parts.forEach((p, k) => svg.append(el("text", { x: M.l + i * gw + gw / 2, y: H - M.b + 16 + 13 * k, class: "tick", "text-anchor": "middle" }, p)));
    const note = spec.note?.(i);
    if (note) svg.append(el("text", { x: M.l + i * gw + gw / 2, y: H - M.b + 16 + 13 * parts.length + 2, class: "note", "text-anchor": "middle" }, note));
  });
}

export type ScatterPoint = { x: number; y: number; series: number; label?: string; dx?: number; dy?: number };

/** Scatter plot with a logarithmic x axis; smaller y is better. */
export function scatter(
  svg: SVGSVGElement,
  W: number,
  spec: { title: string; xLabel: string; yLabel: string; points: ScatterPoint[]; yFormat?: (v: number) => string },
) {
  const M = { l: 64, r: 14, t: 42, b: 40 };
  const H = Math.round(Math.max(226, Math.min(296, W * 0.7)));
  reset(svg, W, H);
  const [name, ...rest] = spec.title.split(" (");
  svg.append(el("text", { x: M.l, y: 15, class: "ttl" }, name));
  if (rest.length) svg.append(el("text", { x: M.l, y: 30, class: "tick" }, "(" + rest.join(" (")));
  const xs = spec.points.map((p) => Math.log10(p.x)), ys = spec.points.map((p) => p.y);
  const x0 = Math.floor(Math.min(...xs) - 0.1), x1 = Math.ceil(Math.max(...xs) + 0.1);
  const ymin = Math.min(...ys), ymax = Math.max(...ys), pad = (ymax - ymin) * 0.18 || 0.01;
  const y0 = ymin - pad, y1 = ymax + pad;
  const px = (v: number) => M.l + ((Math.log10(v) - x0) / (x1 - x0)) * (W - M.l - M.r);
  const py = (v: number) => H - M.b - ((v - y0) / (y1 - y0)) * (H - M.t - M.b);
  const yf = spec.yFormat ?? ((v: number) => v.toPrecision(3));
  for (let t = x0; t <= x1; t++) {
    svg.append(el("line", { x1: px(10 ** t), x2: px(10 ** t), y1: M.t, y2: H - M.b, class: "grid" }));
    svg.append(el("text", { x: px(10 ** t), y: H - M.b + 15, class: "tick", "text-anchor": "middle" }, t >= 0 ? String(10 ** t) : (10 ** t).toString()));
  }
  for (let k = 0; k <= 3; k++) {
    const v = y0 + ((y1 - y0) * k) / 3;
    svg.append(el("line", { x1: M.l, x2: W - M.r, y1: py(v), y2: py(v), class: "grid" }));
    svg.append(el("text", { x: M.l - 6, y: py(v) + 3, class: "tick", "text-anchor": "end" }, yf(v)));
  }
  svg.append(el("text", { x: M.l + (W - M.l - M.r) / 2, y: H - 5, class: "lbl", "text-anchor": "middle" }, spec.xLabel));
  svg.append(el("text", { x: 12, y: M.t + (H - M.t - M.b) / 2, class: "lbl", "text-anchor": "middle", transform: `rotate(-90 12 ${M.t + (H - M.t - M.b) / 2})` }, spec.yLabel));
  // points first, then labels placed where they overlap neither a point nor another label
  const placed: [number, number, number, number][] = spec.points.map((p) => [px(p.x) - 7, py(p.y) - 7, px(p.x) + 7, py(p.y) + 7]);
  const hit = (a: number[], b: number[]) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1];
  for (const p of spec.points) {
    svg.append(el("circle", { cx: px(p.x), cy: py(p.y), r: 6, class: `fill ${SERIES[p.series % 4]}`, stroke: "var(--surface)", "stroke-width": 1.5 }));
  }
  for (const p of spec.points) {
    if (!p.label) continue;
    const w = p.label.length * 6.1, h = 11, cx = px(p.x), cy = py(p.y);
    const options: [number, number, string][] = [
      [cx + 10, cy + 4, "start"], [cx - 10, cy + 4, "end"], [cx, cy - 11, "middle"], [cx, cy + 20, "middle"],
    ];
    const fits = (o: [number, number, string]) => {
      const x0 = o[2] === "start" ? o[0] : o[2] === "end" ? o[0] - w : o[0] - w / 2;
      const box = [x0, o[1] - h, x0 + w, o[1] + 2];
      return box[0] >= 0 && box[2] <= W && !placed.some((b) => hit(box, b)) ? box : null;
    };
    let choice = options.find((o) => fits(o)) ?? options[0];
    const box = fits(choice) ?? [choice[0], choice[1] - h, choice[0] + w, choice[1] + 2];
    placed.push(box as [number, number, number, number]);
    svg.append(el("text", { x: choice[0], y: choice[1], class: "tick", "text-anchor": choice[2] }, p.label));
  }
}

export type HeatSpec = {
  rows: string[];
  cols: string[];
  values: number[][]; // relative time, >= 1
  rowLabel: string;
  colLabel: string;
  globalBest?: boolean; // outline only the fastest cell of the whole map
};

/** Heat map of relative times (1 = best of the row); the best cell of each row is outlined. */
export function heatmap(svg: SVGSVGElement, W: number, spec: HeatSpec) {
  const M = { l: 74, r: 10, t: 24, b: 46 };
  const cw = (W - M.l - M.r) / spec.cols.length, ch = Math.min(40, Math.max(28, cw * 0.8));
  const H = M.t + ch * spec.rows.length + M.b;
  reset(svg, W, H);
  const surface = css("--surface"), accent = css("--accent");
  const rgb = (c: string) => {
    const g = document.createElement("canvas").getContext("2d")!;
    g.fillStyle = "#000";
    g.fillStyle = c;
    const h = String(g.fillStyle).replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  };
  const a = rgb(accent), s = rgb(surface);
  const color = (v: number) => {
    const t = Math.min(Math.log10(v) / Math.log10(4), 1); // 1 .. 4+ -> light .. strong
    const k = 0.12 + 0.88 * t;
    return `rgb(${[0, 1, 2].map((i) => Math.round(s[i] * (1 - k) + a[i] * k)).join(",")})`;
  };
  spec.rows.forEach((r, i) => {
    svg.append(el("text", { x: M.l - 8, y: M.t + i * ch + ch / 2 + 4, class: "tick", "text-anchor": "end" }, r));
    const rowBest = spec.values[i].indexOf(Math.min(...spec.values[i]));
    const flat = spec.values.flat();
    const gmin = Math.min(...flat);
    spec.cols.forEach((c, j) => {
      const v = spec.values[i][j];
      svg.append(el("rect", { x: M.l + j * cw, y: M.t + i * ch, width: cw - 1, height: ch - 1, fill: color(v), class: "cell" }));
      if (cw > 30) {
        const dark = Math.log10(v) / Math.log10(4) > 0.55;
        svg.append(el("text", { x: M.l + j * cw + cw / 2, y: M.t + i * ch + ch / 2 + 4, class: dark ? "cellt inv" : "cellt", "text-anchor": "middle" }, v < 10 ? v.toFixed(1) : v.toFixed(0)));
      }
      if (spec.globalBest ? v === gmin : j === rowBest) svg.append(el("rect", { x: M.l + j * cw + 1, y: M.t + i * ch + 1, width: cw - 3, height: ch - 3, class: "best" }));
    });
  });
  spec.cols.forEach((c, j) => svg.append(el("text", { x: M.l + j * cw + cw / 2, y: M.t + spec.rows.length * ch + 15, class: "tick", "text-anchor": "middle" }, c)));
  svg.append(el("text", { x: M.l + (W - M.l - M.r) / 2, y: H - 6, class: "lbl", "text-anchor": "middle" }, spec.colLabel));
  svg.append(el("text", { x: M.l - 8, y: 14, class: "lbl", "text-anchor": "end" }, spec.rowLabel));
}

function css(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export type XY = { x: number; y: number; series: number };

/** Log-log scatter plot with the diagonal y = x; x and y must be positive. */
export function loglog(
  svg: SVGSVGElement,
  W: number,
  spec: { title: string; xLabel: string; yLabel: string; points: XY[]; legend: string[] },
) {
  const M = { l: 58, r: 14, t: 46, b: 42 };
  const H = Math.round(Math.max(250, Math.min(330, W * 0.82)));
  reset(svg, W, H);
  svg.append(el("text", { x: M.l, y: 15, class: "ttl" }, spec.title));
  legend(svg, spec.legend, M.l, 32, W - M.l - M.r);
  const lx = spec.points.map((p) => Math.log10(p.x)), ly = spec.points.map((p) => Math.log10(p.y));
  const lo = Math.floor(Math.min(...lx, ...ly) - 0.05), hi = Math.ceil(Math.max(...lx, ...ly) + 0.05);
  const px = (v: number) => M.l + ((v - lo) / (hi - lo)) * (W - M.l - M.r);
  const py = (v: number) => H - M.b - ((v - lo) / (hi - lo)) * (H - M.t - M.b);
  const stride = Math.max(1, Math.ceil((hi - lo) / Math.max(4, Math.floor((W - M.l - M.r) / 44))));
  for (let t = lo; t <= hi; t++) {
    svg.append(el("line", { x1: px(t), x2: px(t), y1: M.t, y2: H - M.b, class: "grid" }));
    svg.append(el("line", { x1: M.l, x2: W - M.r, y1: py(t), y2: py(t), class: "grid" }));
    if ((t - lo) % stride === 0) {
      svg.append(el("text", { x: px(t), y: H - M.b + 14, class: "tick", "text-anchor": "middle" }, `1e${t}`));
      svg.append(el("text", { x: M.l - 6, y: py(t) + 3, class: "tick", "text-anchor": "end" }, `1e${t}`));
    }
  }
  svg.append(el("line", { x1: px(lo), y1: py(lo), x2: px(hi), y2: py(hi), class: "hline" }));
  svg.append(el("text", { x: M.l + (W - M.l - M.r) / 2, y: H - 5, class: "lbl", "text-anchor": "middle" }, spec.xLabel));
  svg.append(el("text", { x: 12, y: M.t + (H - M.t - M.b) / 2, class: "lbl", "text-anchor": "middle", transform: `rotate(-90 12 ${M.t + (H - M.t - M.b) / 2})` }, spec.yLabel));
  spec.points.forEach((p, i) => {
    svg.append(el("circle", { cx: px(lx[i]), cy: py(ly[i]), r: 2.8, class: `fill ${SERIES[p.series % 4]}`, opacity: 0.65 }));
  });
}
