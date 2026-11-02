import { useRef } from 'react';
import { readPalette } from './theme';
import { useCanvas } from './useCanvas';

export interface Series {
  name: string;
  values: ArrayLike<number>;
  /** 'accent' | 'ink' | 'goal' | 'trap' (palette keys) */
  color: 'accent' | 'ink' | 'goal' | 'trap';
  dashed?: boolean;
  /** Draw as faint dots instead of a line. */
  dots?: boolean;
}

interface Props {
  /** Read every frame, so the caller can mutate its data in place. */
  getSeries: () => Series[];
  xLabel: string;
  /** Label for the first x position (default '1'). */
  xStart?: string;
  aspect?: number;
  emptyText?: string;
  label: string;
  /** Optional fixed y range. */
  yRange?: [number, number];
  /** x positions (indices) drawn as small ticks along the bottom, e.g. omission trials. */
  getMarks?: () => number[];
}

/** Small generic line chart on canvas: shared x index, auto y range, zero line. */
export function LineChart({ getSeries, xLabel, xStart = '1', aspect = 0.45, emptyText = 'No data yet', label, yRange, getMarks }: Props) {
  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const P = readPalette();
    const series = getSeries();
    ctx.fillStyle = P.paper;
    ctx.fillRect(0, 0, w, h);
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const padL = 40, padR = 10, padT = 10, padB = 22;
    const pw = w - padL - padR, ph = h - padT - padB;
    const len = Math.max(0, ...series.map((s) => s.values.length));
    if (len < 2) {
      ctx.fillStyle = P.faint;
      ctx.textAlign = 'center';
      ctx.fillText(emptyText, w / 2, h / 2);
      return;
    }
    let lo = Infinity, hi = -Infinity;
    if (yRange) [lo, hi] = yRange;
    else {
      for (const s of series) for (let i = 0; i < s.values.length; i++) { lo = Math.min(lo, s.values[i]); hi = Math.max(hi, s.values[i]); }
      lo = Math.min(lo, 0); hi = Math.max(hi, 0);
      if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
      const pad = (hi - lo) * 0.06;
      hi += pad; if (lo < 0) lo -= pad;
    }
    const X = (i: number) => padL + (i / (len - 1)) * pw;
    const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * ph;
    ctx.strokeStyle = P.grid; ctx.fillStyle = P.faint; ctx.lineWidth = 1; ctx.textAlign = 'right';
    for (let k = 0; k <= 3; k++) {
      const v = lo + ((hi - lo) * k) / 3;
      ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke();
      ctx.fillText(Math.abs(hi - lo) < 5 ? v.toFixed(2) : v.toFixed(0), padL - 6, Y(v) + 4);
    }
    if (lo < 0 && hi > 0) { ctx.strokeStyle = P.faint; ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(w - padR, Y(0)); ctx.stroke(); }
    ctx.textAlign = 'right'; ctx.fillStyle = P.faint;
    ctx.fillText(xLabel, w - padR, h - 6);
    ctx.textAlign = 'left';
    ctx.fillText(xStart, padL, h - 6);
    const col = (c: Series['color']) => (c === 'accent' ? P.accent : c === 'goal' ? P.goal : c === 'trap' ? P.trap : P.ink);
    for (const s of series) {
      ctx.strokeStyle = col(s.color); ctx.fillStyle = col(s.color);
      if (s.dots) {
        ctx.globalAlpha = 0.35;
        for (let i = 0; i < s.values.length; i++) { ctx.beginPath(); ctx.arc(X(i), Y(s.values[i]), 2, 0, Math.PI * 2); ctx.fill(); }
        ctx.globalAlpha = 1;
        continue;
      }
      ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.setLineDash(s.dashed ? [6, 4] : []);
      ctx.beginPath();
      for (let i = 0; i < s.values.length; i++) { if (i === 0) ctx.moveTo(X(i), Y(s.values[i])); else ctx.lineTo(X(i), Y(s.values[i])); }
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (getMarks) {
      ctx.strokeStyle = P.trap; ctx.lineWidth = 2;
      for (const m of getMarks()) { ctx.beginPath(); ctx.moveTo(X(m), h - padB); ctx.lineTo(X(m), h - padB - 6); ctx.stroke(); }
    }
  };
  const aspectRef = useRef(() => aspect);
  aspectRef.current = () => aspect;
  const canvasRef = useCanvas(draw, aspectRef.current);
  return <canvas ref={canvasRef} className="chart-canvas" role="img" aria-label={label} />;
}
