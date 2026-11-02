import { useRef } from 'react';
import type { LiveData } from './useTrainer';
import { readPalette } from './theme';
import { useCanvas } from './useCanvas';

/**
 * Trace of the TD error over training. Each point is one batch of updates (a
 * single step in step mode). The line is the mean |TD error|; faint dots are
 * the mean signed TD error (positive = better than predicted, negative = worse).
 * With two agents, agent 0 is solid and agent 1 is dashed.
 */
export function TdChart({ live }: { live: React.MutableRefObject<LiveData[]> }) {
  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const P = readPalette();
    const all = live.current;
    ctx.fillStyle = P.paper;
    ctx.fillRect(0, 0, w, h);
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const padL = 38, padR = 10, padT = 10, padB = 22;
    const pw = w - padL - padR, ph = h - padT - padB;
    const longest = Math.max(0, ...all.map((d) => d.tdAbs.length));
    if (longest < 2) {
      ctx.fillStyle = P.faint;
      ctx.textAlign = 'center';
      ctx.fillText('TD error appears once training starts', w / 2, h / 2);
      return;
    }
    let lo = 0, hi = 1e-6;
    for (const d of all) {
      for (let i = 0; i < d.tdAbs.length; i++) {
        hi = Math.max(hi, d.tdAbs[i], Math.abs(d.tdSigned[i]));
        lo = Math.min(lo, d.tdSigned[i]);
      }
    }
    hi *= 1.08; lo *= 1.08;
    const X = (i: number, len: number) => padL + (len <= 1 ? 0 : (i / (len - 1)) * pw);
    const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * ph;

    ctx.strokeStyle = P.grid;
    ctx.fillStyle = P.faint;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    for (let k = 0; k <= 3; k++) {
      const v = lo + ((hi - lo) * k) / 3;
      ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke();
      ctx.fillText(v.toFixed(Math.abs(hi - lo) < 5 ? 2 : 0), padL - 6, Y(v) + 4);
    }
    ctx.strokeStyle = P.faint;
    ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(w - padR, Y(0)); ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText('recent updates', w - padR, h - 6);

    all.forEach((d, idx) => {
      const len = d.tdAbs.length;
      const color = idx === 0 ? P.accent : P.ink;
      if (len <= 150) {
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.3;
        for (let i = 0; i < len; i++) {
          ctx.beginPath(); ctx.arc(X(i, len), Y(d.tdSigned[i]), 2, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.setLineDash(idx === 0 ? [] : [6, 4]);
      ctx.beginPath();
      for (let i = 0; i < len; i++) {
        if (i === 0) ctx.moveTo(X(i, len), Y(d.tdAbs[i])); else ctx.lineTo(X(i, len), Y(d.tdAbs[i]));
      }
      ctx.stroke();
      ctx.setLineDash([]);
    });
  };
  const aspect = useRef(() => 0.4);
  const canvasRef = useCanvas(draw, aspect.current);
  return <canvas ref={canvasRef} className="chart-canvas" role="img" aria-label="Mean absolute TD error over recent updates" />;
}
