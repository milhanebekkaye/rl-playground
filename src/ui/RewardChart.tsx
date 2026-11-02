import { useRef } from 'react';
import type { LiveData } from './useTrainer';
import { readPalette } from './theme';
import { useCanvas } from './useCanvas';

const MAX_BINS = 320;

interface Props {
  live: React.MutableRefObject<LiveData[]>;
  index: number;
  /** Use one y-range across all agents so curves are directly comparable. */
  shareAxis: boolean;
}

export function RewardChart({ live, index, shareAxis }: Props) {
  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const P = readPalette();
    const all = live.current;
    const { rewards, episodeOffset } = all[index] ?? { rewards: [] as number[], episodeOffset: 0 };
    const n = rewards.length;
    ctx.fillStyle = P.paper;
    ctx.fillRect(0, 0, w, h);
    const padL = 38, padR = 10, padT = 10, padB = 22;
    const pw = w - padL - padR, ph = h - padT - padB;
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = P.faint;
    ctx.strokeStyle = P.grid;
    ctx.lineWidth = 1;

    if (n === 0) {
      ctx.textAlign = 'center';
      ctx.fillText('Press play to start training', w / 2, h / 2);
      return;
    }

    // Bin episodes so long runs stay readable; each bin plots its mean.
    const per = Math.max(1, Math.ceil(n / MAX_BINS));
    const bins: number[] = [];
    for (let i = 0; i < n; i += per) {
      let s = 0, c = 0;
      for (let j = i; j < Math.min(n, i + per); j++) { s += rewards[j]; c++; }
      bins.push(s / c);
    }
    let lo = Infinity, hi = -Infinity;
    for (const d of shareAxis ? all : [all[index]]) {
      for (const v of d.rewards) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
    const padY = (hi - lo) * 0.08;
    lo -= padY; hi += padY;
    const X = (i: number) => padL + (bins.length <= 1 ? 0 : (i / (bins.length - 1)) * pw);
    const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * ph;

    // Axes and y gridlines.
    ctx.textAlign = 'right';
    for (let k = 0; k <= 3; k++) {
      const v = lo + ((hi - lo) * k) / 3;
      const y = Y(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(v.toFixed(Math.abs(hi - lo) < 5 ? 1 : 0), padL - 6, y + 4);
    }
    if (lo < 0 && hi > 0) {
      ctx.strokeStyle = P.faint;
      ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(w - padR, Y(0)); ctx.stroke();
    }
    ctx.textAlign = 'left';
    ctx.fillStyle = P.faint;
    ctx.fillText(String(episodeOffset + 1), padL, h - 6);
    ctx.textAlign = 'right';
    ctx.fillText(`episode ${episodeOffset + n}`, w - padR, h - 6);

    // Raw episodes as dots when there are few; otherwise the binned mean only.
    if (per === 1) {
      ctx.fillStyle = P.accent;
      ctx.globalAlpha = 0.35;
      for (let i = 0; i < n; i++) {
        ctx.beginPath(); ctx.arc(X(i), Y(rewards[i]), 2, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    // Smoothed line (moving average over 10 bins).
    ctx.strokeStyle = per === 1 ? P.ink : P.accent;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    const win = per === 1 ? 10 : 3;
    let run = 0;
    for (let i = 0; i < bins.length; i++) {
      run += bins[i];
      if (i >= win) run -= bins[i - win];
      const v = run / Math.min(i + 1, win);
      if (i === 0) ctx.moveTo(X(i), Y(v)); else ctx.lineTo(X(i), Y(v));
    }
    ctx.stroke();
  };
  const aspect = useRef(() => 0.5);
  const canvasRef = useCanvas(draw, aspect.current);
  return <canvas ref={canvasRef} className="chart-canvas" role="img" aria-label="Reward per episode over training" />;
}
