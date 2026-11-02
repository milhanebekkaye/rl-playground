import { useRef } from 'react';
import { NUM_ACTIONS, type EnvSpec } from '../core/env';
import type { LiveData } from './useTrainer';
import { paperRgb, readPalette, tint } from './theme';
import { useCanvas } from './useCanvas';

interface Props {
  spec: EnvSpec;
  live: React.MutableRefObject<LiveData>;
  showNumbers: boolean;
  showTrail: boolean;
  onPaint: (x: number, y: number) => void;
}

// Unit vectors for up, right, down, left.
const DIRS: ReadonlyArray<readonly [number, number]> = [[0, -1], [1, 0], [0, 1], [-1, 0]];

function drawArrow(ctx: CanvasRenderingContext2D, cx: number, cy: number, dir: readonly [number, number], len: number) {
  const [dx, dy] = dir;
  const tipX = cx + dx * len, tipY = cy + dy * len;
  const tailX = cx - dx * len, tailY = cy - dy * len;
  const head = len * 0.75;
  ctx.beginPath();
  ctx.moveTo(tailX, tailY);
  ctx.lineTo(tipX, tipY);
  ctx.moveTo(tipX - dx * head - dy * head, tipY - dy * head + dx * head);
  ctx.lineTo(tipX, tipY);
  ctx.lineTo(tipX - dx * head + dy * head, tipY - dy * head - dx * head);
  ctx.stroke();
}

export function GridCanvas({ spec, live, showNumbers, showTrail, onPaint }: Props) {
  const specRef = useRef(spec);
  specRef.current = spec;
  const opts = useRef({ showNumbers, showTrail });
  opts.current = { showNumbers, showTrail };
  const painting = useRef(false);

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const { width, height, cells } = specRef.current;
    const { q, state, trail } = live.current;
    const P = readPalette();
    const paper = paperRgb();
    const cs = w / width;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = P.paper;
    ctx.fillRect(0, 0, w, h);

    // Normalise the heatmap by the largest |max Q| among ordinary cells.
    let scale = 1e-6;
    for (let s = 0; s < cells.length; s++) {
      if (cells[s] === 'wall' || cells[s] === 'goal' || cells[s] === 'trap') continue;
      let m = -Infinity;
      for (let a = 0; a < NUM_ACTIONS; a++) m = Math.max(m, q[s * NUM_ACTIONS + a]);
      scale = Math.max(scale, Math.abs(m));
    }

    ctx.font = `600 ${Math.max(9, cs * 0.17)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textBaseline = 'alphabetic';
    for (let s = 0; s < cells.length; s++) {
      const x = (s % width) * cs, y = Math.floor(s / width) * cs;
      const cell = cells[s];
      if (cell === 'wall') {
        ctx.fillStyle = P.wall;
        ctx.fillRect(x, y, cs, cs);
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, cs, cs);
        ctx.clip();
        ctx.strokeStyle = P.paper;
        ctx.globalAlpha = 0.18;
        ctx.lineWidth = 1;
        for (let k = -cs; k < cs * 2; k += cs / 4) {
          ctx.beginPath();
          ctx.moveTo(x + k, y + cs);
          ctx.lineTo(x + k + cs, y);
          ctx.stroke();
        }
        ctx.restore();
        continue;
      }
      let best = -Infinity, bestA = 0, anyNonZero = false;
      for (let a = 0; a < NUM_ACTIONS; a++) {
        const v = q[s * NUM_ACTIONS + a];
        if (v !== 0) anyNonZero = true;
        if (v > best + 1e-9) { best = v; bestA = a; }
      }
      const terminal = cell === 'goal' || cell === 'trap';
      if (!terminal && anyNonZero) {
        const t = Math.min(1, Math.abs(best) / scale);
        ctx.fillStyle = tint(best >= 0 ? P.pos : P.neg, paper, 0.08 + 0.8 * t);
        ctx.fillRect(x, y, cs, cs);
      }
      if (terminal) {
        ctx.fillStyle = cell === 'goal' ? P.goal : P.trap;
        ctx.globalAlpha = 0.9;
        ctx.fillRect(x, y, cs, cs);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = P.paper;
        ctx.lineWidth = Math.max(2, cs * 0.06);
        ctx.lineCap = 'round';
        const c = cs / 2, r = cs * 0.2;
        ctx.beginPath();
        if (cell === 'goal') {
          ctx.arc(x + c, y + c, r * 1.1, 0, Math.PI * 2);
          ctx.moveTo(x + c + r * 0.3, y + c);
          ctx.arc(x + c, y + c, r * 0.3, 0, Math.PI * 2);
        } else {
          ctx.moveTo(x + c - r, y + c - r); ctx.lineTo(x + c + r, y + c + r);
          ctx.moveTo(x + c + r, y + c - r); ctx.lineTo(x + c - r, y + c + r);
        }
        ctx.stroke();
      } else if (anyNonZero) {
        ctx.strokeStyle = P.ink;
        ctx.globalAlpha = 0.8;
        ctx.lineWidth = Math.max(1.5, cs * 0.045);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        drawArrow(ctx, x + cs / 2, y + cs / 2 - (opts.current.showNumbers ? cs * 0.06 : 0), DIRS[bestA], cs * 0.2);
        ctx.globalAlpha = 1;
        if (opts.current.showNumbers) {
          ctx.fillStyle = P.ink;
          ctx.textAlign = 'center';
          ctx.fillText(best.toFixed(2), x + cs / 2, y + cs - cs * 0.08);
        }
      }
      if (cell === 'start') {
        ctx.strokeStyle = P.accent;
        ctx.lineWidth = Math.max(2, cs * 0.05);
        ctx.setLineDash([cs * 0.1, cs * 0.07]);
        ctx.strokeRect(x + cs * 0.07, y + cs * 0.07, cs * 0.86, cs * 0.86);
        ctx.setLineDash([]);
        ctx.fillStyle = P.accent;
        ctx.textAlign = 'left';
        ctx.fillText('S', x + cs * 0.14, y + cs * 0.3);
      }
    }

    // Grid lines.
    ctx.strokeStyle = P.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= width; i++) { ctx.moveTo(Math.round(i * cs) + 0.5, 0); ctx.lineTo(Math.round(i * cs) + 0.5, h); }
    for (let j = 0; j <= height; j++) { ctx.moveTo(0, Math.round(j * cs) + 0.5); ctx.lineTo(w, Math.round(j * cs) + 0.5); }
    ctx.stroke();

    // Current-episode trail, older segments fainter, then the agent.
    const center = (s: number): [number, number] => [(s % width + 0.5) * cs, (Math.floor(s / width) + 0.5) * cs];
    if (opts.current.showTrail && trail.length > 1) {
      ctx.strokeStyle = P.trail;
      ctx.lineWidth = Math.max(2, cs * 0.07);
      ctx.lineCap = 'round';
      const from = Math.max(1, trail.length - 80);
      for (let i = from; i < trail.length; i++) {
        const [ax, ay] = center(trail[i - 1]);
        const [bx, by] = center(trail[i]);
        ctx.globalAlpha = 0.25 + 0.65 * ((i - from + 1) / (trail.length - from));
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    const [ax, ay] = center(state);
    ctx.fillStyle = P.accent;
    ctx.strokeStyle = P.paper;
    ctx.lineWidth = Math.max(2, cs * 0.05);
    ctx.beginPath();
    ctx.arc(ax, ay, cs * 0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  };

  const canvasRef = useCanvas(draw, () => specRef.current.height / specRef.current.width);

  const cellAt = (e: React.PointerEvent<HTMLCanvasElement>): [number, number] | null => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = Math.floor(((e.clientX - r.left) / r.width) * spec.width);
    const y = Math.floor(((e.clientY - r.top) / r.height) * spec.height);
    return x >= 0 && y >= 0 && x < spec.width && y < spec.height ? [x, y] : null;
  };

  return (
    <canvas
      ref={canvasRef}
      className="grid-canvas"
      role="img"
      aria-label="Grid world. Colour shows the learned value of each cell, arrows show the greedy action. Click or drag to edit."
      onPointerDown={(e) => {
        painting.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        const c = cellAt(e);
        if (c) onPaint(c[0], c[1]);
      }}
      onPointerMove={(e) => {
        if (!painting.current) return;
        const c = cellAt(e);
        if (c) onPaint(c[0], c[1]);
      }}
      onPointerUp={() => { painting.current = false; }}
      onPointerCancel={() => { painting.current = false; }}
    />
  );
}
