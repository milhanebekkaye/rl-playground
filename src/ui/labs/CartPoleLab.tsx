import { useCallback, useEffect, useRef, useState } from 'react';
import { MAX_STEPS, X_LIMIT, type CartState } from '../../core/cartpole';
import { DEFAULT_DQN } from '../../core/dqn';
import type { CartFromWorker, CartToWorker } from '../../worker/cartpole.protocol';
import { LineChart, type Series } from '../LineChart';
import { readPalette } from '../theme';
import { useCanvas } from '../useCanvas';

const KEEP = 4000;
const AVG = 20;
/** Epsilon decays by this per step; the slider is expressed as "steps until it reaches the floor". */
const decayFor = (steps: number) => Math.pow(DEFAULT_DQN.epsEnd / DEFAULT_DQN.epsStart, 1 / steps);
const stepsFor = (decay: number) => Math.round(Math.log(DEFAULT_DQN.epsEnd / DEFAULT_DQN.epsStart) / Math.log(decay));

interface Live {
  state: CartState;
  q: number[];
  returns: number[];
  /** Greedy replay in progress. */
  replay: { states: CartState[]; q: number[][]; ret: number; startedAt: number } | null;
  lastReplay: number | null;
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; hint?: string; display: string }) {
  const { label, value, min, max, step, onChange, hint, display } = props;
  return (
    <label className="slider">
      <span className="slider-head"><span className="slider-name">{label}</span><output>{display}</output></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

function CartView({ live }: { live: React.MutableRefObject<Live> }) {
  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const P = readPalette();
    const L = live.current;
    let state = L.state;
    if (L.replay) {
      const i = Math.min(L.replay.states.length - 1, Math.floor(((performance.now() - L.replay.startedAt) / 1000) * 50));
      state = L.replay.states[i];
      if (i >= L.replay.states.length - 1 && performance.now() - L.replay.startedAt > L.replay.states.length * 20 + 800) {
        L.lastReplay = L.replay.ret;
        L.replay = null;
      }
    }
    ctx.fillStyle = P.paper;
    ctx.fillRect(0, 0, w, h);
    const margin = 24;
    const scale = (w - 2 * margin) / (2 * X_LIMIT);
    const groundY = h * 0.72;
    const cx = (x: number) => w / 2 + x * scale;
    // track and failure limits
    ctx.strokeStyle = P.ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(margin, groundY); ctx.lineTo(w - margin, groundY); ctx.stroke();
    ctx.strokeStyle = P.trap; ctx.lineWidth = 3;
    for (const x of [-X_LIMIT, X_LIMIT]) { ctx.beginPath(); ctx.moveTo(cx(x), groundY - 18); ctx.lineTo(cx(x), groundY + 8); ctx.stroke(); }
    // cart
    const cartW = Math.max(40, scale * 0.5), cartH = cartW * 0.5;
    const x = cx(state[0]);
    ctx.fillStyle = P.ink;
    ctx.fillRect(x - cartW / 2, groundY - cartH - 6, cartW, cartH);
    ctx.fillStyle = P.paper;
    ctx.fillRect(x - cartW / 2 + 3, groundY - cartH - 3, cartW - 6, 3);
    ctx.fillStyle = P.ink;
    for (const dx of [-cartW * 0.28, cartW * 0.28]) { ctx.beginPath(); ctx.arc(x + dx, groundY - 3, 6, 0, Math.PI * 2); ctx.fill(); }
    // pole (full length 1 m drawn at 2.2 * scale)
    const len = Math.min((groundY - 40) * 0.8, scale * 2.2);
    const px = x, py = groundY - cartH - 6;
    const tx = px + Math.sin(state[2]) * len, ty = py - Math.cos(state[2]) * len;
    ctx.strokeStyle = P.accent; ctx.lineWidth = 8; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tx, ty); ctx.stroke();
    ctx.fillStyle = P.paper; ctx.strokeStyle = P.ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = P.faint; ctx.textAlign = 'left';
    ctx.fillText(L.replay ? `greedy episode, step ${Math.min(L.replay.states.length - 1, Math.floor(((performance.now() - L.replay.startedAt) / 1000) * 50))}` : 'training (live)', margin, 18);
  };
  const aspectRef = useRef(() => 0.5);
  const canvasRef = useCanvas(draw, aspectRef.current);
  return <canvas ref={canvasRef} className="chart-canvas" role="img" aria-label="Animated cart and pole" />;
}

function QBars({ live }: { live: React.MutableRefObject<Live> }) {
  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const P = readPalette();
    const L = live.current;
    let q = L.q;
    if (L.replay) {
      const i = Math.min(L.replay.q.length - 1, Math.floor(((performance.now() - L.replay.startedAt) / 1000) * 50));
      q = L.replay.q[i];
    }
    ctx.fillStyle = P.paper; ctx.fillRect(0, 0, w, h);
    const lo = Math.min(0, ...q), hi = Math.max(1, ...q) * 1.1;
    const best = q[1] > q[0] ? 1 : 0;
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    const labels = ['push left', 'push right'];
    for (let a = 0; a < 2; a++) {
      const y = 12 + a * ((h - 24) / 2);
      const bh = (h - 24) / 2 - 10;
      const bw = Math.max(0, ((q[a] - lo) / (hi - lo)) * (w - 120));
      ctx.fillStyle = a === best ? P.accent : P.faint;
      ctx.globalAlpha = a === best ? 1 : 0.55;
      ctx.fillRect(100, y, bw, bh);
      ctx.globalAlpha = 1;
      ctx.fillStyle = P.ink; ctx.textAlign = 'right';
      ctx.fillText(labels[a], 92, y + bh / 2 + 4);
      ctx.textAlign = 'left';
      ctx.fillText(q[a].toFixed(2), 104 + bw, y + bh / 2 + 4);
    }
  };
  const aspectRef = useRef(() => 0.2);
  const canvasRef = useCanvas(draw, aspectRef.current);
  return <canvas ref={canvasRef} className="chart-canvas" role="img" aria-label="Q-values of the two actions" />;
}

export function CartPoleLab() {
  const workerRef = useRef<Worker | null>(null);
  const genRef = useRef(0);
  const live = useRef<Live>({ state: [0, 0, 0, 0], q: [0, 0], returns: [], replay: null, lastReplay: null });
  const [playing, setPlaying] = useState(false);
  const [nonce, setNonce] = useState(0);
  const [speed, setSpeed] = useState(1500);
  const [lr, setLr] = useState(DEFAULT_DQN.lr);
  const [decay, setDecay] = useState(DEFAULT_DQN.epsDecay);
  const [stats, setStats] = useState({ episodes: 0, steps: 0, epsilon: 1, buffer: 0, loss: 0 });
  const [, setBeat] = useState(0);
  // Light heartbeat so text that depends on live refs (greedy-episode result) refreshes while paused.
  useEffect(() => { const id = setInterval(() => setBeat((b) => b + 1), 500); return () => clearInterval(id); }, []);
  const cfgRef = useRef({ lr, epsDecay: decay });
  cfgRef.current = { lr, epsDecay: decay };
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const send = useCallback((m: CartToWorker) => workerRef.current?.postMessage(m), []);

  useEffect(() => {
    const worker = new Worker(new URL('../../worker/cartpole.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.onmessage = (e: MessageEvent<CartFromWorker>) => {
      const m = e.data;
      if (m.gen !== genRef.current) return;
      const L = live.current;
      if (m.type === 'greedy') {
        L.replay = { states: m.states, q: m.q, ret: m.ret, startedAt: performance.now() };
        return;
      }
      L.state = m.state;
      L.q = m.q;
      for (const r of m.finished) L.returns.push(r);
      if (L.returns.length > KEEP + 500) L.returns.splice(0, L.returns.length - KEEP);
      setStats({ episodes: m.episodes, steps: m.steps, epsilon: m.epsilon, buffer: m.bufferSize, loss: m.lastLoss });
    };
    return () => { worker.terminate(); workerRef.current = null; };
  }, []);

  useEffect(() => {
    genRef.current += 1;
    live.current = { state: [0, 0, 0, 0], q: [0, 0], returns: [], replay: null, lastReplay: null };
    setPlaying(false);
    setStats({ episodes: 0, steps: 0, epsilon: 1, buffer: 0, loss: 0 });
    send({ type: 'init', gen: genRef.current, seed: nonce + 1, cfg: cfgRef.current });
    send({ type: 'speed', stepsPerSecond: speedRef.current });
  }, [nonce, send]);

  useEffect(() => send({ type: 'config', cfg: { lr, epsDecay: decay } }), [lr, decay, send]);
  useEffect(() => send({ type: 'speed', stepsPerSecond: speed }), [speed, send]);

  const play = () => { setPlaying(true); send({ type: 'play' }); };
  const pause = () => { setPlaying(false); send({ type: 'pause' }); };
  const greedy = () => send({ type: 'greedy', gen: genRef.current, seed: Math.floor(Math.random() * 1e6) });

  const series = (): Series[] => {
    const r = live.current.returns;
    const avg: number[] = [];
    let run = 0;
    for (let i = 0; i < r.length; i++) {
      run += r[i];
      if (i >= AVG) run -= r[i - AVG];
      avg.push(run / Math.min(i + 1, AVG));
    }
    return [
      { name: 'return', values: r, color: 'accent', dots: true },
      { name: `moving average (${AVG})`, values: avg, color: 'ink' },
    ];
  };

  const r = live.current.returns;
  const recent = r.slice(-AVG);
  const avgRecent = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : null;
  const lrLog = Math.log10(lr);

  return (
    <div className="lab">
      <header className="lab-head">
        <p className="eyebrow">CartPole lab</p>
        <h1>From a table to a <em>network</em></h1>
        <p className="lede">
          Keep a pole upright on a cart by pushing left or right. The state is four continuous numbers, too many to put in a
          table, so a small neural network learns Q-values instead. It runs entirely in your browser, with the network and its
          backpropagation written by hand in TypeScript.
        </p>
      </header>
      <main className="layout lab-layout">
        <section className="card" aria-label="Cart and pole">
          <h2>Cart and pole</h2>
          <CartView live={live} />
          <h2 style={{ marginTop: 14 }}>Q-values</h2>
          <QBars live={live} />
          <p className="hint">The network's estimate of future reward for each push, in the state currently shown. The brighter bar is its greedy choice.</p>
        </section>

        <section className="card" aria-label="Controls">
          <div className="buttons">
            {playing ? <button className="btn primary" onClick={pause}>Pause</button> : <button className="btn primary" onClick={play}>Play</button>}
            <button className="btn" onClick={() => setNonce((n) => n + 1)}>Reset</button>
            <button className="btn" onClick={greedy} title="Play one episode with no exploration, using the current network">Run greedy episode</button>
          </div>
          {live.current.lastReplay !== null && <p className="hint" role="status">Last greedy episode: {live.current.lastReplay} of {MAX_STEPS} steps.</p>}
          <Slider label="Speed" value={Math.log10(speed)} min={1.7} max={4.3} step={0.05}
            onChange={(v) => setSpeed(Math.round(Math.pow(10, v)))} display={`${speed.toLocaleString()} steps/s`} />
          <Slider label="Learning rate" value={lrLog} min={-4} max={-2} step={0.05}
            onChange={(v) => setLr(Number(Math.pow(10, v).toPrecision(2)))} display={lr.toExponential(1)} hint="Adam step size." />
          <Slider label="Epsilon decay" value={stepsFor(decay)} min={1000} max={30000} step={500}
            onChange={(n) => setDecay(decayFor(n))} display={`${stepsFor(decay).toLocaleString()} steps`} hint="Steps for exploration to fall from 100% to 5% random moves." />
          <dl className="stats">
            <div><dt>Episode</dt><dd>{stats.episodes}</dd></div>
            <div><dt>Steps</dt><dd>{stats.steps.toLocaleString()}</dd></div>
            <div><dt>Avg return ({AVG})</dt><dd>{avgRecent === null ? '–' : avgRecent.toFixed(0)}</dd></div>
            <div><dt>Epsilon</dt><dd>{stats.epsilon.toFixed(2)}</dd></div>
            <div><dt>Replay buffer</dt><dd>{stats.buffer.toLocaleString()}</dd></div>
            <div><dt>Loss</dt><dd>{stats.loss.toFixed(3)}</dd></div>
          </dl>
        </section>

        <section className="card" aria-label="Learning curve" style={{ gridColumn: '1 / -1' }}>
          <h2>Return per episode</h2>
          <LineChart getSeries={series} xLabel={`episode ${stats.episodes}`} label="Return per episode with moving average" aspect={0.3} emptyText="Press play. A random policy scores about 20; the maximum is 500." />
          <p className="hint">Return is the number of steps the pole stayed up (capped at {MAX_STEPS}). Dots are single episodes, the line is a {AVG}-episode average.</p>
        </section>

        <section className="card why-card" aria-label="About this lab">
          <h2>From a table to a network</h2>
          <p>
            In the gridworld, Q-learning stores one number per (cell, action) in a table and nudges it toward
            <code> r + γ·max Q(s′, ·)</code>. CartPole has a continuous state (cart position and speed, pole angle and angular
            speed), so there is no table to fill in and nearby states must share what was learned. A neural network turns the state
            into two Q-values and the same target is used as a regression label.
          </p>
          <p>
            Two additions keep that stable. A <strong>replay buffer</strong> stores past transitions and trains on random
            minibatches, breaking the correlation between consecutive steps. A <strong>target network</strong>, a slowly updated copy
            of the online network, supplies the <code>max Q(s′, ·)</code> term so the label does not move with every update.
            The loss is Huber (squared for small errors, linear for large ones), optimised with Adam. Compare the tabular update in
            the Gridworld lab: the structure is the same, only the table became a function approximator.
          </p>
          <p>DQN is known to be sensitive to hyperparameters and seeds. Expect the return to rise, sometimes collapse, and recover; that instability is real, not a bug in the chart.</p>
        </section>
      </main>
    </div>
  );
}
