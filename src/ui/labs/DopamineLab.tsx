import { useCallback, useEffect, useRef, useState } from 'react';
import { DEFAULT_DOPAMINE, DopamineModel, normaliseParams, rewardTime, type DopamineParams } from '../../core/dopamine';
import { mulberry32 } from '../../core/rng';
import { LineChart, type Series } from '../LineChart';
import { readPalette, paperRgb, tint } from '../theme';
import { useCanvas } from '../useCanvas';

const MAX_ROWS = 150;
const MAX_POINTS = 400;
const OMIT_EVERY = 25;

function bin(values: number[], per: number): number[] {
  if (per <= 1) return values;
  const out: number[] = [];
  for (let i = 0; i < values.length; i += per) {
    let s = 0, c = 0;
    for (let j = i; j < Math.min(values.length, i + per); j++) { s += values[j]; c++; }
    out.push(s / c);
  }
  return out;
}

function Heatmap({ model }: { model: React.MutableRefObject<DopamineModel> }) {
  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const P = readPalette();
    const paper = paperRgb();
    const m = model.current;
    const { cue, length } = m.params;
    const rt = rewardTime(m.params);
    ctx.fillStyle = P.paper;
    ctx.fillRect(0, 0, w, h);
    const padL = 44, padT = 26, padR = 8, padB = 8;
    const gw = w - padL - padR, gh = h - padT - padB;
    const cw = gw / length;
    const n = m.trials;
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = P.accent;
    ctx.fillText('cue', padL + (cue + 0.5) * cw, padT - 12);
    ctx.fillStyle = P.goal;
    ctx.fillText('reward', padL + (rt + 0.5) * cw, padT - 12);
    if (n === 0) {
      ctx.fillStyle = P.faint;
      ctx.fillText('Press play: each row is a trial, each column a time step', padL + gw / 2, padT + gh / 2);
    }
    const per = Math.max(1, Math.ceil(n / MAX_ROWS));
    const rows = Math.ceil(n / per);
    const rh = gh / Math.max(rows, MAX_ROWS);
    for (let r = 0; r < rows; r++) {
      const start = r * per, end = Math.min(n, start + per);
      let omitted = false;
      for (let t = 0; t < length; t++) {
        let s = 0;
        for (let k = start; k < end; k++) s += m.history[k].deltas[t];
        const v = Math.max(-1, Math.min(1, s / (end - start)));
        ctx.fillStyle = v >= 0 ? tint(P.pos, paper, Math.min(1, Math.abs(v))) : tint(P.neg, paper, Math.min(1, Math.abs(v)));
        ctx.fillRect(padL + t * cw, padT + r * rh, cw + 0.5, rh + 0.5);
      }
      for (let k = start; k < end; k++) if (m.history[k].omitted) omitted = true;
      if (omitted) {
        ctx.fillStyle = P.trap;
        ctx.fillRect(padL - 8, padT + r * rh, 5, Math.max(2, rh));
      }
    }
    // guides
    ctx.strokeStyle = P.grid;
    ctx.lineWidth = 1;
    for (const t of [cue, rt]) {
      ctx.strokeRect(padL + t * cw + 0.5, padT + 0.5, cw, gh);
    }
    ctx.fillStyle = P.faint;
    ctx.textAlign = 'right';
    ctx.fillText('1', padL - 12, padT + 10);
    ctx.fillText(String(n || ''), padL - 12, padT + Math.min(gh, Math.max(rows, 1) * rh));
    ctx.save(); ctx.translate(11, padT + gh / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('trial', 0, 0); ctx.restore();
  };
  const aspectRef = useRef(() => 0.75);
  const canvasRef = useCanvas(draw, aspectRef.current);
  return <canvas ref={canvasRef} className="chart-canvas" role="img" aria-label="Heatmap of the TD error: trials down, time steps across" />;
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; hint?: string; display?: string }) {
  const { label, value, min, max, step, onChange, hint, display } = props;
  return (
    <label className="slider">
      <span className="slider-head"><span className="slider-name">{label}</span><output>{display ?? value.toFixed(2)}</output></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

export function DopamineLab() {
  const [params, setParams] = useState<DopamineParams>(DEFAULT_DOPAMINE);
  const [speed, setSpeed] = useState(20);
  const [playing, setPlaying] = useState(false);
  const [omitEvery, setOmitEvery] = useState(false);
  const [, setTick] = useState(0);
  const model = useRef(new DopamineModel(DEFAULT_DOPAMINE, mulberry32(7)));
  const omitNext = useRef(false);
  const omitEveryRef = useRef(omitEvery);
  omitEveryRef.current = omitEvery;
  const budget = useRef(0);
  const cache = useRef<{ n: number; per: number; cue: number[]; rew: number[]; marks: number[] }>({ n: -1, per: 1, cue: [], rew: [], marks: [] });

  const runTrials = useCallback((count: number) => {
    const m = model.current;
    for (let i = 0; i < count; i++) {
      const next = m.trials + 1;
      const omit = omitNext.current || (omitEveryRef.current && next % OMIT_EVERY === 0);
      omitNext.current = false;
      m.runTrial(omit);
    }
    setTick((t) => t + 1);
  }, []);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const id = setInterval(() => {
      const now = performance.now();
      budget.current += ((now - last) / 1000) * speed;
      last = now;
      const n = Math.floor(budget.current);
      if (n > 0) { budget.current -= n; runTrials(Math.min(n, 200)); }
    }, 33);
    return () => clearInterval(id);
  }, [playing, speed, runTrials]);

  const reset = (p: DopamineParams = params) => {
    const norm = normaliseParams(p);
    model.current = new DopamineModel(norm, mulberry32(7));
    cache.current.n = -1;
    omitNext.current = false;
    setParams(norm);
    setTick((t) => t + 1);
  };

  const update = (p: Partial<DopamineParams>) => {
    const next = { ...params, ...p };
    if ('cue' in p || 'delay' in p) {
      reset(next);
    } else {
      setParams(next);
      model.current.setLearningParams({ alpha: next.alpha, gamma: next.gamma, prob: next.prob });
    }
  };

  const m = model.current;
  const rt = rewardTime(m.params);
  const last = m.history[m.trials - 1];

  const series = (): Series[] => {
    const mm = model.current;
    const c = cache.current;
    if (c.n !== mm.trials) {
      const per = Math.max(1, Math.ceil(mm.trials / MAX_POINTS));
      const cue = mm.history.map((h) => h.deltas[mm.params.cue]);
      const rew = mm.history.map((h) => h.deltas[rewardTime(mm.params)]);
      c.cue = bin(cue, per);
      c.rew = bin(rew, per);
      c.marks = [];
      mm.history.forEach((h, i) => { if (h.omitted) c.marks.push(Math.floor(i / per)); });
      c.n = mm.trials; c.per = per;
    }
    return [
      { name: 'delta at cue', values: c.cue, color: 'accent' },
      { name: 'delta at reward', values: c.rew, color: 'goal', dashed: true },
    ];
  };

  const valueSeries = (): Series[] => [{ name: 'V(t)', values: model.current.V, color: 'ink' }];

  return (
    <div className="lab">
      <header className="lab-head">
        <p className="eyebrow">Dopamine lab</p>
        <h1>Prediction error, <em>step by step</em></h1>
        <p className="lede">
          A cue, then a reward a few steps later, repeated. A learner with a clock-like state representation updates
          its predictions with TD(0). Watch the surprise (the TD error) move from the reward to the cue.
        </p>
      </header>
      <main className="layout lab-layout">
        <section className="card" aria-label="TD error heatmap">
          <h2>TD error by trial and time step</h2>
          <Heatmap model={model} />
          <div className="legend">
            <span>negative (worse than predicted)</span><span className="ramp" aria-hidden="true" /><span>positive (better)</span>
          </div>
          <p className="hint">Rows are trials, top to bottom. Red ticks on the left mark omission trials. After many trials the reward column fades and the cue column lights up.</p>
        </section>

        <section className="card" aria-label="Controls">
          <div className="buttons">
            <button className="btn primary" onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
            <button className="btn" onClick={() => { setPlaying(false); runTrials(1); }}>Step trial</button>
            <button className="btn" onClick={() => { setPlaying(false); reset(); }}>Reset</button>
          </div>
          <div className="buttons" style={{ marginTop: 10 }}>
            <button className="btn" onClick={() => { omitNext.current = true; if (!playing) runTrials(1); }} title="Run the next trial with the reward withheld">Omit next reward</button>
          </div>
          <div className="checks">
            <label><input type="checkbox" checked={omitEvery} onChange={(e) => setOmitEvery(e.target.checked)} /> Omit the reward on every {OMIT_EVERY}th trial</label>
          </div>
          <Slider label="Speed" value={Math.log(speed) / Math.log(600)} min={0} max={1} step={0.01}
            onChange={(p) => setSpeed(Math.max(1, Math.round(Math.pow(600, p))))} display={`${speed} trials/s`} />
          <dl className="stats">
            <div><dt>Trial</dt><dd>{m.trials}</dd></div>
            <div><dt>δ at cue</dt><dd>{last ? last.deltas[m.params.cue].toFixed(2) : '–'}</dd></div>
            <div><dt>δ at reward</dt><dd>{last ? last.deltas[rt].toFixed(2) : '–'}{last?.omitted ? ' (omitted)' : ''}</dd></div>
          </dl>
        </section>

        <section className="card" aria-label="Delta over trials">
          <h2>δ at the cue and at the reward</h2>
          <LineChart getSeries={series} getMarks={() => cache.current.marks} xLabel={`trial ${m.trials}`} label="TD error at cue time and reward time over trials" emptyText="Press play to start" />
          <p className="hint"><span className="chip chip-0">solid</span> δ at cue time &nbsp; <span className="chip chip-1">dashed</span> δ at reward time. A dip below zero at the reward line is an omitted reward. Long runs are averaged into bins.</p>
        </section>

        <section className="card" aria-label="Task parameters">
          <h2>Task and learner</h2>
          <Slider label="Learning rate  α" value={params.alpha} min={0.02} max={1} step={0.01} onChange={(alpha) => update({ alpha })} />
          <Slider label="Discount  γ" value={params.gamma} min={0.5} max={1} step={0.01} onChange={(gamma) => update({ gamma })} hint="Lower γ shrinks the cue response: far-off reward is worth less." />
          <Slider label="Cue time" value={params.cue} min={1} max={10} step={1} display={`step ${params.cue}`} onChange={(cue) => update({ cue })} hint="Changing the task restarts learning." />
          <Slider label="Reward delay" value={params.delay} min={1} max={12} step={1} display={`${params.delay} steps`} onChange={(delay) => update({ delay })} />
          <Slider label="Reward probability" value={params.prob} min={0} max={1} step={0.05} onChange={(prob) => update({ prob })} hint="Below 1, reward stays partly surprising: δ at reward never fully vanishes." />
        </section>

        <section className="card" aria-label="Learned values">
          <h2>Learned value V(t)</h2>
          <LineChart getSeries={valueSeries} xLabel="time step" xStart="0" label="Learned value of each time step" aspect={0.4} emptyText="Press play" />
          <p className="hint">Value ramps up toward the reward and is flat before the cue, where the cue's arrival cannot be predicted.</p>
        </section>

        <section className="card why-card" aria-label="About this lab">
          <h2>What is this showing?</h2>
          <p>
            Each time step has a learned value V(t): the reward the learner expects from there on. Moving from one step to the next
            gives a prediction error, δ = r + γ·V(next) − V(current), and the value of the step you just left is nudged by α·δ.
            Early on nothing is predicted, so the reward itself is the surprise. As values propagate backward from the reward, the
            reward stops being surprising and the first thing that <em>is</em> unpredictable, the cue, takes over the error. If the
            expected reward is withheld, V says "reward" but none arrives, and δ goes negative at exactly that moment.
          </p>
          <p>
            This pattern is inspired by the classic reward-prediction-error account of dopamine neurons (Schultz, Dayan and
            Montague, 1997, <em>A neural substrate of prediction and reward</em>, Science). Treat it as an analogy: a minimal tabular
            TD(0) learner with a clock-like state representation, not a model of the brain.
          </p>
        </section>
      </main>
    </div>
  );
}
