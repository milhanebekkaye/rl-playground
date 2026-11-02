import { useCallback, useMemo, useState } from 'react';
import { applyTool, resizeSpec, type Tool } from '../core/edit';
import { defaultSpec, emptySpec, type EnvSpec } from '../core/env';
import type { QParams } from '../core/qlearning';
import { GridCanvas } from './GridCanvas';
import { NumField } from './NumField';
import { RewardChart } from './RewardChart';
import { useTrainer } from './useTrainer';

const TOOLS: { id: Tool; label: string; key: string }[] = [
  { id: 'wall', label: 'Wall', key: 'wall' },
  { id: 'start', label: 'Start', key: 'start' },
  { id: 'goal', label: 'Goal', key: 'goal' },
  { id: 'trap', label: 'Trap', key: 'trap' },
  { id: 'erase', label: 'Erase', key: 'erase' },
];

const MAX_SPEED = 20000;
const toSpeed = (p: number) => Math.max(1, Math.round(Math.pow(MAX_SPEED, p)));
const fromSpeed = (s: number) => Math.log(s) / Math.log(MAX_SPEED);
const sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12];

function Slider(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; hint: string; display?: string }) {
  const { label, value, min, max, step, onChange, hint, display } = props;
  return (
    <label className="slider">
      <span className="slider-head">
        <span className="slider-name">{label}</span>
        <output>{display ?? value.toFixed(2)}</output>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
      <span className="hint">{hint}</span>
    </label>
  );
}

export function App() {
  const [spec, setSpec] = useState<EnvSpec>(defaultSpec);
  const [params, setParams] = useState<QParams>({ alpha: 0.3, gamma: 0.95, epsilon: 0.15 });
  const [speed, setSpeed] = useState(120);
  const [tool, setTool] = useState<Tool>('wall');
  const [resetNonce, setResetNonce] = useState(0);
  const [showNumbers, setShowNumbers] = useState(false);
  const [showTrail, setShowTrail] = useState(true);

  const { live, stats, playing, play, pause, step } = useTrainer(spec, params, speed, resetNonce);

  const onPaint = useCallback((x: number, y: number) => {
    setSpec((s) => applyTool(s, tool, y * s.width + x));
  }, [tool]);

  const patch = (p: Partial<EnvSpec>) => setSpec((s) => ({ ...s, ...p }));
  const setParam = (p: Partial<QParams>) => setParams((q) => ({ ...q, ...p }));
  const hasGoal = useMemo(() => spec.cells.includes('goal'), [spec]);

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <p className="eyebrow">Reinforcement learning, in your browser</p>
          <h1>RL <em>Playground</em></h1>
        </div>
        <p className="lede">
          Draw a small world. A Q-learning agent explores it by trial and error, and the board shows what it believes:
          colour is value, arrows are its current best move.
        </p>
      </header>

      <main className="layout">
        <section className="card world" aria-label="Environment">
          <div className="toolbar" role="radiogroup" aria-label="Drawing tool">
            {TOOLS.map((t) => (
              <button
                key={t.id}
                role="radio"
                aria-checked={tool === t.id}
                className={`tool tool-${t.key}${tool === t.id ? ' on' : ''}`}
                onClick={() => setTool(t.id)}
              >
                <i aria-hidden="true" />{t.label}
              </button>
            ))}
          </div>
          <div className="canvas-wrap">
            <GridCanvas spec={spec} live={live} showNumbers={showNumbers} showTrail={showTrail} onPaint={onPaint} />
          </div>
          <div className="legend">
            <span>negative value</span><span className="ramp" aria-hidden="true" /><span>positive value</span>
          </div>
          {!hasGoal && <p className="warn">There is no goal cell, so the agent has nothing to find. Draw one.</p>}
          <p className="hint">Click or drag on the grid to draw. Editing the world resets what the agent has learned.</p>
        </section>

        <section className="card controls" aria-label="Playback">
          <div className="buttons">
            {playing ? (
              <button className="btn primary" onClick={pause}>Pause</button>
            ) : (
              <button className="btn primary" onClick={play}>Play</button>
            )}
            <button className="btn" onClick={step}>Step</button>
            <button className="btn" onClick={() => setResetNonce((n) => n + 1)}>Reset</button>
          </div>
          <Slider
            label="Speed" value={fromSpeed(speed)} min={0} max={1} step={0.005}
            onChange={(p) => setSpeed(toSpeed(p))} display={`${speed.toLocaleString()} steps/s`}
            hint="Environment steps per second. Slow it down to watch single decisions."
          />
          <dl className="stats">
            <div><dt>Episode</dt><dd>{stats.episodes.toLocaleString()}</dd></div>
            <div><dt>Steps</dt><dd>{stats.totalSteps.toLocaleString()}</dd></div>
            <div><dt>Avg reward (50)</dt><dd>{stats.recentReward === null ? '–' : stats.recentReward.toFixed(2)}</dd></div>
            <div><dt>Goal rate (50)</dt><dd>{stats.recentSuccess === null ? '–' : `${Math.round(stats.recentSuccess * 100)}%`}</dd></div>
            <div><dt>Last TD error</dt><dd>{stats.lastTdError.toFixed(3)}</dd></div>
          </dl>
        </section>

        <section className="card chart" aria-label="Learning curve">
          <h2>Reward per episode</h2>
          <RewardChart live={live} />
          <p className="hint">Dots are single episodes; the line is a moving average. It rises as the policy improves.</p>
        </section>

        <section className="card params" aria-label="Learning parameters">
          <h2>Agent</h2>
          <Slider label="Learning rate  α" value={params.alpha} min={0.01} max={1} step={0.01}
            onChange={(alpha) => setParam({ alpha })} hint="How far each update moves Q toward its target." />
          <Slider label="Discount  γ" value={params.gamma} min={0} max={0.99} step={0.01}
            onChange={(gamma) => setParam({ gamma })} hint="How much future reward counts versus immediate reward." />
          <Slider label="Exploration  ε" value={params.epsilon} min={0} max={1} step={0.01}
            onChange={(epsilon) => setParam({ epsilon })} hint="Chance of a random move instead of the greedy one." />
          <div className="checks">
            <label><input type="checkbox" checked={showNumbers} onChange={(e) => setShowNumbers(e.target.checked)} /> Show values</label>
            <label><input type="checkbox" checked={showTrail} onChange={(e) => setShowTrail(e.target.checked)} /> Show trail</label>
          </div>
        </section>

        <section className="card env" aria-label="Environment settings">
          <h2>World</h2>
          <div className="fields">
            <NumField label="Goal reward" value={spec.goalReward} onChange={(v) => patch({ goalReward: v })} />
            <NumField label="Trap reward" value={spec.trapReward} onChange={(v) => patch({ trapReward: v })} />
            <NumField label="Step reward" value={spec.stepReward} step={0.01} onChange={(v) => patch({ stepReward: v })} />
            <NumField label="Max steps" value={spec.maxSteps} step={10} min={1} max={5000} onChange={(v) => patch({ maxSteps: Math.round(v) })} />
          </div>
          <div className="fields">
            <label className="field">
              <span>Width</span>
              <select value={spec.width} onChange={(e) => setSpec((s) => resizeSpec(s, +e.target.value, s.height))}>
                {sizes.map((n) => <option key={n}>{n}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Height</span>
              <select value={spec.height} onChange={(e) => setSpec((s) => resizeSpec(s, s.width, +e.target.value))}>
                {sizes.map((n) => <option key={n}>{n}</option>)}
              </select>
            </label>
          </div>
          <div className="buttons">
            <button className="btn" onClick={() => setSpec(defaultSpec())}>Example maze</button>
            <button className="btn" onClick={() => setSpec(emptySpec(spec.width, spec.height))}>Clear grid</button>
          </div>
        </section>
      </main>

      <footer className="foot">
        MIT licensed. Training runs in a Web Worker in your browser; nothing leaves the page.
      </footer>
    </div>
  );
}
