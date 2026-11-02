import { useCallback, useEffect, useMemo, useState } from 'react';
import { applyTool, resizeSpec, type Tool } from '../core/edit';
import { cliffSpec, defaultSpec, emptySpec, type EnvSpec } from '../core/env';
import type { QParams } from '../core/agent';
import { decodeShare, encodeShare } from '../core/share';
import type { Algo } from '../core/trainer';
import { GridCanvas } from './GridCanvas';
import { NumField } from './NumField';
import { RewardChart } from './RewardChart';
import { TdChart } from './TdChart';
import { useTrainer } from './useTrainer';

const TOOLS: { id: Tool; label: string; key: string }[] = [
  { id: 'wall', label: 'Wall', key: 'wall' },
  { id: 'start', label: 'Start', key: 'start' },
  { id: 'goal', label: 'Goal', key: 'goal' },
  { id: 'trap', label: 'Trap', key: 'trap' },
  { id: 'erase', label: 'Erase', key: 'erase' },
];

const ALGO_NAME: Record<Algo, string> = { qlearning: 'Q-learning', sarsa: 'SARSA' };
const DEFAULT_PARAMS: QParams = { alpha: 0.3, gamma: 0.95, epsilon: 0.15 };
const CLIFF_PARAMS: QParams = { alpha: 0.2, gamma: 0.99, epsilon: 0.1 };

function initialFromHash() {
  const hash = typeof location === 'undefined' ? '' : location.hash;
  const decoded = hash.length > 1 ? decodeShare(hash) : null;
  return {
    state: decoded ?? { spec: defaultSpec(), params: DEFAULT_PARAMS, compare: false, algo: 'qlearning' as Algo },
    invalid: hash.length > 1 && decoded === null,
  };
}

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
  const [initial] = useState(initialFromHash);
  const [spec, setSpec] = useState<EnvSpec>(initial.state.spec);
  const [params, setParams] = useState<QParams>(initial.state.params);
  const [compare, setCompare] = useState(initial.state.compare);
  const [algo, setAlgo] = useState<Algo>(initial.state.algo);
  const [view, setView] = useState<'value' | 'td'>('value');
  const [notice, setNotice] = useState<string | null>(
    initial.invalid ? 'That link was not a valid maze, so the example maze was loaded instead.' : null,
  );
  const [link, setLink] = useState<{ url: string; copied: boolean } | null>(null);
  const [speed, setSpeed] = useState(120);
  const [tool, setTool] = useState<Tool>('wall');
  const [resetNonce, setResetNonce] = useState(0);
  const [showNumbers, setShowNumbers] = useState(false);
  const [showTrail, setShowTrail] = useState(true);

  const algos = useMemo<Algo[]>(() => (compare ? ['qlearning', 'sarsa'] : [algo]), [compare, algo]);
  const { live, stats, playing, play, pause, step } = useTrainer(spec, params, speed, resetNonce, algos);

  // Load a maze when the hash changes while the page is open (e.g. pasting a link in the same tab).
  useEffect(() => {
    const onHash = () => {
      const d = decodeShare(location.hash);
      if (!d) return;
      setSpec(d.spec); setParams(d.params); setCompare(d.compare); setAlgo(d.algo);
      setNotice(null); setLink(null);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const copyLink = async () => {
    const hash = encodeShare({ spec, params, compare, algo });
    const url = `${location.origin}${location.pathname}${location.search}#${hash}`;
    try { history.replaceState(null, '', `#${hash}`); } catch { /* sandboxed frames may forbid this */ }
    let copied = false;
    try { await navigator.clipboard.writeText(url); copied = true; } catch { /* fall back to the visible field */ }
    setLink({ url, copied });
  };

  const onPaint = useCallback((x: number, y: number) => {
    setSpec((s) => applyTool(s, tool, y * s.width + x));
  }, [tool]);

  const patch = (p: Partial<EnvSpec>) => setSpec((s) => ({ ...s, ...p }));
  const setParam = (p: Partial<QParams>) => setParams((q) => ({ ...q, ...p }));
  const hasGoal = useMemo(() => spec.cells.includes('goal'), [spec]);

  const statBlock = (st: (typeof stats)[number] | undefined) => {
    const t = st ?? { episodes: 0, totalSteps: 0, lastTdError: 0, recentReward: null, recentSuccess: null };
    return [
      ['Episode', t.episodes.toLocaleString()],
      ['Steps', t.totalSteps.toLocaleString()],
      ['Avg reward (50)', t.recentReward === null ? '–' : t.recentReward.toFixed(2)],
      ['Goal rate (50)', t.recentSuccess === null ? '–' : `${Math.round(t.recentSuccess * 100)}%`],
      ['Last TD error', t.lastTdError.toFixed(3)],
    ] as const;
  };

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <p className="eyebrow">Reinforcement learning, in your browser</p>
          <h1>RL <em>Playground</em></h1>
        </div>
        <p className="lede">
          Draw a small world. A Q-learning or SARSA agent explores it by trial and error, and the board shows what it
          believes: colour is value, arrows are its current best move. Compare the two on the same maze, or switch to
          the TD-error view to see where it is still surprised.
        </p>
      </header>

      {notice && <p className="warn" role="status">{notice}</p>}

      <main className={`layout${compare ? ' compare' : ''}`}>
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
            <div className="seg right" role="radiogroup" aria-label="Board colouring">
              <button role="radio" aria-checked={view === 'value'} className={view === 'value' ? 'on' : ''} onClick={() => setView('value')}>Value</button>
              <button role="radio" aria-checked={view === 'td'} className={view === 'td' ? 'on' : ''} onClick={() => setView('td')}>TD error</button>
            </div>
          </div>
          <div className="boards">
            {algos.map((a, i) => (
              <figure className="board" key={a}>
                {compare && <figcaption><span className={`chip chip-${i}`}>{ALGO_NAME[a]}</span></figcaption>}
                <div className="canvas-wrap">
                  <GridCanvas spec={spec} live={live} index={i} view={view} showNumbers={showNumbers} showTrail={showTrail} onPaint={onPaint} />
                </div>
              </figure>
            ))}
          </div>
          <div className="legend">
            {view === 'value' ? (
              <>
                <span>negative value</span><span className="ramp" aria-hidden="true" /><span>positive value</span>
              </>
            ) : (
              <>
                <span>small TD error</span><span className="ramp ramp-td" aria-hidden="true" /><span>large TD error</span>
              </>
            )}
          </div>
          {!hasGoal && <p className="warn">There is no goal cell, so the agent has nothing to find. Draw one.</p>}
          <p className="hint">
            {view === 'value'
              ? 'Click or drag on the grid to draw. Editing the world resets what the agents have learned.'
              : 'Colour is the running mean of |TD error| per cell: where predictions are still being corrected. Arrows still show the greedy policy.'}
          </p>
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
          <div className="modes">
            <div className="seg" role="radiogroup" aria-label="Mode">
              <button role="radio" aria-checked={!compare} className={!compare ? 'on' : ''} onClick={() => setCompare(false)}>Single agent</button>
              <button role="radio" aria-checked={compare} className={compare ? 'on' : ''} onClick={() => setCompare(true)}>Compare</button>
            </div>
            {!compare && (
              <label className="field inline">
                <span>Algorithm</span>
                <select value={algo} onChange={(e) => setAlgo(e.target.value as Algo)}>
                  <option value="qlearning">Q-learning</option>
                  <option value="sarsa">SARSA</option>
                </select>
              </label>
            )}
          </div>
          {compare && <p className="hint">Both agents get the same maze, hyperparameters, seed and number of steps.</p>}
          <Slider
            label="Speed" value={fromSpeed(speed)} min={0} max={1} step={0.005}
            onChange={(p) => setSpeed(toSpeed(p))} display={`${speed.toLocaleString()} steps/s`}
            hint="Environment steps per second. Slow it down to watch single decisions."
          />
          {algos.map((a, i) => (
            <div key={a}>
              {compare && <p className="stats-title"><span className={`chip chip-${i}`}>{ALGO_NAME[a]}</span></p>}
              <dl className="stats">
                {statBlock(stats[i]).map(([k, v]) => (
                  <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
                ))}
              </dl>
            </div>
          ))}
        </section>

        <section className="card chart" aria-label="Learning curve">
          <h2>Reward per episode</h2>
          <div className="charts">
            {algos.map((a, i) => (
              <div key={a}>
                {compare && <p className="stats-title"><span className={`chip chip-${i}`}>{ALGO_NAME[a]}</span></p>}
                <RewardChart live={live} index={i} shareAxis={compare} />
              </div>
            ))}
          </div>
          <p className="hint">
            Dots are single episodes; the line is a moving average. It rises as the policy improves.
            {compare && ' Both charts share one vertical scale.'}
          </p>
        </section>

        <section className="card td" aria-label="TD error">
          <h2>TD error</h2>
          <TdChart live={live} />
          <p className="hint">
            {compare ? 'Solid: Q-learning. Dashed: SARSA. ' : ''}
            Line: mean |TD error| per batch of updates. Dots: mean signed TD error (above zero means the outcome was better than predicted).
          </p>
          <details className="why">
            <summary>What is the TD error, and why does it matter?</summary>
            <p>
              After each step the agent compares its prediction, Q(s, a), with a better-informed estimate: the reward it
              just received plus what it now expects from the next state. The gap is the temporal-difference (TD) error,
              and the update shrinks it. Early on it is large and jumps around; as the Q-values become consistent it
              settles near zero, except where the world is genuinely surprising, such as near traps and the goal.
            </p>
            <p>
              This signal parallels the reward prediction error described in neuroscience, where the activity of
              dopamine neurons has been compared to a TD error (Schultz, Dayan and Montague, 1997). Treat it as an
              analogy: the same kind of surprise-driven correction, not a model of how the brain actually works.
            </p>
          </details>
        </section>

        <section className="card params" aria-label="Learning parameters">
          <h2>Agent</h2>
          <Slider label="Learning rate  α" value={params.alpha} min={0.01} max={1} step={0.01}
            onChange={(alpha) => setParam({ alpha })} hint="How far each update moves Q toward its target." />
          <Slider label="Discount  γ" value={params.gamma} min={0} max={1} step={0.01}
            onChange={(gamma) => setParam({ gamma })} hint="How much future reward counts versus immediate reward." />
          <Slider label="Exploration  ε" value={params.epsilon} min={0} max={1} step={0.01}
            onChange={(epsilon) => setParam({ epsilon })} hint="Chance of a random move instead of the greedy one. SARSA learns around it; Q-learning ignores it when valuing the next state." />
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
            <button
              className="btn"
              title="Classic cliff walk: Q-learning and SARSA learn different paths"
              onClick={() => { setSpec(cliffSpec()); setParams(CLIFF_PARAMS); setCompare(true); }}
            >Cliff walk</button>
            <button className="btn" onClick={() => setSpec(emptySpec(spec.width, spec.height))}>Clear grid</button>
          </div>
          <div className="share">
            <button className="btn" onClick={copyLink}>Copy link</button>
            {link && (
              <>
                <input readOnly value={link.url} aria-label="Shareable link" onFocus={(e) => e.currentTarget.select()} />
                <span className="hint" role="status">{link.copied ? 'Link copied.' : 'Copy the link above to share this maze.'}</span>
              </>
            )}
            {!link && <span className="hint">Encodes the maze, rewards, hyperparameters and mode in the URL.</span>}
          </div>
        </section>
      </main>

      <footer className="foot">
        MIT licensed. Training runs in a Web Worker in your browser; nothing leaves the page.
      </footer>
    </div>
  );
}
