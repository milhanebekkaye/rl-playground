import { useEffect, useState } from 'react';
import { CartPoleLab } from './labs/CartPoleLab';
import { DopamineLab } from './labs/DopamineLab';
import { GridworldLab } from './labs/GridworldLab';

export type LabId = 'home' | 'gridworld' | 'dopamine' | 'cartpole';

/** `#v1&...` share links from V1/V2 still open the gridworld; `#/name` selects a lab. */
export function labFromHash(hash: string): LabId {
  if (hash.startsWith('#v1')) return 'gridworld';
  const name = hash.replace(/^#\/?/, '').split(/[?&]/)[0];
  return name === 'gridworld' || name === 'dopamine' || name === 'cartpole' ? name : 'home';
}

const NAV: { id: LabId; label: string }[] = [
  { id: 'home', label: 'Home' },
  { id: 'gridworld', label: 'Gridworld' },
  { id: 'dopamine', label: 'Dopamine lab' },
  { id: 'cartpole', label: 'CartPole' },
];

export function Shell() {
  const [lab, setLab] = useState<LabId>(() => labFromHash(location.hash));
  useEffect(() => {
    const on = () => setLab(labFromHash(location.hash));
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const go = (id: LabId) => {
    if (id === lab) return;
    setLab(id);
    history.pushState(null, '', id === 'home' ? location.pathname + location.search : `#/${id}`);
  };

  return (
    <div className="app">
      <nav className="topnav" aria-label="Labs">
        <a className="brand" href={location.pathname} onClick={(e) => { e.preventDefault(); go('home'); }}>RL <em>Playground</em></a>
        <ul>
          {NAV.map((n) => (
            <li key={n.id}>
              <a href={n.id === 'home' ? location.pathname : `#/${n.id}`} aria-current={lab === n.id ? 'page' : undefined}
                onClick={(e) => { e.preventDefault(); go(n.id); }}>{n.label}</a>
            </li>
          ))}
        </ul>
      </nav>
      {lab === 'gridworld' && <GridworldLab />}
      {lab === 'dopamine' && <DopamineLab />}
      {lab === 'cartpole' && <CartPoleLab />}
      {lab === 'home' && <Home go={go} />}
      <footer className="foot">MIT licensed. Everything runs in your browser; nothing leaves the page.</footer>
    </div>
  );
}

const LABS: { id: Exclude<LabId, 'home'>; title: string; kicker: string; body: string; shows: string }[] = [
  {
    id: 'gridworld',
    title: 'Gridworld',
    kicker: 'Tabular RL',
    body: 'Draw a maze, then watch Q-learning, SARSA or Dyna-Q learn it. Heatmap, policy arrows, a Bellman inspector that shows each update with real numbers, and shareable links.',
    shows: 'Q-values, TD error, on- vs off-policy, planning',
  },
  {
    id: 'dopamine',
    title: 'Dopamine lab',
    kicker: 'Neuro-AI analogy',
    body: 'A cue, then a reward. A TD(0) learner shows the prediction error moving from the reward to the cue, and dipping below zero when an expected reward is omitted.',
    shows: 'Reward prediction error, as an analogy',
  },
  {
    id: 'cartpole',
    title: 'CartPole',
    kicker: 'Deep RL from scratch',
    body: 'A small neural network, with hand-written backprop and Adam, learns to balance a pole. Replay buffer, target network, live Q-values.',
    shows: 'From a table to a network (DQN)',
  },
];

function Home({ go }: { go: (id: LabId) => void }) {
  return (
    <div className="lab">
      <header className="lab-head">
        <p className="eyebrow">Reinforcement learning, in your browser</p>
        <h1>RL <em>Playground</em></h1>
        <p className="lede">
          Three small experiments that show how an agent learns from reward: you can edit the world, change the
          hyperparameters and watch the learning signal as it happens. No backend and no ML library; the algorithms are
          written in plain TypeScript and run in Web Workers.
        </p>
      </header>
      <ul className="lab-cards">
        {LABS.map((l) => (
          <li key={l.id} className="card lab-card">
            <p className="eyebrow">{l.kicker}</p>
            <h2>{l.title}</h2>
            <p>{l.body}</p>
            <p className="shows">{l.shows}</p>
            <a className="btn primary" href={`#/${l.id}`} onClick={(e) => { e.preventDefault(); go(l.id); }}>Open {l.title}</a>
          </li>
        ))}
      </ul>
      <p className="hint home-note">
        Honest limits: these are teaching-scale demos. The dopamine lab is an analogy, not a brain model, and the CartPole
        DQN is not guaranteed to stay at its peak (see the README for measured results).
      </p>
    </div>
  );
}
