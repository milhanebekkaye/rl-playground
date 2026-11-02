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

function Home({ go }: { go: (id: LabId) => void }) {
  return (
    <div className="lab">
      <header className="lab-head">
        <p className="eyebrow">Reinforcement learning, in your browser</p>
        <h1>RL <em>Playground</em></h1>
        <p className="lede">Small, honest experiments you can poke at. Pick a lab.</p>
      </header>
      <ul className="lab-cards">
        <li className="card"><h2>Gridworld</h2><p>Draw a maze and watch Q-learning, SARSA and Dyna-Q learn it.</p><button className="btn primary" onClick={() => go('gridworld')}>Open</button></li>
        <li className="card"><h2>Dopamine lab</h2><p>Prediction error moving from reward to cue.</p><button className="btn primary" onClick={() => go('dopamine')}>Open</button></li>
      </ul>
    </div>
  );
}
