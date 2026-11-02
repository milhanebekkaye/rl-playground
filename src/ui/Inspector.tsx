import { ACTION_NAMES, NUM_ACTIONS } from '../core/env';
import { readInspect } from '../core/inspect';
import type { LiveData } from './useTrainer';

const ARROWS = ['↑', '→', '↓', '←'];
const f = (v: number, d = 3) => (Math.abs(v) < 5e-4 ? '0' : v.toFixed(d)).replace(/^-0$/, '0');
const paren = (v: number) => (v < 0 && f(v) !== '0' ? `(${f(v)})` : f(v));

interface Props {
  live: LiveData;
  cell: number;
  width: number;
  /** Agent label shown in compare mode. */
  title?: string;
}

/**
 * Bellman inspector: for each action in one cell, the current Q-value, the
 * number of real updates, and the arithmetic of the most recent update.
 */
export function Inspector({ live, cell, width, title }: Props) {
  const x = cell % width;
  const y = Math.floor(cell / width);
  const target = live.algo === 'sarsa' ? "Q(s′,a′)" : "max Q(s′,·)";
  let best = -Infinity;
  for (let a = 0; a < NUM_ACTIONS; a++) best = Math.max(best, live.q[cell * NUM_ACTIONS + a]);
  return (
    <div className="inspector">
      <p className="inspector-head">
        {title && <strong>{title} · </strong>}cell ({x}, {y})
      </p>
      <table>
        <thead>
          <tr><th>Action</th><th>Q(s,a)</th><th>Visits</th><th>Last update: r + γ·{target} − Q(s,a) = δ</th></tr>
        </thead>
        <tbody>
          {Array.from({ length: NUM_ACTIONS }, (_, a) => {
            const q = live.q[cell * NUM_ACTIONS + a];
            const { visits, last } = readInspect(live.inspect, cell, a, NUM_ACTIONS);
            const isBest = visits > 0 && q === best;
            return (
              <tr key={a} className={isBest ? 'best' : ''}>
                <td><span aria-hidden="true">{ARROWS[a]}</span> {ACTION_NAMES[a]}</td>
                <td className="num">{f(q)}</td>
                <td className="num">{visits.toLocaleString()}</td>
                <td className="num">
                  {last
                    ? `${f(last.reward)} + ${f(last.gamma, 2)} × ${last.terminal ? '0 (terminal)' : paren(last.bootstrap)} − ${paren(last.qBefore)} = ${f(last.tdError)}`
                    : 'never tried'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
