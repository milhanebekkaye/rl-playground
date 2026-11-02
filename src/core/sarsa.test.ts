import { describe, expect, it } from 'vitest';
import { Sarsa } from './sarsa';
import { QLearning } from './qlearning';
import { Trainer } from './trainer';
import { mulberry32 } from './rng';
import { cliffSpec, validateSpec } from './env';

const P = { alpha: 0.5, gamma: 0.9, epsilon: 0 };

describe('SARSA update rule', () => {
  it('uses the next action actually taken: Q = Q + alpha * (r + gamma * Q(s\',a\') - Q)', () => {
    const agent = new Sarsa(3, 4, P, mulberry32(1));
    agent.q[0 * 4 + 1] = 1;
    agent.q[1 * 4 + 2] = 3; // the action SARSA will take next
    agent.q[1 * 4 + 0] = 10; // the best action; SARSA must not use it
    // target = -1 + 0.9 * 3 = 1.7, td = 0.7, Q = 1 + 0.5 * 0.7 = 1.35
    const td = agent.update(0, 1, -1, 1, 2, false);
    expect(td).toBeCloseTo(0.7, 12);
    expect(agent.value(0, 1)).toBeCloseTo(1.35, 12);
  });

  it('differs from Q-learning on the same transition', () => {
    const sarsa = new Sarsa(3, 4, P, mulberry32(1));
    const ql = new QLearning(3, 4, P, mulberry32(1));
    for (const ag of [sarsa, ql]) {
      ag.q[1 * 4 + 2] = 3;
      ag.q[1 * 4 + 0] = 10;
    }
    sarsa.update(0, 1, -1, 1, 2, false); // 0.5 * (-1 + 2.7)      = 0.85
    ql.update(0, 1, -1, 1, false); //       0.5 * (-1 + 9)        = 4
    expect(sarsa.value(0, 1)).toBeCloseTo(0.85, 12);
    expect(ql.value(0, 1)).toBeCloseTo(4, 12);
  });

  it('ignores the next state when terminal', () => {
    const agent = new Sarsa(3, 4, P, mulberry32(1));
    agent.q[1 * 4 + 0] = 100;
    const td = agent.update(0, 2, 10, 1, 0, true);
    expect(td).toBe(10);
    expect(agent.value(0, 2)).toBe(5);
  });

  it('reduces to Q-learning when the next action is the greedy one', () => {
    const sarsa = new Sarsa(3, 4, P, mulberry32(1));
    const ql = new QLearning(3, 4, P, mulberry32(1));
    for (const ag of [sarsa, ql]) ag.q[1 * 4 + 3] = 2;
    sarsa.update(0, 0, 1, 1, 3, false);
    ql.update(0, 0, 1, 1, false);
    expect(sarsa.value(0, 0)).toBeCloseTo(ql.value(0, 0), 12);
  });
});

describe('cliff walking: on-policy vs off-policy', () => {
  // Row index of the cliff edge (the row directly above the cliff) is height - 2 = 2.
  const EDGE_ROW = 2;
  const params = { alpha: 0.2, gamma: 0.99, epsilon: 0.1 };

  function greedyPathRows(algo: 'qlearning' | 'sarsa', seed: number) {
    const t = new Trainer(cliffSpec(), params, seed, algo);
    while (t.episodes < 3000) t.stepOnce();
    const { states, outcome } = t.greedyRollout();
    const width = t.env.spec.width;
    return { outcome, rows: states.map((s) => Math.floor(s / width)), length: states.length - 1 };
  }

  it('ships a valid cliff spec', () => {
    expect(validateSpec(cliffSpec())).toBeNull();
  });

  it.each([1, 2, 3, 4])('seed %i: Q-learning hugs the cliff edge, SARSA takes the safe route', (seed) => {
    const ql = greedyPathRows('qlearning', seed);
    const sarsa = greedyPathRows('sarsa', seed);
    expect(ql.outcome).toBe('goal');
    expect(sarsa.outcome).toBe('goal');
    // Q-learning: shortest path (up, 10 right... along the edge row, down) = 13 moves, on the edge row.
    expect(ql.length).toBe(13);
    expect(ql.rows.filter((r) => r === EDGE_ROW).length).toBeGreaterThanOrEqual(10);
    // SARSA: a longer path that keeps clear of the edge row for most of the walk.
    expect(sarsa.length).toBeGreaterThan(13);
    expect(sarsa.rows.filter((r) => r === EDGE_ROW).length).toBeLessThanOrEqual(3);
    expect(Math.min(...sarsa.rows)).toBeLessThan(EDGE_ROW);
  });
});
