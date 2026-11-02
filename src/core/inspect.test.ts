import { describe, expect, it } from 'vitest';
import { QLearning } from './qlearning';
import { Sarsa } from './sarsa';
import { readInspect } from './inspect';
import { mulberry32 } from './rng';

describe('Bellman inspector records', () => {
  it('Q-learning: stores the numbers of the last update (hand-computed)', () => {
    const a = new QLearning(3, 4, { alpha: 0.5, gamma: 0.9, epsilon: 0 }, mulberry32(1));
    a.q[1 * 4 + 2] = 2; // maxQ(s'=1) = 2
    a.q[0 * 4 + 1] = 1; // Q(0, 1) = 1 before
    a.update(0, 1, -0.5, 1, false);
    // delta = -0.5 + 0.9 * 2 - 1 = 0.3, new Q = 1 + 0.5 * 0.3 = 1.15
    const last = a.lastUpdate(0, 1)!;
    expect(last).toMatchObject({ visits: 1, reward: -0.5, next: 1, terminal: false, qBefore: 1, gamma: 0.9 });
    expect(last.bootstrap).toBe(2);
    expect(last.tdError).toBeCloseTo(0.3, 12);
    expect(a.value(0, 1)).toBeCloseTo(1.15, 12);
    // the identity the UI prints: r + gamma * bootstrap - qBefore = delta
    expect(last.reward + last.gamma * last.bootstrap - last.qBefore).toBeCloseTo(last.tdError, 12);
  });

  it('counts visits per (s, a) and keeps the latest update', () => {
    const a = new QLearning(3, 4, { alpha: 0.1, gamma: 0.9, epsilon: 0 }, mulberry32(1));
    a.update(0, 1, 1, 1, false);
    a.update(0, 1, 2, 2, false);
    a.update(0, 3, 0, 0, false);
    expect(a.visits(0, 1)).toBe(2);
    expect(a.visits(0, 3)).toBe(1);
    expect(a.visits(0, 0)).toBe(0);
    expect(a.lastUpdate(0, 0)).toBeNull();
    expect(a.lastUpdate(0, 1)).toMatchObject({ reward: 2, next: 2 });
  });

  it('terminal transitions record a zero bootstrap', () => {
    const a = new QLearning(3, 4, { alpha: 1, gamma: 0.9, epsilon: 0 }, mulberry32(1));
    a.q[1 * 4] = 50;
    a.update(0, 0, 3, 1, true);
    const last = a.lastUpdate(0, 0)!;
    expect(last.terminal).toBe(true);
    expect(last.bootstrap).toBe(0);
    expect(last.tdError).toBe(3);
  });

  it('SARSA: bootstrap is Q(s\', a\') of the action taken, not the max', () => {
    const a = new Sarsa(3, 4, { alpha: 0.5, gamma: 0.9, epsilon: 0 }, mulberry32(1));
    a.q[1 * 4 + 0] = 10;
    a.q[1 * 4 + 3] = 4;
    a.update(0, 2, 1, 1, 3, false);
    expect(a.lastUpdate(0, 2)!.bootstrap).toBe(4);
    expect(a.lastUpdate(0, 2)!.tdError).toBeCloseTo(1 + 0.9 * 4, 12);
  });

  it('readInspect works on a copied buffer (as sent to the UI)', () => {
    const a = new QLearning(3, 4, { alpha: 0.5, gamma: 0.9, epsilon: 0 }, mulberry32(1));
    a.update(2, 3, 1, 0, false);
    const copy = Float32Array.from(a.inspect);
    const { visits, last } = readInspect(copy, 2, 3);
    expect(visits).toBe(1);
    expect(last!.reward).toBe(1);
  });
});
