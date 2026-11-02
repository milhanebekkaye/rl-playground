import { describe, expect, it } from 'vitest';
import { DynaQ } from './dyna';
import { QLearning } from './qlearning';
import { Trainer } from './trainer';
import { mulberry32 } from './rng';
import { GridEnv, defaultSpec } from './env';

const P = { alpha: 0.5, gamma: 0.9, epsilon: 0 };

describe('Dyna-Q', () => {
  it('with 0 planning steps it is exactly Q-learning', () => {
    const q = new QLearning(6, 4, P, mulberry32(1));
    const d = new DynaQ(6, 4, P, mulberry32(1), 0);
    const moves: [number, number, number, number, boolean][] = [[0, 1, -1, 1, false], [1, 2, -1, 2, false], [2, 0, 5, 3, true]];
    for (const m of moves) {
      q.update(...m);
      d.update(...m);
    }
    expect(Array.from(d.q)).toEqual(Array.from(q.q));
  });

  it('planning replays remembered transitions: one real step propagates value back', () => {
    const d = new DynaQ(4, 4, { alpha: 1, gamma: 1, epsilon: 0 }, mulberry32(3), 0);
    d.update(1, 0, 10, 2, true); // learn that (1, a0) is worth 10
    d.update(0, 0, 0, 1, false); // (0, a0) -> state 1: without planning Q(0,0) = maxQ(1) = 10 already...
    expect(d.value(0, 0)).toBe(10);
    // Now a chain learned in the "wrong" order: the reward is discovered last.
    const e = new DynaQ(4, 4, { alpha: 1, gamma: 1, epsilon: 0 }, mulberry32(3), 0);
    e.update(0, 0, 0, 1, false); // 0 -> 1, Q(0,0) = 0
    e.update(1, 0, 0, 2, false); // 1 -> 2, Q(1,0) = 0
    e.update(2, 0, 10, 3, true); // 2 -> goal
    expect(e.value(0, 0)).toBe(0);
    e.plan(200); // replay the model: value flows back to the start
    expect(e.value(1, 0)).toBe(10);
    expect(e.value(0, 0)).toBe(10);
    expect(e.modelSize).toBe(3);
  });

  it('planning updates do not change visit counts or the last-update record', () => {
    const d = new DynaQ(4, 4, P, mulberry32(2), 20);
    d.update(0, 1, -1, 1, false);
    expect(d.visits(0, 1)).toBe(1);
    expect(d.lastUpdate(0, 1)?.qBefore).toBe(0);
    expect(d.lastUpdate(0, 1)?.reward).toBe(-1);
  });

  // Greedy policy (first-argmax, no RNG use) reaches the goal in the shortest number of moves.
  function isSolved(t: Trainer, shortest: number): boolean {
    const env = new GridEnv(t.env.spec);
    env.reset();
    for (let i = 0; i < shortest; i++) {
      let best = 0;
      for (let a = 1; a < 4; a++) if (t.agent.value(env.currentState, a) > t.agent.value(env.currentState, best)) best = a;
      const r = env.step(best);
      if (r.done) return r.outcome === 'goal' && i + 1 === shortest;
    }
    return false;
  }

  function episodesToSolve(algo: 'qlearning' | 'dyna', seed: number, shortest: number): number {
    const t = new Trainer(defaultSpec(), { alpha: 0.5, gamma: 0.95, epsilon: 0.2 }, seed, algo, 20);
    for (let ep = 1; ep <= 600; ep++) {
      while (t.stepOnce() === null) { /* run one episode */ }
      if (isSolved(t, shortest)) return ep;
    }
    return 600;
  }

  it('reaches the goal in fewer real episodes than Q-learning (fixed seeds)', () => {
    // Shortest path of the example maze, measured with a long-trained agent.
    const ref = new Trainer(defaultSpec(), { alpha: 0.5, gamma: 0.95, epsilon: 0.2 }, 99, 'dyna', 50);
    ref.run(30000);
    const shortest = ref.greedyRollout().states.length - 1;
    const seeds = [1, 2, 3, 4, 5, 6];
    const q = seeds.map((s) => episodesToSolve('qlearning', s, shortest));
    const d = seeds.map((s) => episodesToSolve('dyna', s, shortest));
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
    console.info(`[dyna] shortest=${shortest} qlearning=${q} dyna=${d} means ${mean(q).toFixed(1)} vs ${mean(d).toFixed(1)}`);
    expect(mean(d)).toBeLessThan(mean(q) * 0.75); // loose margin
    expect(d.filter((x) => x < 600).length).toBe(seeds.length);
  });
});
