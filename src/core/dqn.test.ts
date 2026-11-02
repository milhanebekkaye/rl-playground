import { describe, expect, it } from 'vitest';
import { CartPole } from './cartpole';
import { CartPoleTrainer, evaluate } from './cartpoleTrainer';
import { DEFAULT_DQN, DqnAgent, ReplayBuffer } from './dqn';
import { mulberry32 } from './rng';

describe('replay buffer', () => {
  it('overwrites the oldest transitions once full', () => {
    const b = new ReplayBuffer(3, 2);
    for (let i = 0; i < 5; i++) b.push([i, i], i % 2, i, [i + 1, i + 1], i === 4);
    expect(b.size).toBe(3);
    expect(Array.from(b.r).sort()).toEqual([2, 3, 4]);
  });
});

describe('DQN agent', () => {
  it('learns a one-step bandit-like target: Q(s, a) moves toward r + gamma * max Q_target(s\')', () => {
    const agent = new DqnAgent({ ...DEFAULT_DQN, stateSize: 1, numActions: 2, hidden: [8], warmup: 1, batchSize: 16, lr: 0.01, gamma: 0.9 }, mulberry32(2));
    // Terminal transitions only: target is exactly the reward. Action 1 pays 1, action 0 pays 0.
    for (let i = 0; i < 400; i++) {
      agent.buffer.push([0.5], 0, 0, [0.5], true);
      agent.buffer.push([0.5], 1, 1, [0.5], true);
    }
    for (let i = 0; i < 600; i++) agent.trainStep();
    const q = agent.qValues([0.5]);
    expect(q[1]).toBeGreaterThan(0.85);
    expect(q[1]).toBeLessThan(1.15);
    expect(Math.abs(q[0])).toBeLessThan(0.15);
    expect(agent.greedyAction([0.5])).toBe(1);
  });

  it('copies the online network into the target network on schedule', () => {
    const agent = new DqnAgent({ ...DEFAULT_DQN, warmup: 1, batchSize: 4, targetSync: 5 }, mulberry32(3));
    const x = [0.1, 0.2, 0.3, 0.4];
    for (let i = 0; i < 4; i++) agent.observe(x, i % 2, 1, x, false);
    const a = Array.from(agent.online.forward(x, agent.online.makeActivations()));
    const b = Array.from(agent.target.forward(x, agent.target.makeActivations()));
    expect(a).not.toEqual(b); // online moved, target not yet synced
    agent.observe(x, 0, 1, x, false); // step 5 -> train, then sync
    const a2 = Array.from(agent.online.forward(x, agent.online.makeActivations()));
    const b2 = Array.from(agent.target.forward(x, agent.target.makeActivations()));
    expect(a2).toEqual(b2);
  });

  it('decays epsilon toward its floor', () => {
    const agent = new DqnAgent({ ...DEFAULT_DQN, epsDecay: 0.5, epsEnd: 0.1, warmup: 1e9 }, mulberry32(1));
    for (let i = 0; i < 10; i++) agent.observe([0, 0, 0, 0], 0, 1, [0, 0, 0, 0], false);
    expect(agent.epsilon).toBe(0.1);
  });
});

describe('CartPole DQN learning (fixed seeds)', () => {
  // Measured, see README. A uniformly random policy scores about 22 steps on average.
  const BUDGET = 20000;
  const EVERY = 2500;
  const SEEDS = [1, 2, 3];

  function randomBaseline(): number {
    const rng = mulberry32(5);
    let total = 0;
    for (let i = 0; i < 200; i++) {
      const env = new CartPole(mulberry32(100 + i));
      for (;;) {
        total += 1;
        const r = env.step(rng() < 0.5 ? 0 : 1);
        if (r.terminated || r.truncated) break;
      }
    }
    return total / 200;
  }

  it('clearly beats a random policy within a bounded step budget', () => {
    const random = randomBaseline();
    const report: string[] = [`random policy mean return = ${random.toFixed(1)}`];
    const lateMeans: number[] = [];
    const bests: number[] = [];
    for (const seed of SEEDS) {
      const t = new CartPoleTrainer(seed);
      const evals: number[] = [];
      for (let k = 0; k < BUDGET / EVERY; k++) {
        t.run(EVERY);
        evals.push(evaluate(t, 5));
      }
      const late = evals.slice(2); // after the first 5000 steps
      lateMeans.push(late.reduce((a, b) => a + b, 0) / late.length);
      bests.push(Math.max(...evals));
      report.push(`seed ${seed}: greedy eval every ${EVERY} steps = ${evals.map((e) => Math.round(e)).join(', ')}`);
    }
    report.push(`mean of post-5000-step evals per seed = ${lateMeans.map((m) => m.toFixed(0)).join(', ')}; best = ${bests.map((m) => m.toFixed(0)).join(', ')}`);
    console.info('\n[dqn] ' + report.join('\n[dqn] '));
    // Every seed's average greedy performance is several times the random policy, and every seed reaches a
    // long-balance (>= 150 steps) policy at some checkpoint. It is deliberately not "solved": DQN is unstable.
    for (const m of lateMeans) expect(m).toBeGreaterThan(3 * random);
    for (const b of bests) expect(b).toBeGreaterThanOrEqual(150);
  }, 240000);
});
