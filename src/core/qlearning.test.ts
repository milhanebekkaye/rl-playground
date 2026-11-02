import { describe, expect, it } from 'vitest';
import { QLearning } from './qlearning';
import { mulberry32 } from './rng';
import { GridEnv, type CellType, type EnvSpec } from './env';
import { Trainer } from './trainer';

const params = { alpha: 0.5, gamma: 0.9, epsilon: 0 };
const make = (p = params) => new QLearning(3, 4, p, mulberry32(7));

describe('Q-learning update rule', () => {
  it('first update from zeros: Q = alpha * (r + gamma * maxQ(s\'))', () => {
    const agent = make();
    agent.q[1 * 4 + 2] = 2; // maxQ(s'=1) = 2
    // target = 1 + 0.9 * 2 = 2.8, td = 2.8 - 0 = 2.8, Q = 0 + 0.5 * 2.8 = 1.4
    const td = agent.update(0, 1, 1, 1, false);
    expect(td).toBeCloseTo(2.8, 12);
    expect(agent.value(0, 1)).toBeCloseTo(1.4, 12);
  });

  it('moves an existing value toward the target', () => {
    const agent = make({ alpha: 0.1, gamma: 0.5, epsilon: 0 });
    agent.q[0 * 4 + 0] = 4;
    agent.q[2 * 4 + 3] = 6;
    // target = -1 + 0.5 * 6 = 2, td = 2 - 4 = -2, Q = 4 + 0.1 * -2 = 3.8
    const td = agent.update(0, 0, -1, 2, false);
    expect(td).toBeCloseTo(-2, 12);
    expect(agent.value(0, 0)).toBeCloseTo(3.8, 12);
  });

  it('ignores the next state value when the transition is terminal', () => {
    const agent = make();
    agent.q[1 * 4 + 0] = 100;
    // target = r = 10, td = 10, Q = 0.5 * 10 = 5
    const td = agent.update(0, 2, 10, 1, true);
    expect(td).toBe(10);
    expect(agent.value(0, 2)).toBe(5);
  });

  it('bootstraps from the max over next actions, not the action taken', () => {
    const agent = make({ alpha: 1, gamma: 1, epsilon: 0 });
    agent.q[1 * 4 + 0] = -3;
    agent.q[1 * 4 + 3] = 2;
    agent.update(0, 0, 0, 1, false);
    expect(agent.value(0, 0)).toBe(2);
  });

  it('only changes the (s, a) entry it updates', () => {
    const agent = make();
    agent.update(0, 1, 1, 1, true);
    const changed = Array.from(agent.q).filter((v) => v !== 0).length;
    expect(changed).toBe(1);
  });
});

describe('epsilon-greedy', () => {
  it('epsilon = 0 always picks the greedy action', () => {
    const agent = make({ alpha: 0.5, gamma: 0.9, epsilon: 0 });
    agent.q[0 * 4 + 2] = 1;
    for (let i = 0; i < 200; i++) expect(agent.selectAction(0)).toBe(2);
  });

  it('epsilon = 1 explores every action roughly uniformly', () => {
    const agent = make({ alpha: 0.5, gamma: 0.9, epsilon: 1 });
    agent.q[0 * 4 + 2] = 1;
    const counts = [0, 0, 0, 0];
    for (let i = 0; i < 4000; i++) counts[agent.selectAction(0)]++;
    for (const c of counts) expect(c).toBeGreaterThan(800);
  });

  it('breaks greedy ties across all tied actions', () => {
    const agent = make({ alpha: 0.5, gamma: 0.9, epsilon: 0 });
    const seen = new Set<number>();
    for (let i = 0; i < 200; i++) seen.add(agent.selectAction(1));
    expect(seen.size).toBe(4);
  });
});

// 5x5 maze, S start (bottom-left), G goal (top-right), T trap, # wall.
//   . . # . G
//   . # # . .
//   . . . . T
//   # # . # .
//   S . . . .
function maze(): EnvSpec {
  const rows = ['..#.G', '.##..', '....T', '##.#.', 'S....'];
  const cells: CellType[] = [];
  for (const row of rows)
    for (const ch of row)
      cells.push(ch === '#' ? 'wall' : ch === 'S' ? 'start' : ch === 'G' ? 'goal' : ch === 'T' ? 'trap' : 'empty');
  return { width: 5, height: 5, cells, goalReward: 1, trapReward: -1, stepReward: -0.04, maxSteps: 100 };
}

describe('learning a small maze', () => {
  it('finds the shortest path to the goal with a greedy policy', () => {
    const trainer = new Trainer(maze(), { alpha: 0.3, gamma: 0.95, epsilon: 0.2 }, 42);
    // Run 1500 episodes.
    while (trainer.episodes < 1500) trainer.stepOnce();

    // Shortest path from (0,4): right x2 to (2,4), up x2 to (2,2), right x1 (3,2),
    // up x2 via (3,1) to (3,0), right to (4,0): 2 + 2 + 1 + 2 + 1 = 8 moves.
    const { states, outcome } = trainer.greedyRollout();
    expect(outcome).toBe('goal');
    expect(states.length - 1).toBe(8);
    // The start state's value should be positive and below the goal reward.
    const v0 = trainer.agent.maxQ(trainer.env.startState);
    expect(v0).toBeGreaterThan(0.3);
    expect(v0).toBeLessThan(1);
  });

  it('is reproducible for a fixed seed', () => {
    const a = new Trainer(maze(), { alpha: 0.3, gamma: 0.9, epsilon: 0.3 }, 5);
    const b = new Trainer(maze(), { alpha: 0.3, gamma: 0.9, epsilon: 0.3 }, 5);
    a.run(2000);
    b.run(2000);
    expect(Array.from(a.agent.q)).toEqual(Array.from(b.agent.q));
  });

  it('does not bootstrap through a timeout as if it were terminal', () => {
    // 1x3 corridor, no way to the goal within maxSteps = 1: timeout update must still bootstrap.
    const spec: EnvSpec = {
      width: 3, height: 1, cells: ['start', 'empty', 'goal'],
      goalReward: 1, trapReward: -1, stepReward: 0, maxSteps: 1,
    };
    const env = new GridEnv(spec);
    env.reset();
    const agent = new QLearning(3, 4, { alpha: 1, gamma: 0.9, epsilon: 0 }, mulberry32(1));
    agent.q[1 * 4 + 1] = 1;
    const res = env.step(1); // moves to cell 1, then times out
    expect(res.done && !res.terminal).toBe(true);
    agent.update(0, 1, res.reward, res.state, res.terminal);
    expect(agent.value(0, 1)).toBeCloseTo(0.9, 12);
  });
});
