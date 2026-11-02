import { describe, expect, it } from 'vitest';
import { GridEnv, defaultSpec, emptySpec, validateSpec, type CellType, type EnvSpec } from './env';

// Layout used below (3x3), S = start, G = goal, T = trap, # = wall:
//   S . G
//   . # .
//   T . .
function tiny(overrides: Partial<EnvSpec> = {}): EnvSpec {
  const c: CellType[] = ['start', 'empty', 'goal', 'empty', 'wall', 'empty', 'trap', 'empty', 'empty'];
  return { width: 3, height: 3, cells: c, goalReward: 10, trapReward: -5, stepReward: -1, maxSteps: 50, ...overrides };
}
const UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3;

describe('GridEnv', () => {
  it('starts at the start cell', () => {
    const env = new GridEnv(tiny());
    expect(env.reset()).toBe(0);
  });

  it('moves deterministically and charges the step reward', () => {
    const env = new GridEnv(tiny());
    env.reset();
    const r = env.step(RIGHT);
    expect(r).toMatchObject({ state: 1, reward: -1, done: false, terminal: false, outcome: null });
    const r2 = env.step(DOWN); // (1,1) is a wall
    expect(r2.state).toBe(1);
    expect(r2.reward).toBe(-1);
  });

  it('is blocked by walls and by the grid border', () => {
    const env = new GridEnv(tiny());
    env.reset();
    expect(env.step(UP).state).toBe(0);
    expect(env.step(LEFT).state).toBe(0);
    env.step(RIGHT);
    expect(env.step(DOWN).state).toBe(1);
  });

  it('terminates with the goal reward', () => {
    const env = new GridEnv(tiny());
    env.reset();
    env.step(RIGHT);
    const r = env.step(RIGHT);
    expect(r).toMatchObject({ state: 2, reward: 10, done: true, terminal: true, outcome: 'goal' });
    expect(() => env.step(LEFT)).toThrow();
  });

  it('terminates with the trap reward', () => {
    const env = new GridEnv(tiny());
    env.reset();
    env.step(DOWN);
    const r = env.step(DOWN);
    expect(r).toMatchObject({ state: 6, reward: -5, done: true, terminal: true, outcome: 'trap' });
  });

  it('truncates at maxSteps without marking the state terminal', () => {
    const env = new GridEnv(tiny({ maxSteps: 3 }));
    env.reset();
    env.step(UP);
    env.step(UP);
    const r = env.step(UP);
    expect(r).toMatchObject({ done: true, terminal: false, outcome: 'timeout' });
  });

  it('reset starts a fresh episode', () => {
    const env = new GridEnv(tiny());
    env.reset();
    env.step(RIGHT);
    env.step(RIGHT);
    env.reset();
    expect(env.currentState).toBe(0);
    expect(env.stepCount).toBe(0);
    expect(() => env.step(RIGHT)).not.toThrow();
  });

  it('rejects invalid actions and specs without exactly one start', () => {
    const env = new GridEnv(tiny());
    env.reset();
    expect(() => env.step(4)).toThrow();
    const bad = tiny();
    bad.cells[0] = 'empty';
    expect(validateSpec(bad)).not.toBeNull();
    expect(() => new GridEnv(bad)).toThrow();
  });

  it('ships valid default specs', () => {
    expect(validateSpec(defaultSpec())).toBeNull();
    expect(validateSpec(emptySpec(5, 4))).toBeNull();
  });
});
