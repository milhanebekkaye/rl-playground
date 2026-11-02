import { describe, expect, it } from 'vitest';
import { CartPole, FORCE, MAX_STEPS, THETA_LIMIT, normalise } from './cartpole';
import { mulberry32 } from './rng';

describe('CartPole physics', () => {
  it('starts near the upright equilibrium and is reproducible for a seed', () => {
    const a = new CartPole(mulberry32(4));
    const b = new CartPole(mulberry32(4));
    expect(a.state).toEqual(b.state);
    for (const v of a.state) expect(Math.abs(v)).toBeLessThanOrEqual(0.05);
  });

  it('pushing right accelerates the cart right and tips the pole left (hand-checked first step)', () => {
    const env = new CartPole(mulberry32(1));
    env.state = [0, 0, 0, 0];
    const r = env.step(1);
    // From rest, upright: temp = F / M = 10 / 1.1; thetaAcc = -temp / (0.5 * (4/3 - 0.1/1.1)); xAcc = temp - 0.05 * thetaAcc / 1.1
    const temp = FORCE / 1.1;
    const thetaAcc = -temp / (0.5 * (4 / 3 - 0.1 / 1.1));
    const xAcc = temp - (0.05 * thetaAcc) / 1.1;
    expect(r.state[1]).toBeCloseTo(0.02 * xAcc, 12);
    expect(r.state[3]).toBeCloseTo(0.02 * thetaAcc, 12);
    expect(r.state[1]).toBeGreaterThan(0);
    expect(r.state[3]).toBeLessThan(0);
    expect(r.reward).toBe(1);
    expect(r.terminated).toBe(false);
  });

  it('terminates when the pole passes 12 degrees or the cart leaves the track', () => {
    const e = new CartPole(mulberry32(1));
    e.state = [0, 0, THETA_LIMIT - 1e-4, 1];
    expect(e.step(1).terminated).toBe(true);
    e.state = [2.399, 5, 0, 0];
    expect(e.step(1).terminated).toBe(true);
  });

  it('a fixed push-left policy fails quickly; truncation happens at 500 steps', () => {
    const env = new CartPole(mulberry32(2));
    let n = 0;
    while (!env.step(0).terminated) n++;
    expect(n).toBeLessThan(60);
    // A hand-written angle-feedback controller balances for the full 500 steps.
    const e2 = new CartPole(mulberry32(2));
    let last;
    for (let i = 0; i < 1000; i++) {
      const [, , th, thd] = e2.state;
      last = e2.step(th + 0.5 * thd > 0 ? 1 : 0);
      if (last.terminated || last.truncated) break;
    }
    expect(last!.truncated).toBe(true);
    expect(e2.steps).toBe(MAX_STEPS);
  });

  it('normalise gives roughly unit-scale inputs', () => {
    const v = normalise([2.4, 2, THETA_LIMIT, -2]);
    expect(Array.from(v)).toEqual([1, 1, 1, -1]);
  });
});
