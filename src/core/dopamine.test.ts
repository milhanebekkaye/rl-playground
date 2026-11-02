import { describe, expect, it } from 'vitest';
import { DEFAULT_DOPAMINE, DopamineModel, normaliseParams, rewardTime } from './dopamine';
import { mulberry32 } from './rng';

const make = (over = {}) => new DopamineModel({ ...DEFAULT_DOPAMINE, ...over }, mulberry32(1));

describe('dopamine / TD(0) conditioning', () => {
  it('first trial, hand-computed: all the error is at the reward, none at the cue', () => {
    const m = make({ alpha: 0.5, gamma: 0.9, cue: 2, delay: 3, length: 10 }); // reward at t = 5
    const r = m.runTrial();
    expect(r.deltas[2]).toBe(0); // cue: gamma * V(2) - 0
    expect(r.deltas[5]).toBeCloseTo(1, 6); // r + gamma * V(5) - V(4) = 1
    // V(4) += 0.5 * 1
    expect(m.V[4]).toBeCloseTo(0.5, 12);
    // On trial 2, delta(4) = gamma * V(4) - V(3) = 0.45, so V(3) = 0.5 * 0.45 afterwards... check delta itself:
    const r2 = m.runTrial();
    expect(r2.deltas[4]).toBeCloseTo(0.9 * 0.5, 6); // deltas are stored as float32
    expect(r2.deltas[5]).toBeCloseTo(0.5, 6); // 1 - V(4)
  });

  it('error at the reward fades as learning proceeds', () => {
    const m = make();
    const rt = rewardTime(m.params);
    for (let i = 0; i < 400; i++) m.runTrial();
    expect(m.history[0].deltas[rt]).toBeCloseTo(1, 6);
    expect(Math.abs(m.history[399].deltas[rt])).toBeLessThan(0.02);
    // monotone decrease on average: early > middle > late
    expect(m.history[10].deltas[rt]).toBeGreaterThan(m.history[100].deltas[rt]);
    expect(m.history[100].deltas[rt]).toBeGreaterThan(m.history[399].deltas[rt]);
  });

  it('error at the cue grows from zero toward gamma^(delay-1)', () => {
    const m = make();
    const { cue, delay, gamma } = m.params;
    for (let i = 0; i < 400; i++) m.runTrial();
    expect(m.history[0].deltas[cue]).toBe(0);
    const late = m.history[399].deltas[cue];
    expect(late).toBeGreaterThan(0.5);
    expect(late).toBeCloseTo(Math.pow(gamma, delay - 1), 1);
    expect(m.history[50].deltas[cue]).toBeGreaterThan(m.history[5].deltas[cue]);
  });

  it('an omitted reward gives a negative error at the expected reward time', () => {
    const m = make();
    const rt = rewardTime(m.params);
    for (let i = 0; i < 400; i++) m.runTrial();
    const omitted = m.runTrial(true);
    expect(omitted.omitted).toBe(true);
    expect(omitted.rewarded).toBe(false);
    expect(omitted.deltas[rt]).toBeLessThan(-0.8);
    // ... and only there: no large positive error elsewhere in the trial
    for (let t = 0; t < omitted.deltas.length; t++) if (t !== rt) expect(omitted.deltas[t]).toBeLessThan(0.97);
  });

  it('with reward probability p the value converges to p and rewards give a positive error of 1 - p', () => {
    const m = make({ prob: 0.5, alpha: 0.02 }); // small alpha: low-noise estimate of the mean
    const { cue, delay } = m.params;
    for (let i = 0; i < 6000; i++) m.runTrial();
    expect(m.V[cue + delay - 1]).toBeGreaterThan(0.3);
    expect(m.V[cue + delay - 1]).toBeLessThan(0.7);
    const last = m.history.slice(-400);
    const rewardedDeltas = last.filter((t) => t.rewarded).map((t) => t.deltas[cue + delay]);
    const meanR = rewardedDeltas.reduce((a, b) => a + b, 0) / rewardedDeltas.length;
    expect(meanR).toBeGreaterThan(0.3);
    expect(meanR).toBeLessThan(0.7);
  });

  it('normaliseParams keeps cue and reward inside the trial', () => {
    const p = normaliseParams({ ...DEFAULT_DOPAMINE, cue: 99, delay: 99, length: 12 });
    expect(p.cue + p.delay).toBeLessThan(p.length);
    expect(p.cue).toBeGreaterThanOrEqual(1);
    expect(p.delay).toBeGreaterThanOrEqual(1);
  });
});
