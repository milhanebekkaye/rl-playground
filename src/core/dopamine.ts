/**
 * Classical-conditioning task with a time-step state representation, learned
 * with TD(0). Pure TypeScript.
 *
 * Each trial is a short sequence of time steps t = 0 .. length-1. A cue appears
 * at step `cue`, and a reward of size 1 is delivered at step `cue + delay` with
 * probability `prob`. The agent has no choices; it only learns a value V(t) for
 * each step since cue onset (a "tapped delay line" of cue-triggered states).
 * Steps before the cue are not learned: the cue arrives at an unpredictable
 * moment, so their value stays 0.
 *
 * Moving from step t-1 to step t produces a prediction error
 *
 *   delta(t) = r(t) + gamma * V(t) - V(t-1)
 *
 * and V(t-1) += alpha * delta(t). The trace of delta over trials is the
 * quantity compared, as an analogy, to dopamine reward prediction error.
 */
import type { Rng } from './rng';

export interface DopamineParams {
  alpha: number;
  gamma: number;
  /** Time step at which the cue appears (>= 1). */
  cue: number;
  /** Steps between the cue and the reward (>= 1). */
  delay: number;
  /** Probability that the reward is delivered on a normal trial. */
  prob: number;
  /** Steps per trial. Must exceed cue + delay. */
  length: number;
}

export interface TrialResult {
  /** delta[t] for each time step; delta[0] is 0 (nothing precedes it). */
  deltas: Float32Array;
  rewarded: boolean;
  /** The reward was withheld on purpose (omission trial). */
  omitted: boolean;
}

export const DEFAULT_DOPAMINE: DopamineParams = { alpha: 0.2, gamma: 0.95, cue: 4, delay: 8, prob: 1, length: 20 };

export function rewardTime(p: Pick<DopamineParams, 'cue' | 'delay'>): number {
  return p.cue + p.delay;
}

/** Clamps parameters into a consistent range (cue >= 1, delay >= 1, reward before the end). */
export function normaliseParams(p: DopamineParams): DopamineParams {
  const length = Math.max(6, Math.round(p.length));
  const cue = Math.min(Math.max(1, Math.round(p.cue)), length - 3);
  const delay = Math.min(Math.max(1, Math.round(p.delay)), length - 2 - cue);
  return { ...p, length, cue, delay, prob: Math.min(1, Math.max(0, p.prob)) };
}

export class DopamineModel {
  params: DopamineParams;
  /** Learned value of each time step (0 before the cue). */
  readonly V: Float64Array;
  readonly history: TrialResult[] = [];

  constructor(params: DopamineParams, private readonly rng: Rng) {
    this.params = normaliseParams(params);
    this.V = new Float64Array(this.params.length);
  }

  get trials(): number {
    return this.history.length;
  }

  /** Changes alpha, gamma or prob on the fly. cue/delay/length need a new model. */
  setLearningParams(p: Partial<Pick<DopamineParams, 'alpha' | 'gamma' | 'prob'>>): void {
    this.params = { ...this.params, ...p };
  }

  /** Runs one trial. With `omit` the reward is withheld whatever `prob` is. */
  runTrial(omit = false): TrialResult {
    const { alpha, gamma, cue, delay, prob, length } = this.params;
    const rt = cue + delay;
    const rewarded = !omit && this.rng() < prob;
    const deltas = new Float32Array(length);
    for (let t = 1; t < length; t++) {
      const r = t === rt && rewarded ? 1 : 0;
      const prev = t - 1 >= cue ? this.V[t - 1] : 0;
      const cur = t >= cue ? this.V[t] : 0;
      const delta = r + gamma * cur - prev;
      deltas[t] = delta;
      if (t - 1 >= cue) this.V[t - 1] += alpha * delta;
    }
    const result: TrialResult = { deltas, rewarded, omitted: omit };
    this.history.push(result);
    return result;
  }
}
