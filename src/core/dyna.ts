/**
 * Dyna-Q: Q-learning plus a learned model of the world and "planning" updates.
 * Pure TypeScript.
 *
 * After every real step the agent (1) does the usual Q-learning update,
 * (2) stores the observed transition (s, a) -> (r, s', terminal) in a model, and
 * (3) replays `planningSteps` randomly chosen previously observed (s, a) pairs
 * through the same update rule, using the model instead of the environment.
 * With planningSteps = 0 it is exactly Q-learning. Environments here are
 * deterministic, so the model just remembers the last outcome of each pair.
 */
import { QLearning } from './qlearning';
import type { QParams } from './agent';
import type { Rng } from './rng';

export class DynaQ extends QLearning {
  planningSteps: number;
  private readonly seen: Uint8Array;
  private readonly observed: number[] = [];
  private readonly mReward: Float64Array;
  private readonly mNext: Int32Array;
  private readonly mTerminal: Uint8Array;

  constructor(numStates: number, numActions: number, params: QParams, rng: Rng, planningSteps = 10) {
    super(numStates, numActions, params, rng);
    this.planningSteps = planningSteps;
    const n = numStates * numActions;
    this.seen = new Uint8Array(n);
    this.mReward = new Float64Array(n);
    this.mNext = new Int32Array(n);
    this.mTerminal = new Uint8Array(n);
  }

  /** Number of distinct (s, a) pairs in the learned model. */
  get modelSize(): number {
    return this.observed.length;
  }

  /** Real update, model update, then planning. Returns the TD error of the real update. */
  override update(s: number, a: number, reward: number, next: number, terminal: boolean): number {
    const td = this.apply(s, a, reward, next, terminal, true);
    const k = s * this.numActions + a;
    if (!this.seen[k]) {
      this.seen[k] = 1;
      this.observed.push(k);
    }
    this.mReward[k] = reward;
    this.mNext[k] = next;
    this.mTerminal[k] = terminal ? 1 : 0;
    this.plan(this.planningSteps);
    return td;
  }

  /** Replays `n` random remembered transitions through the Q-learning update. */
  plan(n: number): void {
    const count = this.observed.length;
    for (let i = 0; i < n && count > 0; i++) {
      const k = this.observed[Math.floor(this.rng() * count)];
      const s = Math.floor(k / this.numActions);
      const a = k - s * this.numActions;
      this.apply(s, a, this.mReward[k], this.mNext[k], this.mTerminal[k] === 1, false);
    }
  }
}
