/**
 * Couples an environment with a TD-control agent (Q-learning or SARSA) and runs
 * it one step at a time, tracking per-episode and per-cell TD statistics.
 * UI-free so it runs in a Web Worker and in unit tests alike.
 */
import { GridEnv, NUM_ACTIONS, type EnvSpec, type Outcome } from './env';
import type { TabularAgent, QParams } from './agent';
import { QLearning } from './qlearning';
import { Sarsa } from './sarsa';
import { DynaQ } from './dyna';
import { mulberry32 } from './rng';

export type Algo = 'qlearning' | 'sarsa' | 'dyna';

export interface EpisodeRecord {
  reward: number;
  steps: number;
  outcome: Outcome;
}

/** TD errors accumulated since the last drainTd() call. */
export interface TdBatch {
  n: number;
  sumAbs: number;
  sumSigned: number;
}

/** Smoothing for the per-cell running mean of |TD error| (weight of the newest sample). */
export const TD_EMA = 0.1;

export class Trainer {
  readonly env: GridEnv;
  readonly agent: TabularAgent;
  /** Per-cell exponential moving average of |TD error|, updated whenever Q(s, .) is updated. */
  readonly tdMean: Float32Array;
  trail: number[];
  episodes = 0;
  totalSteps = 0;
  lastTdError = 0;
  private episodeReward = 0;
  /** SARSA: the action already chosen for the current state. */
  private pending: number | null = null;
  /** The episode has ended; the next step() starts a new one. The final state stays visible until then. */
  private needsReset = false;
  private td: TdBatch = { n: 0, sumAbs: 0, sumSigned: 0 };

  constructor(spec: EnvSpec, params: QParams, seed = 1, readonly algo: Algo = 'qlearning', planning = 10) {
    this.env = new GridEnv(spec);
    const rng = mulberry32(seed);
    this.agent =
      algo === 'sarsa'
        ? new Sarsa(this.env.numStates, NUM_ACTIONS, params, rng)
        : algo === 'dyna'
          ? new DynaQ(this.env.numStates, NUM_ACTIONS, params, rng, planning)
          : new QLearning(this.env.numStates, NUM_ACTIONS, params, rng);
    this.tdMean = new Float32Array(this.env.numStates);
    this.trail = [this.env.reset()];
  }

  /**
   * Runs one agent-environment interaction. Returns a record if the episode ended.
   * After an episode ends the agent and trail still show the final transition;
   * the environment resets at the start of the following call.
   */
  stepOnce(): EpisodeRecord | null {
    if (this.needsReset) {
      this.trail = [this.env.reset()];
      this.needsReset = false;
    }
    const s = this.env.currentState;
    const a = this.pending ?? this.agent.selectAction(s);
    const res = this.env.step(a);

    let tdError: number;
    if (this.agent instanceof Sarsa) {
      // On-policy: pick a' now, learn from it, and then actually take it next step.
      const next = res.terminal ? 0 : this.agent.selectAction(res.state);
      tdError = this.agent.update(s, a, res.reward, res.state, next, res.terminal);
      this.pending = res.done ? null : next;
    } else {
      tdError = (this.agent as QLearning).update(s, a, res.reward, res.state, res.terminal);
    }

    this.lastTdError = tdError;
    this.tdMean[s] += TD_EMA * (Math.abs(tdError) - this.tdMean[s]);
    this.td.n += 1;
    this.td.sumAbs += Math.abs(tdError);
    this.td.sumSigned += tdError;
    this.episodeReward += res.reward;
    this.totalSteps += 1;
    this.trail.push(res.state);
    if (!res.done) return null;

    const record: EpisodeRecord = {
      reward: this.episodeReward,
      steps: this.env.stepCount,
      outcome: res.outcome ?? 'timeout',
    };
    this.episodes += 1;
    this.episodeReward = 0;
    this.needsReset = true;
    return record;
  }

  run(steps: number): EpisodeRecord[] {
    const done: EpisodeRecord[] = [];
    for (let i = 0; i < steps; i++) {
      const r = this.stepOnce();
      if (r) done.push(r);
    }
    return done;
  }

  /** Changes the number of planning updates per real step (Dyna-Q only). */
  setPlanning(n: number): void {
    if (this.agent instanceof DynaQ) this.agent.planningSteps = n;
  }

  /** Returns and clears the TD errors accumulated since the previous call. */
  drainTd(): TdBatch {
    const out = this.td;
    this.td = { n: 0, sumAbs: 0, sumSigned: 0 };
    return out;
  }

  /** Follows the greedy policy without learning. Returns the visited states. */
  greedyRollout(): { states: number[]; outcome: Outcome } {
    const env = new GridEnv(this.env.spec);
    const states = [env.reset()];
    for (;;) {
      const res = env.step(this.agent.greedyAction(env.currentState));
      states.push(res.state);
      if (res.done) return { states, outcome: res.outcome ?? 'timeout' };
    }
  }
}
