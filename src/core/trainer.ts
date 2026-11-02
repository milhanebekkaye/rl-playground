/**
 * Couples an environment with a Q-learning agent and runs it one step at a
 * time, tracking per-episode statistics. UI-free so it runs in a Web Worker
 * and in unit tests alike.
 */
import { GridEnv, NUM_ACTIONS, type EnvSpec, type Outcome } from './env';
import { QLearning, type QParams } from './qlearning';
import { mulberry32 } from './rng';

export interface EpisodeRecord {
  reward: number;
  steps: number;
  outcome: Outcome;
}

export class Trainer {
  readonly env: GridEnv;
  readonly agent: QLearning;
  trail: number[];
  episodes = 0;
  totalSteps = 0;
  lastTdError = 0;
  private episodeReward = 0;

  constructor(spec: EnvSpec, params: QParams, seed = 1) {
    this.env = new GridEnv(spec);
    this.agent = new QLearning(this.env.numStates, NUM_ACTIONS, params, mulberry32(seed));
    this.trail = [this.env.reset()];
  }

  /** Runs one agent-environment interaction. Returns a record if the episode ended. */
  stepOnce(): EpisodeRecord | null {
    const s = this.env.currentState;
    const a = this.agent.selectAction(s);
    const res = this.env.step(a);
    this.lastTdError = this.agent.update(s, a, res.reward, res.state, res.terminal);
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
    this.trail = [this.env.reset()];
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
