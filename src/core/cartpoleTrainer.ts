/**
 * Couples CartPole with a DQN agent and runs it one environment step at a time.
 * UI-free so it runs in a Web Worker and in Node tests.
 */
import { CartPole, normalise, type CartState } from './cartpole';
import { DEFAULT_DQN, DqnAgent, type DqnConfig } from './dqn';
import { mulberry32 } from './rng';

export class CartPoleTrainer {
  readonly env: CartPole;
  readonly agent: DqnAgent;
  episodes = 0;
  totalSteps = 0;
  lastReturn = 0;
  private epReturn = 0;
  private obs: Float64Array;
  private readonly obs2 = new Float64Array(4);
  private readonly seed: number;

  constructor(seed = 1, cfg: Partial<DqnConfig> = {}) {
    this.seed = seed;
    this.env = new CartPole(mulberry32(seed * 7919 + 1));
    this.agent = new DqnAgent({ ...DEFAULT_DQN, ...cfg }, mulberry32(seed * 104729 + 3));
    this.obs = normalise(this.env.state, new Float64Array(4));
  }

  get state(): CartState {
    return this.env.state;
  }

  /** One environment step with learning. Returns the finished episode's return, or null. */
  step(): number | null {
    const a = this.agent.selectAction(this.obs);
    const r = this.env.step(a);
    normalise(r.state, this.obs2);
    // Only a real failure is terminal for bootstrapping; hitting the step cap is not.
    this.agent.observe(this.obs, a, r.reward, this.obs2, r.terminated);
    this.totalSteps += 1;
    this.epReturn += r.reward;
    this.obs.set(this.obs2);
    if (r.terminated || r.truncated) {
      const ret = this.epReturn;
      this.lastReturn = ret;
      this.episodes += 1;
      this.epReturn = 0;
      normalise(this.env.reset(), this.obs);
      return ret;
    }
    return null;
  }

  /** Runs up to `n` steps; returns returns of episodes finished. */
  run(n: number): number[] {
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      const r = this.step();
      if (r !== null) out.push(r);
    }
    return out;
  }

  /**
   * Plays one greedy (epsilon = 0) episode in a separate environment and
   * returns the visited states and Q-values, without touching training state.
   */
  greedyEpisode(episodeSeed: number): { states: CartState[]; q: number[][]; ret: number } {
    const env = new CartPole(mulberry32(episodeSeed * 15485863 + this.seed));
    const states: CartState[] = [env.state.slice() as CartState];
    const qs: number[][] = [];
    const obs = new Float64Array(4);
    let ret = 0;
    for (;;) {
      normalise(env.state, obs);
      qs.push(this.agent.qValues(obs));
      const r = env.step(this.agent.greedyAction(obs));
      ret += r.reward;
      states.push(r.state);
      if (r.terminated || r.truncated) break;
    }
    return { states, q: qs, ret };
  }
}

/** Mean greedy return over `n` evaluation episodes. */
export function evaluate(t: CartPoleTrainer, n = 10): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += t.greedyEpisode(1000 + i).ret;
  return s / n;
}
