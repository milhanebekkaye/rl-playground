/** Shared base for tabular TD-control agents: a Q-table plus epsilon-greedy action selection. */
import { INSPECT_STRIDE, readInspect, type LastUpdate } from './inspect';
import type { Rng } from './rng';

export interface QParams {
  alpha: number;
  gamma: number;
  epsilon: number;
}

export class TabularAgent {
  /** Row-major Q-table: q[s * numActions + a]. Initialised to zero. */
  readonly q: Float64Array;
  /** Bellman-inspector records, see inspect.ts. */
  readonly inspect: Float64Array;
  params: QParams;

  constructor(
    readonly numStates: number,
    readonly numActions: number,
    params: QParams,
    protected readonly rng: Rng,
  ) {
    this.q = new Float64Array(numStates * numActions);
    this.inspect = new Float64Array(numStates * numActions * INSPECT_STRIDE);
    this.params = { ...params };
  }

  setParams(p: Partial<QParams>): void {
    this.params = { ...this.params, ...p };
  }

  value(s: number, a: number): number {
    return this.q[s * this.numActions + a];
  }

  maxQ(s: number): number {
    let best = -Infinity;
    for (let a = 0; a < this.numActions; a++) best = Math.max(best, this.value(s, a));
    return best;
  }

  /** Greedy action, breaking ties uniformly at random. */
  greedyAction(s: number): number {
    const best = this.maxQ(s);
    const ties: number[] = [];
    for (let a = 0; a < this.numActions; a++) if (this.value(s, a) === best) ties.push(a);
    return ties[Math.floor(this.rng() * ties.length)];
  }

  /** With probability epsilon explore uniformly, otherwise act greedily. */
  selectAction(s: number): number {
    if (this.rng() < this.params.epsilon) return Math.floor(this.rng() * this.numActions);
    return this.greedyAction(s);
  }

  /** Number of real updates applied to (s, a). */
  visits(s: number, a: number): number {
    return readInspect(this.inspect, s, a, this.numActions).visits;
  }

  /** The most recent real update of (s, a), or null. */
  lastUpdate(s: number, a: number): LastUpdate | null {
    return readInspect(this.inspect, s, a, this.numActions).last;
  }

  /** Records a real update for the inspector. */
  protected record(
    s: number, a: number, reward: number, next: number, bootstrap: number,
    tdError: number, terminal: boolean, qBefore: number,
  ): void {
    const o = (s * this.numActions + a) * INSPECT_STRIDE;
    const i = this.inspect;
    i[o] += 1;
    i[o + 1] = reward;
    i[o + 2] = next;
    i[o + 3] = bootstrap;
    i[o + 4] = tdError;
    i[o + 5] = terminal ? 1 : 0;
    i[o + 6] = qBefore;
    i[o + 7] = this.params.gamma;
  }
}
