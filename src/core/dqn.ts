/**
 * Deep Q-Network for small continuous-state tasks. Pure TypeScript, built on
 * the hand-written MLP in nn.ts.
 *
 * Standard ingredients: epsilon-greedy exploration with decay, a replay buffer,
 * a separate target network that is synced every `targetSync` environment steps,
 * Huber loss on the TD error, Adam, and gradient-norm clipping.
 *
 * Target for a sampled transition (s, a, r, s', terminated):
 *   y = r                                   if s' is terminal
 *   y = r + gamma * max_a' Q_target(s', a') otherwise
 * and the network is trained to move Q(s, a) toward y.
 */
import { MLP, huberGrad } from './nn';
import type { Rng } from './rng';

export interface DqnConfig {
  stateSize: number;
  numActions: number;
  hidden: number[];
  lr: number;
  gamma: number;
  batchSize: number;
  bufferSize: number;
  /** Transitions to collect before learning starts. */
  warmup: number;
  /** Hard-copy the online network into the target network every this many steps. */
  targetSync: number;
  epsStart: number;
  epsEnd: number;
  /** Multiplicative per-step decay of epsilon (until epsEnd). */
  epsDecay: number;
  gradClip: number;
  /** Run one gradient update every this many environment steps. */
  trainEvery: number;
}

export const DEFAULT_DQN: DqnConfig = {
  stateSize: 4,
  numActions: 2,
  hidden: [32, 32],
  lr: 5e-4,
  gamma: 0.99,
  batchSize: 64,
  bufferSize: 20000,
  warmup: 500,
  targetSync: 500,
  epsStart: 1,
  epsEnd: 0.05,
  epsDecay: 0.9993,
  gradClip: 10,
  trainEvery: 1,
};

export class ReplayBuffer {
  readonly s: Float64Array;
  readonly s2: Float64Array;
  readonly a: Int32Array;
  readonly r: Float64Array;
  readonly done: Uint8Array;
  size = 0;
  private head = 0;

  constructor(readonly capacity: number, readonly stateSize: number) {
    this.s = new Float64Array(capacity * stateSize);
    this.s2 = new Float64Array(capacity * stateSize);
    this.a = new Int32Array(capacity);
    this.r = new Float64Array(capacity);
    this.done = new Uint8Array(capacity);
  }

  push(s: ArrayLike<number>, a: number, r: number, s2: ArrayLike<number>, done: boolean): void {
    const k = this.stateSize, i = this.head;
    for (let j = 0; j < k; j++) { this.s[i * k + j] = s[j]; this.s2[i * k + j] = s2[j]; }
    this.a[i] = a; this.r[i] = r; this.done[i] = done ? 1 : 0;
    this.head = (this.head + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }
}

export class DqnAgent {
  readonly online: MLP;
  readonly target: MLP;
  readonly buffer: ReplayBuffer;
  cfg: DqnConfig;
  epsilon: number;
  steps = 0;
  lastLoss = 0;
  private readonly actsOnline: Float64Array[];
  private readonly actsTarget: Float64Array[];
  private readonly actsEval: Float64Array[];
  private readonly dOut: Float64Array;

  constructor(cfg: DqnConfig, private readonly rng: Rng) {
    this.cfg = { ...cfg };
    const sizes = [cfg.stateSize, ...cfg.hidden, cfg.numActions];
    this.online = new MLP(sizes, rng);
    this.target = new MLP(sizes, rng);
    this.target.copyFrom(this.online);
    this.buffer = new ReplayBuffer(cfg.bufferSize, cfg.stateSize);
    this.epsilon = cfg.epsStart;
    this.actsOnline = this.online.makeActivations();
    this.actsTarget = this.target.makeActivations();
    this.actsEval = this.online.makeActivations();
    this.dOut = new Float64Array(cfg.numActions);
  }

  setConfig(p: Partial<DqnConfig>): void {
    this.cfg = { ...this.cfg, ...p };
  }

  /** Q-values of the online network for one (already normalised) state. Returns a fresh copy. */
  qValues(state: ArrayLike<number>): number[] {
    return Array.from(this.online.forward(state, this.actsEval));
  }

  greedyAction(state: ArrayLike<number>): number {
    const q = this.online.forward(state, this.actsEval);
    let best = 0;
    for (let a = 1; a < q.length; a++) if (q[a] > q[best]) best = a;
    return best;
  }

  selectAction(state: ArrayLike<number>): number {
    if (this.rng() < this.epsilon) return Math.floor(this.rng() * this.cfg.numActions);
    return this.greedyAction(state);
  }

  /** Stores a transition, decays epsilon, trains and syncs the target network as scheduled. */
  observe(s: ArrayLike<number>, a: number, r: number, s2: ArrayLike<number>, terminated: boolean): void {
    this.buffer.push(s, a, r, s2, terminated);
    this.steps += 1;
    this.epsilon = Math.max(this.cfg.epsEnd, this.epsilon * this.cfg.epsDecay);
    if (this.buffer.size >= this.cfg.warmup && this.steps % this.cfg.trainEvery === 0) this.trainStep();
    if (this.steps % this.cfg.targetSync === 0) this.target.copyFrom(this.online);
  }

  /** One minibatch gradient step. Returns the mean Huber loss of the batch. */
  trainStep(): number {
    const { batchSize, gamma, lr, gradClip, numActions, stateSize } = this.cfg;
    const buf = this.buffer;
    const net = this.online;
    net.zeroGrad();
    let loss = 0;
    const x = new Float64Array(stateSize);
    for (let n = 0; n < batchSize; n++) {
      const i = Math.floor(this.rng() * buf.size);
      for (let j = 0; j < stateSize; j++) x[j] = buf.s[i * stateSize + j];
      let y = buf.r[i];
      if (!buf.done[i]) {
        const q2 = this.target.forward(buf.s2.subarray(i * stateSize, (i + 1) * stateSize), this.actsTarget);
        let m = -Infinity;
        for (let a = 0; a < numActions; a++) m = Math.max(m, q2[a]);
        y += gamma * m;
      }
      const q = net.forward(x, this.actsOnline);
      const a = buf.a[i];
      const e = q[a] - y;
      loss += Math.abs(e) <= 1 ? 0.5 * e * e : Math.abs(e) - 0.5;
      this.dOut.fill(0);
      this.dOut[a] = huberGrad(e);
      net.backward(x, this.actsOnline, this.dOut);
    }
    net.clipGradNorm(gradClip * batchSize);
    net.adamStep(lr, 1 / batchSize);
    this.lastLoss = loss / batchSize;
    return this.lastLoss;
  }
}
