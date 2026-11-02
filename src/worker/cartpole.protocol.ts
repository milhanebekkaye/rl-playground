import type { CartState } from '../core/cartpole';
import type { DqnConfig } from '../core/dqn';

export type CartToWorker =
  | { type: 'init'; gen: number; seed: number; cfg: Partial<DqnConfig> }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'config'; cfg: Partial<DqnConfig> }
  | { type: 'speed'; stepsPerSecond: number }
  | { type: 'greedy'; gen: number; seed: number };

export interface CartSnapshot {
  type: 'snapshot';
  gen: number;
  state: CartState;
  /** Online-network Q-values of [push left, push right] for the current training state. */
  q: number[];
  epsilon: number;
  bufferSize: number;
  steps: number;
  episodes: number;
  lastLoss: number;
  /** Returns of episodes finished since the previous snapshot. */
  finished: number[];
}

export interface CartGreedy {
  type: 'greedy';
  gen: number;
  states: CartState[];
  q: number[][];
  ret: number;
}

export type CartFromWorker = CartSnapshot | CartGreedy;
