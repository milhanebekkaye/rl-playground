import type { EnvSpec } from '../core/env';
import type { QParams } from '../core/qlearning';
import type { EpisodeRecord } from '../core/trainer';

export type ToWorker =
  | { type: 'init'; gen: number; spec: EnvSpec; params: QParams; seed: number }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'step' }
  | { type: 'params'; params: QParams }
  | { type: 'speed'; stepsPerSecond: number };

export interface Snapshot {
  type: 'snapshot';
  gen: number;
  /** Q-table copy, row-major [state * 4 + action]. */
  q: Float32Array;
  /** Agent's current state and the states visited in the current episode. */
  state: number;
  trail: number[];
  episodes: number;
  totalSteps: number;
  lastTdError: number;
  /** Episodes finished since the previous snapshot. */
  finished: EpisodeRecord[];
}
