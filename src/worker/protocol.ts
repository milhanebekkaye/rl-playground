import type { EnvSpec } from '../core/env';
import type { QParams } from '../core/agent';
import type { Algo, EpisodeRecord, TdBatch } from '../core/trainer';

export type ToWorker =
  /** One trainer per entry in `algos`; all share the spec, params and seed. */
  | { type: 'init'; gen: number; spec: EnvSpec; params: QParams; seed: number; algos: Algo[] }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'step' }
  | { type: 'params'; params: QParams }
  | { type: 'speed'; stepsPerSecond: number };

export interface AgentSnapshot {
  algo: Algo;
  /** Q-table copy, row-major [state * 4 + action]. */
  q: Float32Array;
  /** Per-cell running mean of |TD error|. */
  tdMean: Float32Array;
  /** Agent's current state and the states visited in the current episode. */
  state: number;
  trail: number[];
  episodes: number;
  totalSteps: number;
  lastTdError: number;
  /** Episodes finished since the previous snapshot. */
  finished: EpisodeRecord[];
  /** TD errors accumulated since the previous snapshot. */
  td: TdBatch;
}

export interface Snapshot {
  type: 'snapshot';
  gen: number;
  agents: AgentSnapshot[];
}
