import { useCallback, useEffect, useRef, useState } from 'react';
import { NUM_ACTIONS, type EnvSpec, type Outcome } from '../core/env';
import type { QParams } from '../core/agent';
import type { Algo } from '../core/trainer';
import type { Snapshot, ToWorker } from '../worker/protocol';

/** Mutable live data for one agent, read by canvases every animation frame (kept out of React state). */
export interface LiveData {
  algo: Algo;
  q: Float32Array;
  /** Per-cell running mean of |TD error|. */
  tdMean: Float32Array;
  state: number;
  trail: number[];
  rewards: number[];
  outcomes: Outcome[];
  /** Index of rewards[0] in the full episode sequence (old entries are trimmed). */
  episodeOffset: number;
  /** TD trace, one point per snapshot batch: mean |TD| and mean signed TD. */
  tdAbs: number[];
  tdSigned: number[];
}

export interface TrainStats {
  episodes: number;
  totalSteps: number;
  lastTdError: number;
  recentReward: number | null;
  recentSuccess: number | null;
}

const KEEP = 50000;
const KEEP_TD = 600;
const RECENT = 50;
const EMPTY_STATS: TrainStats = { episodes: 0, totalSteps: 0, lastTdError: 0, recentReward: null, recentSuccess: null };

function freshLive(spec: EnvSpec, algo: Algo): LiveData {
  const start = spec.cells.indexOf('start');
  const n = spec.width * spec.height;
  return {
    algo,
    q: new Float32Array(n * NUM_ACTIONS),
    tdMean: new Float32Array(n),
    state: start,
    trail: [start],
    rewards: [],
    outcomes: [],
    episodeOffset: 0,
    tdAbs: [],
    tdSigned: [],
  };
}

export function useTrainer(spec: EnvSpec, params: QParams, speed: number, resetNonce: number, algos: Algo[]) {
  const workerRef = useRef<Worker | null>(null);
  const genRef = useRef(0);
  const live = useRef<LiveData[]>(algos.map((a) => freshLive(spec, a)));
  const paramsRef = useRef(params);
  const speedRef = useRef(speed);
  paramsRef.current = params;
  speedRef.current = speed;
  const [playing, setPlaying] = useState(false);
  const [stats, setStats] = useState<TrainStats[]>(algos.map(() => EMPTY_STATS));
  const algoKey = algos.join(',');

  const send = useCallback((m: ToWorker) => workerRef.current?.postMessage(m), []);

  useEffect(() => {
    const worker = new Worker(new URL('../worker/trainer.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.onmessage = (e: MessageEvent<Snapshot>) => {
      const m = e.data;
      if (m.type !== 'snapshot' || m.gen !== genRef.current) return;
      const next: TrainStats[] = [];
      m.agents.forEach((snap, i) => {
        const d = live.current[i];
        if (!d) return;
        d.q = snap.q;
        d.tdMean = snap.tdMean;
        d.state = snap.state;
        d.trail = snap.trail;
        for (const r of snap.finished) {
          d.rewards.push(r.reward);
          d.outcomes.push(r.outcome);
        }
        if (d.rewards.length > KEEP + 10000) {
          const drop = d.rewards.length - KEEP;
          d.rewards.splice(0, drop);
          d.outcomes.splice(0, drop);
          d.episodeOffset += drop;
        }
        if (snap.td.n > 0) {
          d.tdAbs.push(snap.td.sumAbs / snap.td.n);
          d.tdSigned.push(snap.td.sumSigned / snap.td.n);
          if (d.tdAbs.length > KEEP_TD) {
            d.tdAbs.shift();
            d.tdSigned.shift();
          }
        }
        const n = d.rewards.length;
        const k = Math.min(RECENT, n);
        let sum = 0, wins = 0;
        for (let j = n - k; j < n; j++) {
          sum += d.rewards[j];
          if (d.outcomes[j] === 'goal') wins++;
        }
        next.push({
          episodes: snap.episodes,
          totalSteps: snap.totalSteps,
          lastTdError: snap.lastTdError,
          recentReward: k ? sum / k : null,
          recentSuccess: k ? wins / k : null,
        });
      });
      setStats(next);
    };
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // (Re)initialise whenever the environment or agent set changes, or Reset is pressed.
  useEffect(() => {
    genRef.current += 1;
    live.current = algos.map((a) => freshLive(spec, a));
    setPlaying(false);
    setStats(algos.map(() => EMPTY_STATS));
    send({ type: 'init', gen: genRef.current, spec, params: paramsRef.current, seed: (Date.now() & 0xffff) + 1, algos });
    send({ type: 'speed', stepsPerSecond: speedRef.current });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, resetNonce, send, algoKey]);

  useEffect(() => send({ type: 'params', params }), [params, send]);
  useEffect(() => send({ type: 'speed', stepsPerSecond: speed }), [speed, send]);

  const play = useCallback(() => { setPlaying(true); send({ type: 'play' }); }, [send]);
  const pause = useCallback(() => { setPlaying(false); send({ type: 'pause' }); }, [send]);
  const step = useCallback(() => { setPlaying(false); send({ type: 'step' }); }, [send]);

  return { live, stats, playing, play, pause, step };
}
