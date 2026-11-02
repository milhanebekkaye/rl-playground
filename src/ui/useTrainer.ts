import { useCallback, useEffect, useRef, useState } from 'react';
import { NUM_ACTIONS, type EnvSpec, type Outcome } from '../core/env';
import type { QParams } from '../core/qlearning';
import type { Snapshot, ToWorker } from '../worker/protocol';

/** Mutable live data read by canvases every animation frame (kept out of React state). */
export interface LiveData {
  q: Float32Array;
  state: number;
  trail: number[];
  rewards: number[];
  outcomes: Outcome[];
  /** Index of rewards[0] in the full episode sequence (old entries are trimmed). */
  episodeOffset: number;
}

export interface TrainStats {
  episodes: number;
  totalSteps: number;
  lastTdError: number;
  recentReward: number | null;
  recentSuccess: number | null;
}

const KEEP = 50000;
const RECENT = 50;

function freshLive(spec: EnvSpec): LiveData {
  return {
    q: new Float32Array(spec.width * spec.height * NUM_ACTIONS),
    state: spec.cells.indexOf('start'),
    trail: [spec.cells.indexOf('start')],
    rewards: [],
    outcomes: [],
    episodeOffset: 0,
  };
}

export function useTrainer(spec: EnvSpec, params: QParams, speed: number, resetNonce: number) {
  const workerRef = useRef<Worker | null>(null);
  const genRef = useRef(0);
  const live = useRef<LiveData>(freshLive(spec));
  const paramsRef = useRef(params);
  const speedRef = useRef(speed);
  paramsRef.current = params;
  speedRef.current = speed;
  const [playing, setPlaying] = useState(false);
  const [stats, setStats] = useState<TrainStats>({
    episodes: 0, totalSteps: 0, lastTdError: 0, recentReward: null, recentSuccess: null,
  });

  const send = useCallback((m: ToWorker) => workerRef.current?.postMessage(m), []);

  useEffect(() => {
    const worker = new Worker(new URL('../worker/trainer.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.onmessage = (e: MessageEvent<Snapshot>) => {
      const m = e.data;
      if (m.type !== 'snapshot' || m.gen !== genRef.current) return;
      const d = live.current;
      d.q = m.q;
      d.state = m.state;
      d.trail = m.trail;
      for (const r of m.finished) {
        d.rewards.push(r.reward);
        d.outcomes.push(r.outcome);
      }
      if (d.rewards.length > KEEP + 10000) {
        const drop = d.rewards.length - KEEP;
        d.rewards.splice(0, drop);
        d.outcomes.splice(0, drop);
        d.episodeOffset += drop;
      }
      const n = d.rewards.length;
      const k = Math.min(RECENT, n);
      let sum = 0, wins = 0;
      for (let i = n - k; i < n; i++) {
        sum += d.rewards[i];
        if (d.outcomes[i] === 'goal') wins++;
      }
      setStats({
        episodes: m.episodes,
        totalSteps: m.totalSteps,
        lastTdError: m.lastTdError,
        recentReward: k ? sum / k : null,
        recentSuccess: k ? wins / k : null,
      });
    };
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // (Re)initialise whenever the environment changes or Reset is pressed.
  useEffect(() => {
    genRef.current += 1;
    live.current = freshLive(spec);
    setPlaying(false);
    setStats({ episodes: 0, totalSteps: 0, lastTdError: 0, recentReward: null, recentSuccess: null });
    send({ type: 'init', gen: genRef.current, spec, params: paramsRef.current, seed: (Date.now() & 0xffff) + 1 });
    send({ type: 'speed', stepsPerSecond: speedRef.current });
  }, [spec, resetNonce, send]);

  useEffect(() => send({ type: 'params', params }), [params, send]);
  useEffect(() => send({ type: 'speed', stepsPerSecond: speed }), [speed, send]);

  const play = useCallback(() => { setPlaying(true); send({ type: 'play' }); }, [send]);
  const pause = useCallback(() => { setPlaying(false); send({ type: 'pause' }); }, [send]);
  const step = useCallback(() => { setPlaying(false); send({ type: 'step' }); }, [send]);

  return { live, stats, playing, play, pause, step };
}
