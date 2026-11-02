/// Runs training off the main thread and streams Q-table snapshots back.
/// Holds one Trainer per requested algorithm; all advance in lockstep.
import { Trainer } from '../core/trainer';
import type { EpisodeRecord } from '../core/trainer';
import type { AgentSnapshot, Snapshot, ToWorker } from './protocol';

const ctx = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};

const TICK_MS = 16;
const MAX_STEPS_PER_TICK = 20000;

let trainers: Trainer[] = [];
let pending: EpisodeRecord[][] = [];
let gen = 0;
let playing = false;
let stepsPerSecond = 120;
let budget = 0;
let lastTick = performance.now();

function post(): void {
  if (trainers.length === 0) return;
  const transfer: Transferable[] = [];
  const agents: AgentSnapshot[] = trainers.map((t, i) => {
    const q = Float32Array.from(t.agent.q);
    const tdMean = Float32Array.from(t.tdMean);
    const inspect = Float32Array.from(t.agent.inspect);
    transfer.push(q.buffer, tdMean.buffer, inspect.buffer);
    const snap: AgentSnapshot = {
      algo: t.algo,
      q,
      tdMean,
      inspect,
      state: t.env.currentState,
      trail: t.trail.slice(),
      episodes: t.episodes,
      totalSteps: t.totalSteps,
      lastTdError: t.lastTdError,
      finished: pending[i],
      td: t.drainTd(),
    };
    pending[i] = [];
    return snap;
  });
  const msg: Snapshot = { type: 'snapshot', gen, agents };
  ctx.postMessage(msg, transfer);
}

function tick(): void {
  const now = performance.now();
  const dt = now - lastTick;
  lastTick = now;
  if (!playing || trainers.length === 0) return;
  budget += (stepsPerSecond * dt) / 1000;
  const n = Math.min(Math.floor(budget), MAX_STEPS_PER_TICK);
  budget = n === MAX_STEPS_PER_TICK ? 0 : budget - n;
  if (n > 0) {
    trainers.forEach((t, i) => pending[i].push(...t.run(n)));
    post();
  }
}

setInterval(tick, TICK_MS);

ctx.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case 'init':
      gen = m.gen;
      trainers = m.algos.map((algo) => new Trainer(m.spec, m.params, m.seed, algo, m.planning));
      pending = trainers.map(() => []);
      playing = false;
      budget = 0;
      post();
      break;
    case 'play':
      playing = true;
      lastTick = performance.now();
      break;
    case 'pause':
      playing = false;
      break;
    case 'step':
      if (trainers.length) {
        playing = false;
        trainers.forEach((t, i) => {
          const r = t.stepOnce();
          if (r) pending[i].push(r);
        });
        post();
      }
      break;
    case 'params':
      trainers.forEach((t) => t.agent.setParams(m.params));
      break;
    case 'planning':
      trainers.forEach((t) => t.setPlanning(m.n));
      break;
    case 'speed':
      stepsPerSecond = m.stepsPerSecond;
      break;
  }
};
