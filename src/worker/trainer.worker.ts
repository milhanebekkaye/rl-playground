/// Runs training off the main thread and streams Q-table snapshots back.
import { Trainer } from '../core/trainer';
import type { Snapshot, ToWorker } from './protocol';

const ctx = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};

const TICK_MS = 16;
const MAX_STEPS_PER_TICK = 20000;

let trainer: Trainer | null = null;
let gen = 0;
let playing = false;
let stepsPerSecond = 120;
let budget = 0;
let lastTick = performance.now();
let pending: Snapshot['finished'] = [];

function post(): void {
  if (!trainer) return;
  const q = Float32Array.from(trainer.agent.q);
  const msg: Snapshot = {
    type: 'snapshot',
    gen,
    q,
    state: trainer.env.currentState,
    trail: trainer.trail.slice(),
    episodes: trainer.episodes,
    totalSteps: trainer.totalSteps,
    lastTdError: trainer.lastTdError,
    finished: pending,
  };
  pending = [];
  ctx.postMessage(msg, [q.buffer]);
}

function tick(): void {
  const now = performance.now();
  const dt = now - lastTick;
  lastTick = now;
  if (!playing || !trainer) return;
  budget += (stepsPerSecond * dt) / 1000;
  const n = Math.min(Math.floor(budget), MAX_STEPS_PER_TICK);
  budget = n === MAX_STEPS_PER_TICK ? 0 : budget - n;
  if (n > 0) {
    pending.push(...trainer.run(n));
    post();
  }
}

setInterval(tick, TICK_MS);

ctx.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case 'init':
      gen = m.gen;
      trainer = new Trainer(m.spec, m.params, m.seed);
      playing = false;
      budget = 0;
      pending = [];
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
      if (trainer) {
        playing = false;
        const r = trainer.stepOnce();
        if (r) pending.push(r);
        post();
      }
      break;
    case 'params':
      trainer?.agent.setParams(m.params);
      break;
    case 'speed':
      stepsPerSecond = m.stepsPerSecond;
      break;
  }
};
