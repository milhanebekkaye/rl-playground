/// Trains the CartPole DQN off the main thread and streams snapshots back.
import { normalise } from '../core/cartpole';
import { CartPoleTrainer } from '../core/cartpoleTrainer';
import type { CartGreedy, CartSnapshot, CartToWorker } from './cartpole.protocol';

const ctx = self as unknown as {
  postMessage(message: unknown): void;
  onmessage: ((e: MessageEvent<CartToWorker>) => void) | null;
};

const TICK_MS = 16;
/** Wall-clock budget per tick so the worker stays responsive to messages. */
const MAX_WORK_MS = 12;

let trainer: CartPoleTrainer | null = null;
let gen = 0;
let playing = false;
let stepsPerSecond = 600;
let budget = 0;
let lastTick = performance.now();
let finished: number[] = [];

function post(): void {
  if (!trainer) return;
  const t = trainer;
  const msg: CartSnapshot = {
    type: 'snapshot',
    gen,
    state: t.state.slice() as CartSnapshot['state'],
    q: t.agent.qValues(normalise(t.state)),
    epsilon: t.agent.epsilon,
    bufferSize: t.agent.buffer.size,
    steps: t.totalSteps,
    episodes: t.episodes,
    lastLoss: t.agent.lastLoss,
    finished,
  };
  finished = [];
  ctx.postMessage(msg);
}

function tick(): void {
  const now = performance.now();
  const dt = now - lastTick;
  lastTick = now;
  if (!playing || !trainer) return;
  budget = Math.min(budget + (stepsPerSecond * dt) / 1000, stepsPerSecond * 0.25 + 50);
  const started = performance.now();
  while (budget >= 1 && performance.now() - started < MAX_WORK_MS) {
    const batch = Math.min(Math.floor(budget), 20);
    for (let i = 0; i < batch; i++) {
      const r = trainer.step();
      if (r !== null) finished.push(r);
    }
    budget -= batch;
  }
  post();
}

setInterval(tick, TICK_MS);

ctx.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case 'init':
      gen = m.gen;
      trainer = new CartPoleTrainer(m.seed, m.cfg);
      playing = false;
      budget = 0;
      finished = [];
      post();
      break;
    case 'play':
      playing = true;
      lastTick = performance.now();
      break;
    case 'pause':
      playing = false;
      break;
    case 'config':
      trainer?.agent.setConfig(m.cfg);
      break;
    case 'speed':
      stepsPerSecond = m.stepsPerSecond;
      break;
    case 'greedy':
      if (trainer) {
        const g = trainer.greedyEpisode(m.seed);
        const out: CartGreedy = { type: 'greedy', gen: m.gen, states: g.states, q: g.q, ret: g.ret };
        ctx.postMessage(out);
      }
      break;
  }
};
