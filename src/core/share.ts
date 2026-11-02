/**
 * Shareable-maze encoding for the URL hash. Pure TypeScript.
 *
 * Plain, readable, versioned format (no base64, safe inside embedded pages):
 *
 *   #v1&w=8&h=8&c=<cells>&r=1,-1,-0.04,200&p=0.3,0.95,0.15&m=q
 *
 *   c  one char per cell, row-major: _ empty, w wall, S start, G goal, T trap
 *   r  goal reward, trap reward, step reward, max steps
 *   p  alpha, gamma, epsilon
 *   m  q (Q-learning), s (SARSA) or cmp (side-by-side comparison)
 *
 * decodeShare() validates everything and returns null on any problem so the
 * caller can fall back to the default maze.
 */
import type { CellType, EnvSpec } from './env';
import { validateSpec } from './env';
import type { QParams } from './agent';
import type { Algo } from './trainer';

export const SHARE_VERSION = 'v1';
const MIN_DIM = 2;
const MAX_DIM = 24;
const MAX_REWARD = 1000;

export interface ShareState {
  spec: EnvSpec;
  params: QParams;
  compare: boolean;
  /** Algorithm for single-agent mode. */
  algo: Algo;
}

const CHAR_OF: Record<CellType, string> = { empty: '_', wall: 'w', start: 'S', goal: 'G', trap: 'T' };
const CELL_OF: Record<string, CellType> = { _: 'empty', w: 'wall', S: 'start', G: 'goal', T: 'trap' };

const num = (n: number) => String(Math.round(n * 10000) / 10000);

export function encodeShare(st: ShareState): string {
  const { spec, params } = st;
  const cells = spec.cells.map((c) => CHAR_OF[c]).join('');
  const m = st.compare ? 'cmp' : st.algo === 'sarsa' ? 's' : 'q';
  return [
    SHARE_VERSION,
    `w=${spec.width}`,
    `h=${spec.height}`,
    `c=${cells}`,
    `r=${[spec.goalReward, spec.trapReward, spec.stepReward, spec.maxSteps].map(num).join(',')}`,
    `p=${[params.alpha, params.gamma, params.epsilon].map(num).join(',')}`,
    `m=${m}`,
  ].join('&');
}

function parseNumbers(text: string | undefined, count: number): number[] | null {
  if (text === undefined) return null;
  const parts = text.split(',');
  if (parts.length !== count) return null;
  const out: number[] = [];
  for (const p of parts) {
    if (!/^-?\d+(\.\d+)?(e-?\d+)?$/i.test(p)) return null;
    const v = Number(p);
    if (!Number.isFinite(v)) return null;
    out.push(v);
  }
  return out;
}

/** Parses a hash (with or without the leading '#'). Returns null if invalid. */
export function decodeShare(hash: string): ShareState | null {
  try {
    const raw = hash.startsWith('#') ? hash.slice(1) : hash;
    if (raw.length === 0 || raw.length > 2000) return null;
    const [version, ...rest] = raw.split('&');
    if (version !== SHARE_VERSION) return null;
    const kv = new Map<string, string>();
    for (const part of rest) {
      const i = part.indexOf('=');
      if (i <= 0) return null;
      const k = part.slice(0, i);
      if (kv.has(k)) return null;
      kv.set(k, part.slice(i + 1));
    }

    const dims = parseNumbers(`${kv.get('w')},${kv.get('h')}`, 2);
    if (!dims) return null;
    const [width, height] = dims;
    if (!Number.isInteger(width) || !Number.isInteger(height)) return null;
    if (width < MIN_DIM || width > MAX_DIM || height < MIN_DIM || height > MAX_DIM) return null;

    const c = kv.get('c');
    if (c === undefined || c.length !== width * height) return null;
    const cells: CellType[] = [];
    for (const ch of c) {
      const t = CELL_OF[ch];
      if (!t) return null;
      cells.push(t);
    }

    const r = parseNumbers(kv.get('r'), 4);
    const p = parseNumbers(kv.get('p'), 3);
    if (!r || !p) return null;
    const [goalReward, trapReward, stepReward, maxSteps] = r;
    if ([goalReward, trapReward, stepReward].some((v) => Math.abs(v) > MAX_REWARD)) return null;
    if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 5000) return null;
    const [alpha, gamma, epsilon] = p;
    if ([alpha, gamma, epsilon].some((v) => v < 0 || v > 1)) return null;

    const m = kv.get('m');
    if (m !== 'q' && m !== 's' && m !== 'cmp') return null;

    const spec: EnvSpec = { width, height, cells, goalReward, trapReward, stepReward, maxSteps };
    if (validateSpec(spec) !== null) return null;
    return { spec, params: { alpha, gamma, epsilon }, compare: m === 'cmp', algo: m === 's' ? 'sarsa' : 'qlearning' };
  } catch {
    return null;
  }
}
