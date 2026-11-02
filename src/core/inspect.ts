/**
 * Per-(state, action) bookkeeping for the Bellman inspector. Agents write one
 * fixed-width record per (s, a) into a flat Float64Array so it can be copied
 * to the UI thread cheaply.
 *
 * Record layout (INSPECT_STRIDE numbers):
 *   0 visits      number of real updates of this (s, a)
 *   1 reward      r from the last real update
 *   2 next        s' from the last real update
 *   3 bootstrap   the value used for the future: max_a' Q(s', a') for Q-learning,
 *                 Q(s', a') for SARSA (0 when terminal)
 *   4 tdError     delta = r + gamma * bootstrap - Q(s, a)
 *   5 terminal    1 if s' was a goal/trap
 *   6 qBefore     Q(s, a) just before the last update
 *   7 gamma       discount used in the last update
 */
export const INSPECT_STRIDE = 8;

export interface LastUpdate {
  visits: number;
  reward: number;
  next: number;
  bootstrap: number;
  tdError: number;
  terminal: boolean;
  qBefore: number;
  gamma: number;
}

/** Reads the record for (s, a); `last` is null if (s, a) has never been updated. */
export function readInspect(
  buf: ArrayLike<number>,
  s: number,
  a: number,
  numActions = 4,
): { visits: number; last: LastUpdate | null } {
  const o = (s * numActions + a) * INSPECT_STRIDE;
  const visits = buf[o];
  if (!(visits > 0)) return { visits: 0, last: null };
  return {
    visits,
    last: {
      visits,
      reward: buf[o + 1],
      next: buf[o + 2],
      bootstrap: buf[o + 3],
      tdError: buf[o + 4],
      terminal: buf[o + 5] === 1,
      qBefore: buf[o + 6],
      gamma: buf[o + 7],
    },
  };
}
