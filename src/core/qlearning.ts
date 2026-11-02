/**
 * Tabular Q-learning (off-policy TD control). Pure TypeScript.
 *
 *   target   = r + gamma * max_a' Q(s', a')      (just r if s' is terminal)
 *   tdError  = target - Q(s, a)
 *   Q(s, a) <- Q(s, a) + alpha * tdError
 */
import { TabularAgent } from './agent';
export type { QParams } from './agent';

export class QLearning extends TabularAgent {
  /** Applies one real Q-learning update (recorded for the inspector) and returns the TD error. */
  update(s: number, a: number, reward: number, next: number, terminal: boolean): number {
    return this.apply(s, a, reward, next, terminal, true);
  }

  protected apply(s: number, a: number, reward: number, next: number, terminal: boolean, record: boolean): number {
    const { alpha, gamma } = this.params;
    const qBefore = this.value(s, a);
    const bootstrap = terminal ? 0 : this.maxQ(next);
    const tdError = reward + gamma * bootstrap - qBefore;
    this.q[s * this.numActions + a] = qBefore + alpha * tdError;
    if (record) this.record(s, a, reward, next, bootstrap, tdError, terminal, qBefore);
    return tdError;
  }
}
