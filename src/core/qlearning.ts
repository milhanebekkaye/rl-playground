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
  /** Applies one Q-learning update and returns the TD error. */
  update(s: number, a: number, reward: number, next: number, terminal: boolean): number {
    const { alpha, gamma } = this.params;
    const target = reward + (terminal ? 0 : gamma * this.maxQ(next));
    const tdError = target - this.value(s, a);
    this.q[s * this.numActions + a] += alpha * tdError;
    return tdError;
  }
}
