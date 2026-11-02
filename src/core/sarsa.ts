/**
 * Tabular SARSA (on-policy TD control). Pure TypeScript.
 *
 * Identical to Q-learning except the target uses the action a' the agent will
 * actually take next (chosen by its epsilon-greedy policy), not the best one:
 *
 *   target   = r + gamma * Q(s', a')             (just r if s' is terminal)
 *   tdError  = target - Q(s, a)
 *   Q(s, a) <- Q(s, a) + alpha * tdError
 *
 * Because exploration is baked into the values it learns, SARSA prices in the
 * risk of its own random slips; Q-learning learns values for the greedy policy.
 */
import { TabularAgent } from './agent';

export class Sarsa extends TabularAgent {
  /** Applies one SARSA update and returns the TD error. `nextAction` is ignored when terminal. */
  update(s: number, a: number, reward: number, next: number, nextAction: number, terminal: boolean): number {
    const { alpha, gamma } = this.params;
    const qBefore = this.value(s, a);
    const bootstrap = terminal ? 0 : this.value(next, nextAction);
    const tdError = reward + gamma * bootstrap - qBefore;
    this.q[s * this.numActions + a] = qBefore + alpha * tdError;
    this.record(s, a, reward, next, bootstrap, tdError, terminal, qBefore);
    return tdError;
  }
}
