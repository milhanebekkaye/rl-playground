/**
 * CartPole (the classic Barto, Sutton and Anderson control task) with the
 * standard parameters used by common reference implementations. Pure TypeScript.
 *
 * State: [x, xDot, theta, thetaDot]. Actions: 0 = push left, 1 = push right.
 * Reward: +1 for every step the pole is still up. The episode terminates when
 * |theta| > 12 degrees or |x| > 2.4, and is truncated at 500 steps.
 */
import type { Rng } from './rng';

export const GRAVITY = 9.8;
export const CART_MASS = 1.0;
export const POLE_MASS = 0.1;
export const POLE_HALF_LENGTH = 0.5;
export const FORCE = 10.0;
export const TAU = 0.02;
export const THETA_LIMIT = (12 * Math.PI) / 180;
export const X_LIMIT = 2.4;
export const MAX_STEPS = 500;

const TOTAL_MASS = CART_MASS + POLE_MASS;
const POLE_MASS_LENGTH = POLE_MASS * POLE_HALF_LENGTH;

export type CartState = [number, number, number, number];

export interface CartStep {
  state: CartState;
  reward: number;
  /** The pole fell or the cart left the track. */
  terminated: boolean;
  /** Hit the step cap while still balanced. */
  truncated: boolean;
}

export class CartPole {
  state: CartState = [0, 0, 0, 0];
  steps = 0;

  constructor(private readonly rng: Rng) {
    this.reset();
  }

  /** Each component starts uniformly in [-0.05, 0.05]. */
  reset(): CartState {
    const u = () => (this.rng() - 0.5) * 0.1;
    this.state = [u(), u(), u(), u()];
    this.steps = 0;
    return this.state.slice() as CartState;
  }

  step(action: number): CartStep {
    const force = action === 1 ? FORCE : -FORCE;
    const [x, xDot, theta, thetaDot] = this.state;
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const temp = (force + POLE_MASS_LENGTH * thetaDot * thetaDot * sin) / TOTAL_MASS;
    const thetaAcc =
      (GRAVITY * sin - cos * temp) /
      (POLE_HALF_LENGTH * (4 / 3 - (POLE_MASS * cos * cos) / TOTAL_MASS));
    const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cos) / TOTAL_MASS;
    // Euler integration (as in the usual reference implementation).
    this.state = [x + TAU * xDot, xDot + TAU * xAcc, theta + TAU * thetaDot, thetaDot + TAU * thetaAcc];
    this.steps += 1;
    const [nx, , nt] = this.state;
    const terminated = Math.abs(nx) > X_LIMIT || Math.abs(nt) > THETA_LIMIT;
    const truncated = !terminated && this.steps >= MAX_STEPS;
    return { state: this.state.slice() as CartState, reward: 1, terminated, truncated };
  }
}

/** Roughly unit-scale inputs for the network. */
export function normalise(s: ArrayLike<number>, out: Float64Array = new Float64Array(4)): Float64Array {
  out[0] = s[0] / X_LIMIT;
  out[1] = s[1] / 2;
  out[2] = s[2] / THETA_LIMIT;
  out[3] = s[3] / 2;
  return out;
}
