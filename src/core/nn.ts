/**
 * A small multilayer perceptron with hand-written forward pass, backpropagation
 * and Adam. No ML library. Pure TypeScript.
 *
 * Layers are fully connected; hidden layers use ReLU and the output is linear.
 * Weights for layer l are stored row-major as W[l][i * nIn + j] (output i, input j).
 */
import type { Rng } from './rng';

export class MLP {
  readonly W: Float64Array[] = [];
  readonly b: Float64Array[] = [];
  readonly gW: Float64Array[] = [];
  readonly gb: Float64Array[] = [];
  private readonly mW: Float64Array[] = [];
  private readonly vW: Float64Array[] = [];
  private readonly mb: Float64Array[] = [];
  private readonly vb: Float64Array[] = [];
  private t = 0;

  constructor(readonly sizes: number[], rng: Rng) {
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      const w = new Float64Array(nIn * nOut);
      // He-style uniform initialisation.
      const limit = Math.sqrt(6 / nIn);
      for (let i = 0; i < w.length; i++) w[i] = (rng() * 2 - 1) * limit;
      this.W.push(w);
      this.b.push(new Float64Array(nOut));
      this.gW.push(new Float64Array(w.length));
      this.gb.push(new Float64Array(nOut));
      this.mW.push(new Float64Array(w.length));
      this.vW.push(new Float64Array(w.length));
      this.mb.push(new Float64Array(nOut));
      this.vb.push(new Float64Array(nOut));
    }
  }

  get numLayers(): number {
    return this.W.length;
  }

  /** Allocates activation buffers (one per layer) for use with forward/backward. */
  makeActivations(): Float64Array[] {
    return this.sizes.slice(1).map((n) => new Float64Array(n));
  }

  /** Forward pass. `acts[l]` receives the output of layer l; the last one is the network output. */
  forward(x: ArrayLike<number>, acts: Float64Array[]): Float64Array {
    let input: ArrayLike<number> = x;
    const last = this.numLayers - 1;
    for (let l = 0; l <= last; l++) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1];
      const W = this.W[l], b = this.b[l], out = acts[l];
      for (let i = 0; i < nOut; i++) {
        let s = b[i];
        const row = i * nIn;
        for (let j = 0; j < nIn; j++) s += W[row + j] * input[j];
        out[i] = l < last && s < 0 ? 0 : s;
      }
      input = out;
    }
    return acts[last];
  }

  /**
   * Backpropagation for one sample. `dOut` is dLoss/dOutput; gradients are
   * ACCUMULATED into gW/gb (call zeroGrad() first for a fresh batch).
   */
  backward(x: ArrayLike<number>, acts: Float64Array[], dOut: ArrayLike<number>): void {
    const last = this.numLayers - 1;
    let delta = Float64Array.from(dOut as ArrayLike<number>);
    for (let l = last; l >= 0; l--) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1];
      if (l < last) for (let i = 0; i < nOut; i++) if (acts[l][i] <= 0) delta[i] = 0;
      const input: ArrayLike<number> = l === 0 ? x : acts[l - 1];
      const W = this.W[l], gW = this.gW[l], gb = this.gb[l];
      for (let i = 0; i < nOut; i++) {
        const d = delta[i];
        if (d === 0) continue;
        gb[i] += d;
        const row = i * nIn;
        for (let j = 0; j < nIn; j++) gW[row + j] += d * input[j];
      }
      if (l > 0) {
        const next = new Float64Array(nIn);
        for (let i = 0; i < nOut; i++) {
          const d = delta[i];
          if (d === 0) continue;
          const row = i * nIn;
          for (let j = 0; j < nIn; j++) next[j] += W[row + j] * d;
        }
        delta = next;
      }
    }
  }

  zeroGrad(): void {
    for (let l = 0; l < this.numLayers; l++) {
      this.gW[l].fill(0);
      this.gb[l].fill(0);
    }
  }

  /** Scales all accumulated gradients so their global L2 norm is at most maxNorm. Returns the norm before clipping. */
  clipGradNorm(maxNorm: number): number {
    let sq = 0;
    for (let l = 0; l < this.numLayers; l++) {
      for (const v of this.gW[l]) sq += v * v;
      for (const v of this.gb[l]) sq += v * v;
    }
    const norm = Math.sqrt(sq);
    if (norm > maxNorm) {
      const k = maxNorm / norm;
      for (let l = 0; l < this.numLayers; l++) {
        for (let i = 0; i < this.gW[l].length; i++) this.gW[l][i] *= k;
        for (let i = 0; i < this.gb[l].length; i++) this.gb[l][i] *= k;
      }
    }
    return norm;
  }

  /** One Adam step using the accumulated gradients (scaled by `gradScale`, e.g. 1 / batch size). */
  adamStep(lr: number, gradScale = 1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8): void {
    this.t += 1;
    const c1 = 1 - Math.pow(beta1, this.t);
    const c2 = 1 - Math.pow(beta2, this.t);
    const apply = (p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array) => {
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] * gradScale;
        m[i] = beta1 * m[i] + (1 - beta1) * gi;
        v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
        p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    };
    for (let l = 0; l < this.numLayers; l++) {
      apply(this.W[l], this.gW[l], this.mW[l], this.vW[l]);
      apply(this.b[l], this.gb[l], this.mb[l], this.vb[l]);
    }
  }

  /** Copies weights (not optimiser state) from another network of the same shape. */
  copyFrom(other: MLP): void {
    for (let l = 0; l < this.numLayers; l++) {
      this.W[l].set(other.W[l]);
      this.b[l].set(other.b[l]);
    }
  }

  get paramCount(): number {
    return this.W.reduce((n, w, l) => n + w.length + this.b[l].length, 0);
  }
}

/** Huber loss and its derivative with respect to the prediction error `e = pred - target`. */
export function huber(e: number, delta = 1): number {
  const a = Math.abs(e);
  return a <= delta ? 0.5 * e * e : delta * (a - 0.5 * delta);
}
export function huberGrad(e: number, delta = 1): number {
  return e > delta ? delta : e < -delta ? -delta : e;
}
