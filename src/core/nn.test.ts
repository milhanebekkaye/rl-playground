import { describe, expect, it } from 'vitest';
import { MLP, huber, huberGrad } from './nn';
import { mulberry32 } from './rng';

/** Loss for one sample: sum of Huber(out - target). */
function loss(net: MLP, x: number[], target: number[], delta: number): number {
  const acts = net.makeActivations();
  const out = net.forward(x, acts);
  let s = 0;
  for (let i = 0; i < out.length; i++) s += huber(out[i] - target[i], delta);
  return s;
}

describe('MLP backprop', () => {
  it.each([
    ['small residuals (quadratic region)', 5],
    ['large residuals (linear region)', 0.05],
  ])('matches numerical gradients: %s', (_name, huberDelta) => {
    const rng = mulberry32(11);
    const net = new MLP([4, 7, 5, 2], rng);
    // Give biases non-zero values so every parameter matters.
    for (const b of net.b) for (let i = 0; i < b.length; i++) b[i] = (rng() - 0.5) * 0.5;
    const x = [0.3, -0.7, 0.5, 0.9];
    const target = [0.4, -1.2];
    const acts = net.makeActivations();
    const out = net.forward(x, acts);
    net.zeroGrad();
    net.backward(x, acts, Array.from(out, (o, i) => huberGrad(o - target[i], huberDelta)));

    const eps = 1e-6;
    let worst = 0;
    let checked = 0;
    const check = (p: Float64Array, g: Float64Array) => {
      for (let i = 0; i < p.length; i++) {
        const old = p[i];
        p[i] = old + eps; const up = loss(net, x, target, huberDelta);
        p[i] = old - eps; const down = loss(net, x, target, huberDelta);
        p[i] = old;
        const numeric = (up - down) / (2 * eps);
        const denom = Math.max(1e-7, Math.abs(numeric) + Math.abs(g[i]));
        worst = Math.max(worst, Math.abs(numeric - g[i]) / denom);
        checked++;
      }
    };
    for (let l = 0; l < net.numLayers; l++) { check(net.W[l], net.gW[l]); check(net.b[l], net.gb[l]); }
    expect(checked).toBe(net.paramCount);
    expect(worst).toBeLessThan(1e-5);
  });

  it('accumulates gradients over samples (batch = sum of single-sample gradients)', () => {
    const net = new MLP([2, 3, 1], mulberry32(5));
    const xs = [[1, 2], [-1, 0.5]];
    const acts = net.makeActivations();
    net.zeroGrad();
    for (const x of xs) { net.forward(x, acts); net.backward(x, acts, [1]); }
    const batch = Float64Array.from(net.gW[0]);
    const single = new Float64Array(batch.length);
    for (const x of xs) { net.zeroGrad(); net.forward(x, acts); net.backward(x, acts, [1]); single.forEach((_, i) => (single[i] += net.gW[0][i])); }
    batch.forEach((v, i) => expect(v).toBeCloseTo(single[i], 12));
  });

  it('Adam fits a tiny regression problem', () => {
    const rng = mulberry32(3);
    const net = new MLP([1, 16, 1], rng);
    const acts = net.makeActivations();
    const xs = Array.from({ length: 20 }, (_, i) => -1 + (2 * i) / 19);
    const f = (x: number) => x * x - 0.3;
    const mse = () => xs.reduce((s, x) => s + (net.forward([x], acts)[0] - f(x)) ** 2, 0) / xs.length;
    const before = mse();
    for (let step = 0; step < 1500; step++) {
      net.zeroGrad();
      for (const x of xs) { const o = net.forward([x], acts)[0]; net.backward([x], acts, [o - f(x)]); }
      net.adamStep(0.01, 1 / xs.length);
    }
    expect(mse()).toBeLessThan(before * 0.02);
    expect(mse()).toBeLessThan(0.01);
  });

  it('copyFrom makes two networks agree', () => {
    const a = new MLP([4, 8, 2], mulberry32(1));
    const b = new MLP([4, 8, 2], mulberry32(2));
    const x = [0.1, 0.2, 0.3, 0.4];
    expect(Array.from(a.forward(x, a.makeActivations()))).not.toEqual(Array.from(b.forward(x, b.makeActivations())));
    b.copyFrom(a);
    expect(Array.from(a.forward(x, a.makeActivations()))).toEqual(Array.from(b.forward(x, b.makeActivations())));
  });

  it('huber matches its definition', () => {
    expect(huber(0.5)).toBeCloseTo(0.125, 12);
    expect(huber(3)).toBeCloseTo(2.5, 12);
    expect(huberGrad(0.5)).toBe(0.5);
    expect(huberGrad(-4)).toBe(-1);
  });
});
