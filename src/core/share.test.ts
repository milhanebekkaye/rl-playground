import { describe, expect, it } from 'vitest';
import { decodeShare, encodeShare, type ShareState } from './share';
import { cliffSpec, defaultSpec } from './env';

const base = (): ShareState => ({
  spec: defaultSpec(),
  params: { alpha: 0.3, gamma: 0.95, epsilon: 0.15 },
  compare: false,
  algo: 'qlearning',
});

describe('share hash', () => {
  it('encodes to a plain, readable, versioned string', () => {
    const h = encodeShare(base());
    expect(h.startsWith('v1&w=8&h=8&c=')).toBe(true);
    expect(h).toMatch(/&r=1,-1,-0.04,200&p=0.3,0.95,0.15&m=q$/);
    expect(h).not.toMatch(/[#%\s]/);
  });

  it('round-trips specs, params, mode and algorithm', () => {
    for (const st of [
      base(),
      { ...base(), spec: cliffSpec(), compare: true, params: { alpha: 0.2, gamma: 0.99, epsilon: 0.1 } },
      { ...base(), algo: 'sarsa' as const },
    ]) {
      expect(decodeShare(encodeShare(st))).toEqual(st);
    }
  });

  it('accepts a leading #', () => {
    expect(decodeShare('#' + encodeShare(base()))).toEqual(base());
  });

  it('rounds awkward floats so links stay short', () => {
    const st = base();
    st.params.alpha = 0.1 + 0.2;
    expect(encodeShare(st)).toContain('p=0.3,');
  });

  const good = encodeShare(base());
  const bad: [string, string][] = [
    ['empty', ''],
    ['just #', '#'],
    ['garbage', 'hello world'],
    ['wrong version', good.replace('v1', 'v2')],
    ['cell count mismatch', good.replace('w=8', 'w=9')],
    ['unknown cell char', good.replace('&c=_', '&c=x')],
    ['no start', good.replace('S', '_')],
    ['two starts', good.replace('G', 'S')],
    ['width too big', 'v1&w=99&h=2&c=' + '_'.repeat(198).replace(/^_/, 'S') + '&r=1,-1,0,10&p=0.1,0.9,0.1&m=q'],
    ['non-integer width', good.replace('w=8', 'w=8.5')],
    ['NaN reward', good.replace('r=1,', 'r=NaN,')],
    ['empty number', good.replace('r=1,', 'r=,')],
    ['huge reward', good.replace('r=1,', 'r=99999,')],
    ['alpha out of range', good.replace('p=0.3', 'p=1.5')],
    ['negative epsilon', good.replace('0.15&', '-0.1&')],
    ['zero max steps', good.replace(',200&', ',0&')],
    ['bad mode', good.replace('m=q', 'm=x')],
    ['missing field', good.replace(/&p=[^&]*/, '')],
    ['duplicate key', good + '&m=s'],
    ['too long', good + '&' + 'z'.repeat(3000)],
  ];
  it.each(bad)('rejects invalid hash: %s', (_name, h) => {
    expect(decodeShare(h)).toBeNull();
  });
});
