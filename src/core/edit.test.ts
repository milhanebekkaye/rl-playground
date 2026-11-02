import { describe, expect, it } from 'vitest';
import { applyTool, resizeSpec } from './edit';
import { emptySpec, validateSpec } from './env';

describe('grid editing', () => {
  it('paints walls, goals, traps and erases', () => {
    let s = emptySpec(4, 4);
    s = applyTool(s, 'wall', 5);
    expect(s.cells[5]).toBe('wall');
    s = applyTool(s, 'trap', 5);
    expect(s.cells[5]).toBe('trap');
    s = applyTool(s, 'erase', 5);
    expect(s.cells[5]).toBe('empty');
  });

  it('never overwrites the start cell, and moving start keeps exactly one', () => {
    const s = emptySpec(4, 4);
    const startIdx = s.cells.indexOf('start');
    expect(applyTool(s, 'wall', startIdx)).toBe(s);
    const moved = applyTool(s, 'start', 6);
    expect(moved.cells.filter((c) => c === 'start')).toHaveLength(1);
    expect(moved.cells[6]).toBe('start');
    expect(moved.cells[startIdx]).toBe('empty');
  });

  it('returns the same object when nothing changes', () => {
    const s = emptySpec(4, 4);
    expect(applyTool(s, 'erase', 5)).toBe(s);
  });

  it('resizes keeping content and a valid start', () => {
    let s = emptySpec(6, 6);
    s = applyTool(s, 'wall', 8);
    const bigger = resizeSpec(s, 8, 7);
    expect(bigger.cells).toHaveLength(56);
    expect(bigger.cells[1 * 8 + 2]).toBe('wall');
    expect(validateSpec(bigger)).toBeNull();
    const smaller = resizeSpec(s, 4, 4); // start at (0,5) is cropped away
    expect(validateSpec(smaller)).toBeNull();
  });
});
