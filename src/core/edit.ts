/** Pure helpers for editing an EnvSpec (used by the grid editor). */
import type { CellType, EnvSpec } from './env';

export type Tool = 'wall' | 'start' | 'goal' | 'trap' | 'erase';

/**
 * Applies a drawing tool to one cell. Returns the same object if nothing
 * changed. The single start cell can only be moved (by painting a new start),
 * never overwritten, so the grid always has exactly one start.
 */
export function applyTool(spec: EnvSpec, tool: Tool, index: number): EnvSpec {
  if (index < 0 || index >= spec.cells.length) return spec;
  const current = spec.cells[index];
  if (current === 'start') return spec;
  const next: CellType = tool === 'erase' ? 'empty' : tool;
  if (current === next) return spec;
  const cells = spec.cells.slice();
  if (tool === 'start') {
    const old = cells.indexOf('start');
    if (old >= 0) cells[old] = 'empty';
  }
  cells[index] = next;
  return { ...spec, cells };
}

/** Resizes the grid, keeping the overlapping top-left region. Guarantees one start. */
export function resizeSpec(spec: EnvSpec, width: number, height: number): EnvSpec {
  const cells: CellType[] = new Array(width * height).fill('empty');
  for (let y = 0; y < Math.min(height, spec.height); y++) {
    for (let x = 0; x < Math.min(width, spec.width); x++) {
      cells[y * width + x] = spec.cells[y * spec.width + x];
    }
  }
  if (!cells.includes('start')) cells[(height - 1) * width] = 'start';
  return { ...spec, width, height, cells };
}
