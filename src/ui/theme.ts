/** Reads the CSS custom properties so canvases follow light/dark mode. */
export interface Palette {
  paper: string; ink: string; faint: string; grid: string; wall: string;
  accent: string; accentRgb: [number, number, number]; pos: [number, number, number]; neg: [number, number, number];
  goal: string; trap: string; trail: string;
}

function parseRgb(v: string): [number, number, number] {
  const p = v.split(',').map((s) => parseInt(s.trim(), 10));
  return [p[0] || 0, p[1] || 0, p[2] || 0];
}

export function readPalette(): Palette {
  const cs = getComputedStyle(document.documentElement);
  const g = (n: string) => cs.getPropertyValue(n).trim();
  return {
    paper: g('--paper'), ink: g('--ink'), faint: g('--faint'), grid: g('--grid'), wall: g('--wall'),
    accent: g('--accent'), accentRgb: parseRgb(g('--accent-rgb')), pos: parseRgb(g('--pos-rgb')), neg: parseRgb(g('--neg-rgb')),
    goal: g('--goal'), trap: g('--trap'), trail: g('--trail'),
  };
}

/** Mixes the paper colour toward a tint; t in [0,1]. Returns an rgb() string. */
export function tint(base: [number, number, number], paper: [number, number, number], t: number): string {
  const m = (i: number) => Math.round(paper[i] + (base[i] - paper[i]) * t);
  return `rgb(${m(0)},${m(1)},${m(2)})`;
}

export function paperRgb(): [number, number, number] {
  return parseRgb(getComputedStyle(document.documentElement).getPropertyValue('--paper-rgb'));
}
