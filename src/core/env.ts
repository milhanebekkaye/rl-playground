/**
 * Grid-world environment. Pure TypeScript, no UI or browser dependencies.
 *
 * - State: the cell index `y * width + x`.
 * - Actions: 0 = up, 1 = right, 2 = down, 3 = left.
 * - Transitions are deterministic. Moving into a wall or off the grid leaves
 *   the agent where it is.
 * - Reward for an action is the reward of the cell the agent ends up in:
 *   goalReward on a goal, trapReward on a trap, stepReward otherwise
 *   (including bumping into a wall).
 * - Goal and trap cells are terminal. An episode also ends (truncated, not
 *   terminal) after `maxSteps` actions.
 */

export type CellType = 'empty' | 'wall' | 'start' | 'goal' | 'trap';
export type Outcome = 'goal' | 'trap' | 'timeout';

export const NUM_ACTIONS = 4;
export const ACTION_NAMES = ['up', 'right', 'down', 'left'] as const;
const DELTAS: ReadonlyArray<readonly [number, number]> = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];

export interface EnvSpec {
  width: number;
  height: number;
  /** Row-major, length width * height. */
  cells: CellType[];
  goalReward: number;
  trapReward: number;
  stepReward: number;
  maxSteps: number;
}

export interface StepResult {
  state: number;
  reward: number;
  /** Episode over, for any reason. */
  done: boolean;
  /** Episode ended in a goal or trap (no future value). False for a timeout. */
  terminal: boolean;
  outcome: Outcome | null;
}

export function stateIndex(spec: Pick<EnvSpec, 'width'>, x: number, y: number): number {
  return y * spec.width + x;
}

export function emptySpec(width: number, height: number): EnvSpec {
  const cells: CellType[] = new Array(width * height).fill('empty');
  cells[(height - 1) * width] = 'start';
  cells[width - 1] = 'goal';
  return { width, height, cells, goalReward: 1, trapReward: -1, stepReward: -0.04, maxSteps: 200 };
}

/** A small starter maze with a wall, a detour, and a trap row near the goal. */
export function defaultSpec(): EnvSpec {
  const rows = [
    '..#....G',
    '..#.##..',
    '..#..#T.',
    '.##.##..',
    '....#..#',
    '.##.#T..',
    '....#...',
    'S.#.....',
  ];
  const height = rows.length;
  const width = rows[0].length;
  const cells: CellType[] = [];
  for (const row of rows) {
    for (const ch of row) {
      cells.push(ch === '#' ? 'wall' : ch === 'S' ? 'start' : ch === 'G' ? 'goal' : ch === 'T' ? 'trap' : 'empty');
    }
  }
  return { width, height, cells, goalReward: 1, trapReward: -1, stepReward: -0.04, maxSteps: 200 };
}

/** Returns a human-readable problem with the spec, or null if it is usable. */
export function validateSpec(spec: EnvSpec): string | null {
  if (spec.cells.length !== spec.width * spec.height) return 'cells length does not match width * height';
  const starts = spec.cells.filter((c) => c === 'start').length;
  if (starts !== 1) return 'the grid needs exactly one start cell';
  if (!(spec.maxSteps >= 1)) return 'maxSteps must be at least 1';
  return null;
}

export class GridEnv {
  readonly spec: EnvSpec;
  readonly startState: number;
  private state: number;
  private steps = 0;
  private finished = false;

  constructor(spec: EnvSpec) {
    const problem = validateSpec(spec);
    if (problem) throw new Error(problem);
    this.spec = spec;
    this.startState = spec.cells.indexOf('start');
    this.state = this.startState;
  }

  get numStates(): number {
    return this.spec.width * this.spec.height;
  }

  get currentState(): number {
    return this.state;
  }

  get stepCount(): number {
    return this.steps;
  }

  isTerminalState(s: number): boolean {
    const c = this.spec.cells[s];
    return c === 'goal' || c === 'trap';
  }

  isWall(s: number): boolean {
    return this.spec.cells[s] === 'wall';
  }

  reset(): number {
    this.state = this.startState;
    this.steps = 0;
    this.finished = false;
    return this.state;
  }

  step(action: number): StepResult {
    if (this.finished) throw new Error('episode is over; call reset()');
    if (!(action >= 0 && action < NUM_ACTIONS) || !Number.isInteger(action)) {
      throw new Error(`invalid action ${action}`);
    }
    const { width, height, cells } = this.spec;
    const x = this.state % width;
    const y = Math.floor(this.state / width);
    const nx = x + DELTAS[action][0];
    const ny = y + DELTAS[action][1];
    const inside = nx >= 0 && nx < width && ny >= 0 && ny < height;
    if (inside && cells[ny * width + nx] !== 'wall') this.state = ny * width + nx;
    this.steps += 1;

    const cell = cells[this.state];
    let reward = this.spec.stepReward;
    let terminal = false;
    let outcome: Outcome | null = null;
    if (cell === 'goal') {
      reward = this.spec.goalReward;
      terminal = true;
      outcome = 'goal';
    } else if (cell === 'trap') {
      reward = this.spec.trapReward;
      terminal = true;
      outcome = 'trap';
    }
    let done = terminal;
    if (!terminal && this.steps >= this.spec.maxSteps) {
      done = true;
      outcome = 'timeout';
    }
    this.finished = done;
    return { state: this.state, reward, done, terminal, outcome };
  }
}
