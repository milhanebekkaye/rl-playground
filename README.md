# RL Playground

A browser-only reinforcement learning playground. Draw a small grid world, press play, and watch a tabular Q-learning agent learn to reach the goal. The board shows what the agent believes (value heatmap and policy arrows) while a learning curve shows whether it is actually getting better.

> Demo GIF: _placeholder, to be added (`docs/demo.gif`)._

Stack: Vite, React, TypeScript, Canvas, a Web Worker for training, Vitest for tests. No backend; the build is static and deploys as-is (for example to Vercel).

## How RL is shown here

**State.** The agent's state is the grid cell it stands on, numbered `y * width + x`. Walls are never entered. The agent is the orange dot; its path during the current episode is the orange trail.

**Action.** Four moves: up, right, down, left. Transitions are deterministic. Walking into a wall or the grid border leaves the agent where it is (and still costs a step).

**Reward.** Entering a goal cell gives the goal reward, entering a trap gives the trap reward, and every other step gives the step reward (a small negative number pushes the agent toward short paths). Goals and traps end the episode, and so does hitting the step limit. All four numbers are editable. The chart plots the total reward of each finished episode.

**Q-value.** `Q(s, a)` is the agent's estimate of the total discounted reward it will collect if it takes action `a` in state `s` and then acts well. Each cell is coloured by `max_a Q(s, a)` (teal is positive, red is negative, normalised to the largest value on the board), and the arrow points along `argmax_a Q(s, a)`, the greedy policy. Cells the agent has not learned anything about yet show no arrow.

**Epsilon-greedy.** To learn, the agent must try things. With probability epsilon it picks a random action; otherwise it takes the action with the highest Q-value (ties broken at random). Set epsilon to 0 once it has learned and the trail becomes the greedy path; set it to 1 and it wanders randomly.

**TD error.** After every step the agent compares what it just observed with what it predicted: `target = r + gamma * max_a' Q(s', a')` (just `r` if the next state is terminal), `TD error = target - Q(s, a)`, and then `Q(s, a) += alpha * TD error`. The "Last TD error" readout shows the most recent value; it shrinks toward zero as predictions become consistent. A timeout is not treated as terminal, so the agent still bootstraps from the next state when the step limit cuts an episode short. Alpha is the learning rate, gamma the discount.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
```

Production build: `npm run build` (output in `dist/`, serve with `npm run preview`).

## Test

```bash
npm test           # Vitest: environment, Q-learning update rule, maze solving, editor helpers
npm run typecheck
```

The tests cover deterministic transitions, walls, terminal cells and the step limit; the Q-update checked against hand-computed values (including terminal transitions); epsilon-greedy behaviour; and that the agent finds the shortest path through a small maze.

## Layout

- `src/core/` pure TypeScript with no React or DOM imports: `env.ts`, `qlearning.ts`, `trainer.ts`, `edit.ts`, `rng.ts`
- `src/worker/` the Web Worker that runs training and posts Q-table snapshots and episode stats
- `src/ui/` React components and canvas drawing

## License

MIT, see [LICENSE](LICENSE).
