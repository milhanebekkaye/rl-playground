# RL Playground

A browser-only reinforcement learning playground. Draw a small grid world, press play, and watch a tabular agent (Q-learning or SARSA) learn to reach the goal. The board shows what the agent believes (value heatmap and policy arrows) while a learning curve shows whether it is actually getting better.

Features:

- Grid editor: walls, start, goal and trap cells; editable goal, trap and step rewards, step limit, grid size.
- Live training in a Web Worker: value heatmap, greedy-policy arrows, the current episode's trail, reward-per-episode curve.
- **Compare mode:** Q-learning and SARSA train side by side on the same maze, hyperparameters, seed and step count, each with its own board and curve.
- **TD-error view:** a per-cell heatmap of recent mean absolute TD error plus a TD-error trace.
- **Shareable mazes:** the maze, rewards, hyperparameters and mode are encoded in the URL hash; "Copy link" gives you a link.
- A built-in **Cliff walk** preset where the two algorithms learn visibly different paths.

> Demo GIF: _placeholder, to be added (`docs/demo.gif`)._

Stack: Vite, React, TypeScript, Canvas, a Web Worker for training, Vitest for tests. No backend; the build is static and deploys as-is (for example to Vercel).

## How RL is shown here

**State.** The agent's state is the grid cell it stands on, numbered `y * width + x`. Walls are never entered. The agent is the orange dot; its path during the current episode is the orange trail.

**Action.** Four moves: up, right, down, left. Transitions are deterministic. Walking into a wall or the grid border leaves the agent where it is (and still costs a step).

**Reward.** Entering a goal cell gives the goal reward, entering a trap gives the trap reward, and every other step gives the step reward (a small negative number pushes the agent toward short paths). Goals and traps end the episode, and so does hitting the step limit. All four numbers are editable. The chart plots the total reward of each finished episode.

**Q-value.** `Q(s, a)` is the agent's estimate of the total discounted reward it will collect if it takes action `a` in state `s` and then acts well. Each cell is coloured by `max_a Q(s, a)` (teal is positive, red is negative, normalised to the largest value on the board), and the arrow points along `argmax_a Q(s, a)`, the greedy policy. Cells the agent has not learned anything about yet show no arrow.

**Epsilon-greedy.** To learn, the agent must try things. With probability epsilon it picks a random action; otherwise it takes the action with the highest Q-value (ties broken at random). Set epsilon to 0 once it has learned and the trail becomes the greedy path; set it to 1 and it wanders randomly.

**TD error.** After every step the agent compares what it just observed with what it predicted: `target = r + gamma * max_a' Q(s', a')` (just `r` if the next state is terminal), `TD error = target - Q(s, a)`, and then `Q(s, a) += alpha * TD error`. The "Last TD error" readout shows the most recent value; it shrinks toward zero as predictions become consistent. A timeout is not treated as terminal, so the agent still bootstraps from the next state when the step limit cuts an episode short. Alpha is the learning rate, gamma the discount.

## Q-learning vs SARSA

Both are temporal-difference control methods and differ in one term. Q-learning is off-policy: its target uses the best next action, `r + gamma * max_a' Q(s', a')`, so it learns the value of the greedy policy regardless of how it explores. SARSA is on-policy: its target uses the action it will actually take next, `r + gamma * Q(s', a')`, where `a'` comes from the epsilon-greedy policy, so its values include the cost of its own random slips.

The Cliff walk preset (Sutton and Barto, example 6.6) shows the difference. With a cliff of trap cells along the bottom edge and epsilon above zero, Q-learning learns the shortest path along the edge, which is optimal but gets it pushed off the cliff by exploratory moves during training. SARSA learns a longer path away from the edge, with better reward while it is still exploring. Lower epsilon toward zero and the two converge. The contrast is asserted by a unit test.

## The TD-error view

Switch the board to "TD error" to colour each cell by an exponential moving average of the absolute TD error from updates made in that cell. Bright cells are where predictions are still being corrected: early on that is everywhere near the path, later mostly around traps, the goal and wherever exploration keeps surprising the agent. The chart shows the mean absolute TD error per batch of updates (line) and the mean signed error (dots) over recent training.

The TD error plays the same role as the reward prediction error that neuroscientists have compared to dopamine activity (Schultz, Dayan and Montague, 1997): a surprise signal that drives learning and fades as predictions improve. It is an analogy for intuition, not a model of the brain.

## Share links

The hash is plain and versioned, for example `#v1&w=8&h=8&c=...&r=1,-1,-0.04,200&p=0.3,0.95,0.15&m=q`. `c` is one character per cell (`_` empty, `w` wall, `S` start, `G` goal, `T` trap), `r` is goal/trap/step reward and max steps, `p` is alpha/gamma/epsilon, `m` is `q`, `s` or `cmp`. It is validated on load; an invalid hash falls back to the example maze.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
```

Production build: `npm run build` (output in `dist/`, serve with `npm run preview`).

## Test

```bash
npm test           # Vitest: environment, Q-learning and SARSA updates, cliff contrast, share codec, editor helpers
npm run typecheck
```

The tests cover deterministic transitions, walls, terminal cells and the step limit; the Q-update checked against hand-computed values (including terminal transitions); epsilon-greedy behaviour; that the agent finds the shortest path through a small maze, that Q-learning and SARSA learn different cliff paths, and that share hashes round-trip and reject invalid input.

## Layout

- `src/core/` pure TypeScript with no React or DOM imports: `env.ts`, `agent.ts`, `qlearning.ts`, `sarsa.ts`, `trainer.ts`, `edit.ts`, `share.ts`, `rng.ts`
- `src/worker/` the Web Worker that runs training and posts Q-table snapshots and episode stats
- `src/ui/` React components and canvas drawing

## License

MIT, see [LICENSE](LICENSE).
