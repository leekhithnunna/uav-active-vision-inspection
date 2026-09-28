# Yeseswini — Agent A: Viewpoint Planner (DQN) — 3½ min

**Before your slot:** the menu (`python run_demo.py`) is already open and warmed up with `1 --fast`.
Seed 1000 is the default, so the numbers below are what you will see.

## Guideline mapping

| Guideline step | Where it is shown | Time |
|---|---|---|
| 1 Start the environment | d2, first line: `InspectionEnv(mode='navigate')`, seed 1000 | 0:00 |
| 2 Current observation / state | d2: STATE panel + ASCII map + "118 numbers = 7×7 window + block summary + pose" | 0:10 |
| 3 Available actions + selected action | d2: table of 6 Q-values, argmax highlighted; bar chart in the window | 0:25 |
| 4 Execute action, env response | d2: "UAV now at …, N new cells seen, uncertainty reduced by …" | 0:40 |
| 5 Reward + next state | d2: REWARD table (0.2×Δu, step cost, revisit, out-of-bounds, success) + next STATE | 0:50 |
| 6 Learning / update step | d8 `--agent A`: live TD loss, ε decay, one Bellman update worked out | 1:30 |
| 7 Trained policy | d2 "r" (run to end), then d3 animation vs dense raster | 2:10 |
| 8 Learning curve + final metrics | d3 end table + d9 `--agent A` | 3:00 |

## Script

**0:00 — type `2`.** Say: "This is our shared Gymnasium environment: an 8 × 16 bridge, 26 hidden defects,
the UAV starts at the base at HIGH altitude. My agent decides *where to fly next* — six discrete actions."

**0:10 — state.** Say: "The DQN does not see the whole map. It sees a 7 × 7 window of visited cells and
their uncertainty around the drone, a coarse summary of the bridge, and its own pose and battery: 118 numbers."

**0:25 — Q-values.** Point at the table. Say: "The Q-network outputs one value per action. The action we
take is the argmax, highlighted." On step 1 the argmax is **South**; on step 2 it is **Descend**.

**0:40–0:50 — response and reward.** Say: "The reward is 0.2 × the uncertainty removed, minus a step cost.
Below the table the demo checks that these terms add up to exactly the reward the environment returned."
Press **Enter** twice more, so three steps are narrated in full.

**1:10 — press `r`** to run to the end, which takes a few seconds. Close the window.

**1:30 — type `8 --agent A`** (about 30 s). Say: "Now the learning step itself. This is a *fresh* network
training for 5 000 steps. The grey dots are episode returns, ε falls from 1 to 0.05, and every 4 steps
there is one TD update on a replay batch." When the table appears: "Here is one update written out:
y = r + γ · max Q_target(s′), and the loss is the Huber loss of y − Q(s,a). This short run is *not* our
trained agent. The trained agent used 200 k steps." In our test run the return rose from about −35 to
about +8. The exact numbers can vary slightly between machines.

**2:10 — type `3`.** Say: "Same seed, so the same bridge. On the left is the trained DQN, on the right the
dense low-altitude raster. The path colour shows altitude: the DQN overviews from HIGH and MID, then drops
LOW only where it is still uncertain." At the end: **"on this bridge: −33 % steps, −23 % energy"**
(82 vs 123 steps, energy 0.48 vs 0.63). **Be honest:** the raster found 26/26 defects here, the DQN 24/26.

**3:00 — type `9 --agent A`.** Say: "Over the 20 evaluation bridges (`agentA_vs_baselines.csv`), the DQN
succeeds 100 % of the time in **88.3 steps with 0.502 energy**, against 123 steps and 0.625 for the dense
raster: −28 % steps and −20 % energy at the same return (35.06 vs 35.43). The mid-altitude raster never
succeeds (0 %, 199 steps). ε-decay sensitivity: 10 % decay was best (37.78, 74.7 steps), 60 % worst
(28.09, 80 % success). Those are 150 k-step runs, labelled on screen."

## If asked

- **Real-image detector?** `agentA_learned`: 25 % success, 151.6 steps. It did **not** beat the dense
  raster (100 %, 123 steps) with the harder threshold. That is an honest negative result.
- **Why the ASCII map?** It is `env.render()` from the shared code, shown unmodified.

## If something fails

Open `backup\gifs\agentA_vs_raster.gif` and `backup\snapshots\transcript_d2_agentA_step.txt`.
The results PNG is `backup\snapshots\results_A.png`.
