# Agent A - Viewpoint Planner (DQN) - Yeseswini
Content for your 10-minute presentation, poster and viva. Numbers marked [STEP x] come from your notebook.
Numbers already filled in are from your own Colab run (analytic detector, 20 identical seeded bridges).

## Slide plan (17-slide template -> your agent)
| # | Slide | Content |
|---|---|---|
| 1 | Title | Uncertainty-Aware Active Vision for UAV Infrastructure Inspection - Agent A: Viewpoint Planner (DQN). Name, BL.EN.U4AIE23013, 22AIE401, Team 08, faculty, 2026-27 |
| 2 | Problem | Fixed raster scans waste time/battery and can't go back to uncertain spots. Agent A decides **where to fly next and at what altitude** so the whole bridge is inspected with low uncertainty using as few steps/energy as possible |
| 3 | Why RL | Sequential decisions; each move changes what is seen next (the uncertainty map); reward is delayed (success only at mission end); no labelled "correct path" exists |
| 4 | RL vs other ML | Use the course table: label vs reward, prediction vs action, i.i.d. vs sequential, loss vs return. A supervised model could only imitate a fixed path; a fixed rule (raster) can't adapt to uncertainty |
| 5 | Agent-environment | Agent = UAV navigation policy; Environment = 8x16 bridge surface grid + camera + detector + battery; loop: observe -> choose move -> env updates coverage/uncertainty -> reward |
| 6 | Architecture | Hierarchy: Agent C (supervisor) -> Agent A (where) / Agent B (how to look). Show `learning_curves.png` flow or team diagram |
| 7 | State space | 118 values in [0,1]: 7x7 local window of visited flags + 7x7 local uncertainty, 2x4 block summary (coverage + mean uncertainty), [row, col, altitude, battery]. Egocentric = relative to the drone |
| 8 | Action space | Discrete(6): North, South, East, West, Climb, Descend. Altitude high/mid/low -> camera footprint 5x5 / 3x3 / 1x1 cells, sharper view when lower |
| 9 | Reward | r = 0.2 x (total uncertainty reduced) - 0.01 per step - 0.05 if nothing new seen - 1 if moving out of bounds; success bonus = 10 + 20 x battery left. Success = coverage >= 95% AND mean uncertainty <= 0.32 |
| 10 | Algorithm | DQN: discrete actions, off-policy (replay buffer reuses data -> sample-efficient), target network for stability. PPO/A2C are used by teammates for continuous / option-level decisions |
| 11 | Core equation | y = r + gamma * max_a' Q_target(s', a');  loss = (y - Q(s, a))^2;  epsilon-greedy action selection |
| 12 | Implementation | SB3 DQN, MLP [256, 256], lr 5e-4, gamma 0.995, buffer 100k, batch 128, target update 1000, epsilon 1.0 -> 0.05 over 30%, 200k steps, 10.2 min on Colab CPU |
| 13 | Demo | Step 11 cell: map -> Q-values -> action -> reward -> next map |
| 14 | Results | `learning_curves.png` + table below + `trajectories.png` |
| 15 | Analysis | Convergence, regret, epsilon sensitivity (below) |
| 16 | Limitations | See list below |
| 17 | Conclusion | Map to CO1-CO3 (and CO4-CO5 for Part II) |

## Results table (analytic detector)
| Metric | Raster mid | Raster low (dense) | **DQN (Agent A)** | DQN vs dense raster |
|---|---|---|---|---|
| Success rate | 0% | 100% | **100%** | same |
| Steps (inspection time) | 199 | 123 | **87.5** | **-29%** |
| Energy used | 1.00 | 0.625 | **0.463** | **-26%** |
| Coverage | 100% | 95% | **99.8%** | +5 pts |
| Mean uncertainty | 0.351 | 0.219 | 0.318 | higher |
| Defects found (of 26) | 20.9 | 23.5 | 22.0 | -1.5 |
| mIoU | 0.500 | 0.626 | 0.539 | -14% |
| Return | 6.20 | 35.43 | **35.81** | slightly higher |

**Interpretation:** same success, much cheaper. The agent does *just enough* to meet the uncertainty threshold
and stops to save battery - exactly what the reward asks for. The dense raster over-inspects (better mIoU, 40% more time).
Tightening the threshold would trade energy for mIoU: a reward-design lever.

## Part II evidence
- **Learning behaviour:** return -25 -> ~34 (reaches raster_low line ~episode 900); success 0 -> 1.0 by ~episode 750; episode length 160 -> ~90.
- **Convergence:** [STEP 12: episode / env steps where success >= 95%]. Training return plateaus from ~140k steps.
  Evaluation reward oscillates (36.2 at 180k, 14.0 at 190k, 30.4 at 200k): classic DQN instability from the moving
  target; handled by keeping the best checkpoint (EvalCallback).
- **Sample efficiency:** [STEP 12: env steps to reach 95% of raster_low return]. Replay buffer reuses each transition many times.
- **Regret (vs best baseline, a proxy - the optimal policy is unknown):** [STEP 12 value]; curve rises steeply while exploring, then flattens.
- **Complexity:** 118-dim input, MLP ~100k parameters, ~330 env steps/s on Colab CPU, 10.2 min training, replay buffer 100k x 118 floats (~47 MB).
- **Exploration-exploitation (CO5):** epsilon-greedy, 1.0 -> 0.05 over first 30% of training. Big improvement came *after*
  epsilon hit its floor (60k -> 140k): exploration filled the buffer, exploitation refined the policy.
  [STEP 10b: compare 10% / 30% / 60% decay - which learned fastest, which ended best]
- **gamma:** 0.995 because success comes ~90-120 steps later; with 0.99 the bonus is discounted to ~0.3-0.4 and the agent learned only coverage (we observed this).

## Design decisions that came from experiments (great viva material)
1. First version (global one-hot position, 386 inputs) **did not learn** (0% success). Switching to an **egocentric**
   observation fixed it: an MLP generalises spatial rules much better when features are relative to the agent.
2. First reward (separate coverage + uncertainty terms) let the agent collect reward **without finishing**
   (reward hacking). Rewarding **total uncertainty reduced** + a battery-scaled success bonus aligned reward with the goal.

## Limitations
- Detector in main results is simulated (analytic); real-image (DeepCrack) version is harder to learn [STEP 9 result].
- 2D grid abstraction of a bridge; no wind, collisions or 3D geometry.
- Vanilla DQN (no Double/Dueling) -> evaluation instability.
- Single training seed; multiple seeds would give confidence intervals.
Future work: Double/Dueling DQN, multi-seed runs, 3D simulator (AirSim), curriculum from analytic to real detector.

## Poster (individual) - suggested blocks
1. **Problem** (2 lines) + drone icon
2. **MDP box**: state / actions / reward (from slides 7-9)
3. **DQN equation** + one line on epsilon-greedy
4. **Figure:** `trajectories.png` (DQN vs raster path) - the most visual result
5. **Figure:** `learning_curves.png`
6. **Key numbers:** 100% success, -29% time, -26% energy vs dense raster
7. **Insight:** egocentric state + aligned reward made learning possible
8. Footer: team, course, tools (Gymnasium, Stable-Baselines3, Colab), AI-tool acknowledgement

## Viva - short answers
1. **Why RL?** Sequential, interactive, delayed reward; no labelled optimal path.
2. **Observation vs state?** The true state includes the hidden defect map; the agent only observes coverage + detector uncertainty -> POMDP, solved as an MDP over the observation.
3. **What is optimized?** Expected discounted return: uncertainty reduced minus time/energy costs, plus success bonus.
4. **Why this reward?** Each term maps to a goal: see more (uncertainty), be fast (step cost), don't hover (revisit), stay inside (out-of-bounds), finish cheaply (battery-scaled bonus).
5. **After an action?** Drone moves/changes altitude -> camera footprint observed -> uncertainty map updated -> reward -> next observation.
6. **How does it learn?** Stores transitions in replay buffer; samples mini-batches; regresses Q(s,a) toward r + gamma max Q_target(s', a').
7. **Why DQN?** Discrete 6 actions, sample-efficient replay, standard and stable with target network.
8. **Exploration?** Epsilon-greedy with linear decay; greedy (argmax Q) at evaluation.
9. **Discount factor?** Weights future rewards; 0.995 so the end-of-mission bonus still matters.
10. **Did it learn?** Success 0 -> 100%, beats raster in time/energy on identical seeded worlds.
11. **Convergence evidence?** Plateau of training return, flattening regret, best-checkpoint eval.
12. **Limits?** Simulated detector, 2D grid, vanilla DQN instability, single seed.
13. **What did you implement?** Shared Gymnasium environment (all modes), baselines, Agent A training/evaluation, the redesign that made it learn.
