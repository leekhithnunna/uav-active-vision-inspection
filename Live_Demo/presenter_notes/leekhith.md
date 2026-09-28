# Leekhith — Agent B: View Refiner (PPO) + full integration — 4 min

**Before your slot:** the menu is open. Seed 1000 is the default.

## Guideline mapping

| Guideline step | Where it is shown | Time |
|---|---|---|
| 1 Start the environment | d4, first line: `InspectionEnv(mode='refine')`, one uncertain target cell | 0:00 |
| 2 Current observation / state | d4: OBSERVATION table [distance, angle, lateral, target u, battery, time left] | 0:10 |
| 3 Available actions + selected action | d4: continuous Box(3) in [−1, 1]: μ and σ per dimension, executed a = clip(μ) | 0:25 |
| 4 Execute action, env response | d4: new view quality s → level q, new target uncertainty | 0:40 |
| 5 Reward + next state | d4: REWARD table (Δu, action cost, potential-based shaping, +5 success) | 0:50 |
| 6 Learning / update step | d8 `--agent B`: PPO updates (value loss, clipped policy loss, entropy, σ, clip fraction) | 1:30 |
| 7 Trained policy | d4 animation vs FixedOrbit; d7 full mission | 1:10 / 2:10 |
| 8 Learning curve + final metrics | d4/d7 end tables + d9 `--agent B` and `--agent full` | 3:20 |

## Script

**0:00 — type `4`.** Say: "When the supervisor says *Refine*, my agent takes over. The UAV is near one
uncertain cell and moves the camera continuously: distance, viewing angle, lateral offset."

**0:10–0:50 — the narrated steps.**
- **Observation:** "Six numbers."
- **Policy:** "PPO outputs a Gaussian for each action dimension. In the demo we act deterministically on
  the mean, clipped to [−1, 1]. σ is the exploration the policy learned."
- **Response:** "Watch view quality s climb: q2, q3, q4."
- **Reward:** "The reward is the uncertainty reduction plus potential-based shaping 0.5·(γ s′ − s). The
  shaping does not change the optimal policy, and the terms add up to the env reward."

**1:10 — the animation.** Say: "Three targets. Left: PPO reaches q4 in **4, 3 and 4 steps** (returns
+5.56, +5.73, +5.78). Right: the FixedOrbit baseline sweeps the angle, never gets past q2, and times out
after 30 steps (about −2.1 to −2.4)."

**1:30 — type `8 --agent B`** (about 30 s). Say: "A fresh PPO: collect 512 steps, then 10 epochs of
clipped-surrogate updates. Each UPDATE line is one PPO iteration: value loss, policy-gradient loss,
entropy, the policy's σ and the clip fraction. The mean episode return climbs from about −2 to about +4 in 6 k steps.
The trained model used 100 k steps."

**2:10 — type `7`.** Say: "Now all three trained agents together. Agent C picks an option, EXPLORE runs
Agent A for 5 moves, REFINE would run Agent B." At the end:
- "On this bridge the learned hierarchy uses **16 decisions and 75 UAV moves against 34 and 149, and
  energy 0.54 against 0.96**. The scripted system finds 2 more defects (26 vs 24) and gets a higher
  return (49.5 vs 45.7)."
- **Say it before they ask:** "Our trained supervisor never chooses REFINE. We checked all 20 evaluation
  bridges, so in this mission my Agent B is not called. It learned that the Refine reward is not worth
  the energy under our reward."

**Optional, if there is time — `7 --supervisor rule`** (about 35 s). "With the rule supervisor on top of the learned A and B,
REFINE hands over to Agent B 10 times, and you can see the geometry panel move. This is the best
configuration we measured."

**3:20 — type `9 --agent full`** (close the window after about 15 s). From `full_mode_results.csv`, 20 bridges:

| config | return | decisions | energy | defects |
|---|---|---|---|---|
| ALL LEARNED | 40.94 | 17.2 | 0.517 | 21.5 |
| ALL SCRIPTED | 42.82 | 33.8 | 0.954 | 22.55 |
| rule C + learned A, B | **44.84** | 32.05 | 0.925 | **23.55** |

Say: "Learned: −49 % decisions and −46 % energy, for about one defect less."
And for Agent B (`agentB_vs_baselines.csv`, 50 targets): PPO **5.67, 3.66 steps, 100 %**, FixedOrbit
−2.28 (0 %), random −2.04. PPO equals the scripted upper bound (5.67, 3.64 steps). The entropy and
clip-range runs all give 100 % in 3.6–3.7 steps, so the result is robust to those settings.

## If asked

- **Real-image detector?** `agentB_learned`: 5.32 return, 100 % success, 3.40 steps. That again equals the
  scripted upper bound (5.32).
- **Why does B only tie the upper bound?** The refine task has a known optimum (move straight to d=0,
  angle 0, lateral 0). PPO learned it from reward alone, without that formula.

## If something fails

Open `backup\gifs\agentB_refine.gif`, `backup\gifs\full_mission.gif` or `backup\gifs\full_mission_rule.gif`.
The results PNGs are `backup\snapshots\results_B.png` and `results_full.png`.
