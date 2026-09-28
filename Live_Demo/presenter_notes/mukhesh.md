# Mukhesh — real-image detector + Agent C: Mission Supervisor (A2C) — 3½ min

**Before your slot:** the menu is open. Seed 1000 is the default.

## Guideline mapping

| Guideline step | Where it is shown | Time |
|---|---|---|
| 1 Start the environment | d6, first line: `InspectionEnv(mode='supervise')` (d5: U-Net loaded, TEST patches) | 0:45 |
| 2 Current observation / state | d6: 6-number state [battery, coverage, mean u, found frac, dist to base, time used] | 0:55 |
| 3 Available actions + selected action | d6: π(Explore / Refine / Return) table + bar chart, chosen option | 1:05 |
| 4 Execute action, env response | d6: the option runs up to 5 navigator / 10 refiner moves; the drone moves | 1:15 |
| 5 Reward + next state | d6: REWARD = 2 × new defects − 0.5 × energy (−10 if the battery empties) | 1:20 |
| 6 Learning / update step | d8 `--agent C`: A2C updates every 8 decisions (value loss, policy loss, entropy) | 2:00 |
| 7 Trained policy | d6 animation: full mission + decision timeline | 1:30 |
| 8 Learning curve + final metrics | d6 end tables + d9 `--agent C` | 2:40 |

## Script

**0:00 — type `5`** (about 15 s). Say: "Our uncertainty is not made up. This is the U-Net we trained on
DeepCrack, running live with MC-dropout on patches from the TEST split, seen at five view qualities: 4 px
(far) up to 64 px (Agent B's best view)."
- Point at the crack patch: at q0 the network **misses the crack (IoU 0)**; from q1 on it finds it (IoU
  about 0.7–0.9).
- **Be honest:** "For a single patch the raw uncertainty is not monotone. The report says 51 % of q-steps
  go up. So the cache fuses the best view (a running minimum over q). Averaged over 1 200 crack patches,
  from q0 to q4: detect rate 40 % → 98 %, IoU 0.19 → 0.72, but false positives also rise, 0.9 % → 12.5 %."
  These numbers come from `detector_report.json` and are printed by the demo.

**0:45 — type `6`.** Say: "My agent is the supervisor. It works on a semi-MDP: each decision is an
*option* that runs for several low-level steps."
- **State (0:55):** "Six numbers."
- **Policy (1:05):** "The A2C actor gives a probability per option, and we act greedily."
- **Response and reward (1:15–1:20):** "Explore flew 5 moves. The reward is +2 per newly found defect
  minus 0.5 × energy."

**1:30 — the animation.** The banner shows the current option, and the timeline fills with one block per
decision. Say: "On this bridge: **18 decisions, return 47.74, 25/26 defects, energy 0.51**. The rule
supervisor needs 34 decisions and 0.96 energy for 49.52 and 26/26."
- **Be honest:** "Two things it has *not* learned. It never picks REFINE. And after coverage reaches
  100 % it keeps exploring for about seven decisions that earn ≈ 0 before it returns home."

**2:00 — type `8 --agent C`** (about 20 s). Say: "A fresh A2C. Every 8 decisions there is one actor-critic
update: the advantage is the return minus the critic's value. You see the value loss, the policy loss
and the entropy bonus. The mean episode reward climbs from about +13 to about +30 in 2 400 decisions.
The trained model used 100 k."

**2:40 — type `9 --agent C`.** From `agentC_vs_baselines.csv`, 20 bridges:

| policy | return | decisions | defects | energy |
|---|---|---|---|---|
| A2C (Agent C) | 39.84 | 19.25 | 20.95 | **0.510** |
| rule_supervisor | **42.82** | 33.8 | **22.55** | 0.954 |
| random | 4.07 | 2.95 | 2.95 | 0.066 |

Say: "A2C returns 7 % less than the hand-written rule but uses **47 % less energy** with 100 % safe
return. The entropy study (`ent 0.0 / 0.01 / 0.05`, 60 k-step short runs) is in the table below it. With
the real-image detector (`agentC_learned`) we get the same return as the rule (41.12 vs 41.32) with 63 %
less energy (0.35 vs 0.95)."

## If asked

- **Trained with the real Agent A inside (`agentC_withA`)?** 43.68 vs rule 44.64, energy 0.844 vs 0.926.
  It explores longer.
- **Why no REFINE?** Refine costs energy and rarely adds a *new* defect, so under our reward
  (+2 per defect) Explore dominates. This is a reward-design lesson.
- **Repo README row "A2C, learned detector" says 15.5 decisions / energy 0.40.** The CSV says 13.40 / 0.350.
  Quote the CSV.

## If something fails

Open `backup\gifs\detector.gif`, `backup\gifs\agentC_mission.gif`,
`backup\snapshots\transcript_d6_agentC_mission.txt` or `backup\snapshots\results_C.png`.
