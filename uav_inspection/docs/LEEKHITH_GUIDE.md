# Leekhith - your part (Team 08)

You own:
1. **Agent B - Active View Refiner (PPO)**: your individual viva component.
2. **Integration**: running all three agents together (team demo) + team results.
3. **Assembly**: team report (Overleaf), team poster, slide template.

Code is written and tested. Your job: run it, understand it, analyse it, present it, and assemble the team deliverables.

## 0. Setup (5 min)
1. Accept the Drive share for `team08` -> drive.google.com -> **Shared with me -> right-click team08 -> Organize -> Add shortcut -> My Drive**.
2. Accept the GitHub invite to `uav-inspection`.
3. Open `Leekhith_AgentB_Integration.ipynb` in Colab (CPU runtime is fine).

## 1. Agent B - PPO (you can start immediately, ~30 min total)
**MDP (mode `refine`)**: the UAV is near ONE uncertain cell and must get the best possible view of it.
| | |
|---|---|
| State (6) | distance to target, viewing angle, lateral offset, target uncertainty, battery, time left |
| Actions | **Continuous** Box(3) in [-1,1]: change in distance / angle / lateral offset (x0.2 per step) |
| Dynamics | view quality s = 1 - 0.6 d - 0.4 abs(angle) - 0.3 abs(lateral); q = round(4 s); cell keeps its best view |
| Reward | 1.0 x uncertainty reduced - 0.01 x abs(a) + 0.5 x (0.99 s' - s) shaping; +5 at best view (q=4); -2 on timeout |
| Episode | <= 30 steps; success = best view reached |
| Algorithm | PPO (clipped objective), Gaussian policy, 4 parallel envs, n_steps 256, clip 0.2 |

The shaping term is **potential-based** (Ng, Harada & Russell, 1999): it speeds learning without changing the optimal policy.
Why PPO: continuous actions (DQN can't do these), stable clipped updates, on-policy.
Core equation: L_CLIP = E[ min( r_t(theta) A_t , clip(r_t(theta), 1-eps, 1+eps) A_t ) ], r_t = pi_theta(a|s) / pi_old(a|s).
Exploration (CO5): Gaussian std sigma (learned) + entropy bonus (ent_coef).

Run: Step 1 -> 2 -> 3 (train) -> 4 (curves + table) -> 5 (live demo) -> 6 (convergence) -> 7 (entropy + clip sensitivity).

**Our test run (60k steps, 0.6 min):** PPO success 100% in 3.7 steps vs fixed orbit 0% and random 2%;
it matches the scripted "fly straight to the ideal view" upper bound. Present the scripted policy honestly as an
**upper bound that uses privileged knowledge of the geometry**, not as a competitor.

## 2. Integration (after Agent A and Agent C models exist in Drive)
Step 8 runs `scripts/run_full_mode.py`: Agent C picks Explore/Refine/Return, Explore calls Agent A, Refine calls Agent B.
It also runs an **ablation** on 20 identical bridges. Our test numbers:

| Config | Return | Coverage | Defects (/26) | mIoU | Energy |
|---|---|---|---|---|---|
| All learned (C + A + B) | 39.3 | 0.96 | 20.6 | 0.504 | **0.49** |
| All scripted (rule + raster + approach) | 42.8 | 1.00 | 22.6 | 0.597 | 0.95 |
| C learned, A/B scripted | 41.1 | 1.00 | 21.6 | 0.520 | 0.45 |
| Rule C, A + B learned | **43.9** | 1.00 | **23.1** | **0.618** | 0.94 |

Honest team message: learned hierarchy uses **~half the energy** with slightly fewer defects; learned A+B under a
simple rule supervisor gives the best defect/mIoU. Agent C was trained with scripted sub-agents, so swapping in
learned ones shifts its input distribution - a real limitation to mention (fix: retrain C with `--nav_model`).

Step 9 builds the team table + `team_summary.png` + a narrated mission (`demo_mission.txt`) for the team slides/poster.

## 3. Assembly checklist (you coordinate, each member writes their own content)
- [ ] Overleaf: team intro, problem, hierarchy/workflow, literature survey, integration results, conclusion
      + paste sections from Yeseswini (`AgentA_report_section.tex`), Mukhesh (Agent C + detector) and yours (Agent B)
- [ ] AI-tool usage acknowledgement section in the report
- [ ] Team poster: problem -> hierarchy diagram -> 3 agents in one row -> `team_summary.png` -> key numbers
- [ ] Slide template shared with both (common first 2-3 slides identical for all)
- [ ] Upload everything to the Teams channel **before the slot**

## 4. GitHub (PowerShell)
```powershell
git clone https://github.com/<yeseswini-username>/uav-inspection.git
cd uav-inspection
git checkout -b leekhith/agent-b
# add your executed notebook (Colab: File -> Download .ipynb) and small result files
git add .
git commit -m "Agent B + integration results"
git push -u origin leekhith/agent-b      # then open a Pull Request on GitHub
```
