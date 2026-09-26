# Yeseswini - what's left (in order)

Done: Steps 1-8 (env, baselines, Agent A trained: 100% success, -29% steps, -26% energy vs dense raster).

## A. Right now (while Mukhesh builds the cache)
1. Drive `team08/`: delete old `uav_inspection.zip`, upload the NEW one (same name).
2. GitHub: create repo `uav-inspection` and push (commands in chat / README). Add Mukhesh + Leekhith as collaborators.
3. Send Mukhesh: `For_Mukhesh.zip`. Send Leekhith: `For_Leekhith.zip`.
4. Open `Yeseswini_AgentA.ipynb` (v2) -> run **Step 1**, then **Steps 10 -> 10b -> 11 -> 12 -> 13** (~40 min).
   - 10/10b: epsilon-decay sensitivity (CO5)
   - 11: live demo cell (CO3) - rehearse it
   - 12: convergence / sample efficiency / regret numbers (CO4)
   - 13: flight-path figure for your poster
5. Fill every `[STEP x]` in `AgentA_slides_poster_viva.md` and `TODO` in `AgentA_report_section.tex`.

## B. After Mukhesh posts "cache ready"
6. Notebook **Step 9**: baselines + DQN on real images (~25 min) + comparison table.
   If DQN doesn't beat raster_low on real images: present it as CO4 sample-efficiency evidence (see slide notes).

## C. Wrap-up
7. Slides (17-slide plan in the .md) + individual poster (layout in the .md).
8. Send `AgentA_report_section.tex` + poster to Leekhith.
9. Push results: branch `yeseswini/agent-a` -> Pull Request -> merge; then merge Mukhesh's and Leekhith's PRs.
10. Practise the 13 viva answers + the Step 11 demo.
