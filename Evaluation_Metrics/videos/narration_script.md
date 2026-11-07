# Narration script (about 60 seconds per clip)

Simple English voice-over for each clip. Read at a calm pace (about 130 words per minute). The sentence starting 'In this run' contains the values read from the recorded run.

## V1_env_tour  (`videos/V1_env_tour.mp4`, 33 s)

*Question:* What is the task and what do we measure?  
*Metric:* coverage, mean uncertainty, energy, return

This is our simulated bridge inspection task. The bridge surface is a grid of eight by sixteen cells, and about one in five cells has a crack. The drone starts at the base in the top-left corner, at high altitude. High altitude sees a five by five area, but blurry; low altitude sees one cell, but sharp. The colour of each cell is the detector's uncertainty: dark means we are not sure yet. Watch the counters: coverage, mean uncertainty, energy and return. Here a random policy flies around without a plan, so the uncertainty stays high and the energy is wasted. Every metric in our report is measured on this task.

## V2_agentA_vs_raster_seed1000  (`videos/V2_agentA_vs_raster_seed1000.mp4`, 33 s)

*Question:* Does Agent A finish the inspection with fewer steps?  
*Metric:* steps, energy, success

Here Agent A, our DQN, races the dense raster scan on the same bridge. The raster flies low over every row. The DQN uses higher altitudes where one wide view is enough and descends only where uncertainty stays high. Watch the step counter and the energy. In this run, Agent A - Viewpoint Planner (DQN) (seed 1000) succeeded with return 36.0 in 82 steps, energy 0.48; BASELINE Dense raster, LOW altitude (1x1) (seed 1000) succeeded with return 35.5 in 123 steps, energy 0.63. The question is: does the agent complete the task, and how efficiently? On the 100 unseen bridges the DQN needs fewer steps and less energy, but it fails on a few bridges by running out of battery.

## V3_agentA_vs_raster_heldout2005  (`videos/V3_agentA_vs_raster_heldout2005.mp4`, 33 s)

*Question:* Does Agent A generalise to an unseen bridge?  
*Metric:* success, steps, energy

This bridge, seed 2005, was never used in training or model selection. It tests generalisation. In this run, Agent A - Viewpoint Planner (DQN) (seed 2005) succeeded with return 30.3 in 118 steps, energy 0.65; BASELINE Dense raster, LOW altitude (1x1) (seed 2005) succeeded with return 35.6 in 123 steps, energy 0.63. The route is different from the previous bridge because the defects are in different places, so the agent is not replaying a memorised path. Over all 100 held-out bridges the DQN succeeds on most of them; the failures are battery-empty episodes.

## V4_agentA_real_images_seed2000  (`videos/V4_agentA_real_images_seed2000.mp4`, 32 s)

*Question:* Does Agent A work with real-image uncertainty?  
*Metric:* success (mean uncertainty must reach the calibrated 0.282)

Now the same comparison uses the real-image detector: uncertainty comes from a U-Net run on DeepCrack photographs. Watch the mean uncertainty: it has to drop below a stricter threshold. In this run, Agent A - Viewpoint Planner (DQN) [real-image detector] (seed 2000) ran out of battery with return 10.1 in 150 steps, energy 1.00; BASELINE Dense raster, LOW altitude (1x1) (seed 2000) succeeded with return 33.3 in 125 steps, energy 0.64. Real-image uncertainty is noisier, and high and mid altitudes give very poor images, so the drone has to fly low almost everywhere. The dense raster does exactly that and succeeds. Our DQN, even when trained on real images, succeeds on only a small share of bridges. This is our clearest weak result.

## V5_agentB_nominal_and_hardstart  (`videos/V5_agentB_nominal_and_hardstart.mp4`, 32 s)

*Question:* Does PPO reach the best view (q=4), even from a hard start?  
*Metric:* steps to q=4, return

Agent B, our PPO refiner, has to reach the best possible view of one cell by changing distance, angle and side offset. Watch the view quality rise to level four. The fixed orbit baseline keeps circling and never gets there. In this run, Agent B - View Refiner (PPO) (seed 2000) succeeded with return 5.6 in 4 steps, energy 0.03; BASELINE FixedOrbit (seed 2000) did not succeed with return -2.3 in 30 steps, energy 0.24; Agent B - View Refiner (PPO) (seed 2001) succeeded with return 5.8 in 4 steps, energy 0.03; BASELINE FixedOrbit (seed 2001) did not succeed with return -2.1 in 30 steps, energy 0.24; Agent B - View Refiner (PPO) (seed 2000) succeeded with return 5.7 in 5 steps, energy 0.04; BASELINE FixedOrbit (seed 2000) did not succeed with return -2.3 in 30 steps, energy 0.24. The last part starts from a hard position: far away and at a large angle. PPO still reaches the best view in a few steps, almost exactly like the privileged scripted upper bound.

## V6_agentC_mission_seed2000  (`videos/V6_agentC_mission_seed2000.mp4`, 33 s)

*Question:* Which options does the A2C supervisor choose?  
*Metric:* option sequence (EXPLORE / REFINE / RETURN), defects, energy

Agent C, our A2C supervisor, decides at each step whether to explore, refine or return home. Watch the timeline of options. In this run, Agent C - Mission Supervisor (A2C) (seed 2000) succeeded with return 43.7 in 18 steps, energy 0.51; BASELINE RuleSupervisor (seed 2000) succeeded with return 45.5 in 34 steps, energy 0.95. Notice that it only chooses explore and then return: in the original training run it never chooses refine. Refining costs energy and rarely finds a new defect under our reward, so it is not worth it to the agent. When we retrained with other seeds, some versions did use refine, with the same return.

## V7_full_mission_learned_vs_rule  (`videos/V7_full_mission_learned_vs_rule.mp4`, 33 s)

*Question:* Do the three agents work together?  
*Metric:* defects found, energy, whether Agent B is ever called

This is the full system: the supervisor calls Agent A to explore and Agent B to refine. First, the learned supervisor. In this run, Agent C - Mission Supervisor (A2C) (seed 2000) succeeded with return 41.7 in 18 steps, energy 0.53; ALL SCRIPTED (seed 2000) succeeded with return 45.5 in 34 steps, energy 0.95; RULE supervisor (scripted) (seed 2000) succeeded with return 47.5 in 32 steps, energy 0.94; ALL SCRIPTED (seed 2000) succeeded with return 45.5 in 34 steps, energy 0.95. The learned hierarchy uses much less energy, but never calls Agent B. In the second part a rule-based supervisor drives the same learned Agents A and B, so you can see Agent B working inside the hierarchy. This combination finds the most defects in our ablation.

## V8_detector_q0_to_q4  (`videos/V8_detector_q0_to_q4.mp4`, 33 s)

*Question:* Why are real images harder?  
*Metric:* uncertainty and detection at view quality q0..q4

This shows why real images are harder. The same DeepCrack patch is seen at five view qualities, from four pixels to sixty-four pixels wide. At low quality the U-Net misses cracks and its uncertainty stays high; only at the top qualities is the crack mask accurate. In our environment the high and mid altitudes map to the two worst qualities, which is why the navigation task becomes so much harder with real images.

## V9_agentA_density35_seed2013  (`videos/V9_agentA_density35_seed2013.mp4`, 33 s)

*Question:* Does Agent A cope with many more defects?  
*Metric:* success, battery

Here the bridge has thirty-five percent defective cells instead of twenty. Defects are more ambiguous for the detector, so the mean uncertainty falls more slowly. In this run, Agent A - Viewpoint Planner (DQN) (seed 2013) ran out of battery with return 9.2 in 190 steps, energy 1.00; BASELINE Dense raster, LOW altitude (1x1) (seed 2013) succeeded with return 34.9 in 123 steps, energy 0.63. On this bridge the DQN runs out of battery before meeting the threshold, while the dense raster finishes. Over 100 bridges at this density the DQN's success rate drops, which shows the limits of a policy trained only at twenty percent density.

## V10_agentA_battery0.6_seed2000  (`videos/V10_agentA_battery0.6_seed2000.mp4`, 32 s)

*Question:* Is Agent A safe when the battery is low?  
*Metric:* battery-empty, success

This is our key robustness result. The drone starts with only sixty percent battery. Watch the battery bar. In this run, Agent A - Viewpoint Planner (DQN) (seed 2000) succeeded with return 28.8 in 81 steps, energy 0.85; BASELINE Dense raster, LOW altitude (1x1) (seed 2000) ran out of battery with return 17.2 in 118 steps, energy 1.00. The dense raster needs more than sixty percent of the battery for its fixed path, so it always runs out. The DQN sees its battery level in the observation and is rewarded for battery left, so it finds a cheaper route and succeeds on most bridges.
