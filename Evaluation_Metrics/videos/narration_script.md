# Narration script (about 60 seconds per clip)

Simple English voice-over for each clip, written for a calm pace of about 130 words per minute. The sentence starting 'In this run' contains the values read from the recorded run. Each clip is about one minute long, so the voice-over fits the video.

## V1_env_tour  (`videos/V1_env_tour.mp4`, video 58 s, voice-over about 55 s)

*Question:* What is the task and what do we measure?  
*Metric:* coverage, mean uncertainty, energy, return

This is our bridge inspection task: a grid of eight by sixteen cells, and about one cell in five has a crack. The drone starts at the base, top left, at high altitude. From high up it sees a five by five patch, but blurry; from low down it sees one cell, but sharply. The colour of each cell shows how uncertain the crack detector still is. Watch the counters: coverage, mean uncertainty, energy and return. Here a random policy flies without a plan, so the uncertainty stays high and the battery is wasted. In this run the random policy flew 60 steps and collected a return of -10.4. Every metric in our report is measured on this task.

## V2_agentA_vs_raster_seed1000  (`videos/V2_agentA_vs_raster_seed1000.mp4`, video 58 s, voice-over about 57 s)

*Question:* Does Agent A finish the inspection with fewer steps?  
*Metric:* steps, energy, success

Here Agent A, our DQN, races the dense raster scan on the same bridge. The raster flies low over every row. The DQN uses higher altitudes where one wide view is enough, and only goes down where the uncertainty stays high. Watch the step counter and the battery. In this run, the DQN succeeded after 82 steps, using 48 percent of the battery; the dense raster succeeded after 123 steps, using 63 percent of the battery. On the hundred unseen bridges this original model needs fewer steps and less energy than the raster, but it runs out of battery on a few of them. Three retrained copies of the same DQN do better: they succeed on 99 to 100 percent of these bridges.

## V3_agentA_vs_raster_heldout2005  (`videos/V3_agentA_vs_raster_heldout2005.mp4`, video 58 s, voice-over about 50 s)

*Question:* Does Agent A generalise to an unseen bridge?  
*Metric:* success, steps, energy

This bridge, seed two thousand and five, was never used in training or model selection, so it tests generalisation. In this run, the DQN succeeded after 118 steps, using 65 percent of the battery; the dense raster succeeded after 123 steps, using 63 percent of the battery. The route is different from the previous bridge because the cracks are in different places, so the agent is not replaying a memorised path. Over all hundred held-out bridges the original model succeeds on most of them, and its failures are battery-empty episodes. Three retrained copies of the same DQN do better: they succeed on 99 to 100 percent of these bridges.

## V4_agentA_real_images_seed2000  (`videos/V4_agentA_real_images_seed2000.mp4`, video 58 s, voice-over about 54 s)

*Question:* Does Agent A work with real-image uncertainty?  
*Metric:* success (mean uncertainty must reach the calibrated 0.282)

Now the same comparison uses the real-image detector: the uncertainty comes from a U-Net that looks at DeepCrack photographs. Watch the mean uncertainty, because it has to fall below a stricter threshold. In this run, the real-image DQN ran out of battery after 150 steps, using 100 percent of the battery; the dense raster succeeded after 125 steps, using 64 percent of the battery. Real-image uncertainty is noisier, and the high and mid altitudes give very poor images, so the drone has to fly low almost everywhere. The dense raster does exactly that, so it succeeds. Our DQN, even when trained on real images, succeeds on only a small share of bridges. This is our clearest weak result.

## V5_agentB_nominal_and_hardstart  (`videos/V5_agentB_nominal_and_hardstart.mp4`, video 58 s, voice-over about 50 s)

*Question:* Does PPO reach the best view (q=4), even from a hard start?  
*Metric:* steps to q=4, return

Agent B, our PPO refiner, must reach the best possible view of one suspicious cell by changing its distance, viewing angle and side offset. Watch the view quality climb to level four. The fixed orbit baseline keeps circling and never gets there. In this run PPO reached the best view in 4, 4 and 5 steps, the last one from the hard start, while the fixed orbit never did in 30 steps. The last part starts from a hard position: far away and at a large angle. PPO still reaches the best view in a few steps, almost exactly like the scripted upper bound that knows the true geometry.

## V6_agentC_mission_seed2000  (`videos/V6_agentC_mission_seed2000.mp4`, video 58 s, voice-over about 49 s)

*Question:* Which options does the A2C supervisor choose?  
*Metric:* option sequence (EXPLORE / REFINE / RETURN), defects, energy

Agent C, our A2C supervisor, decides at every step whether to explore, refine one cell, or return home. Watch the timeline of options at the bottom. In this run, the A2C supervisor succeeded after 18 decisions, using 51 percent of the battery; the rule supervisor succeeded after 34 decisions, using 95 percent of the battery. Notice that it only ever chooses explore and then return: this original model never chooses refine. Under our reward, refining costs energy and rarely finds a new crack, so it is hardly worth it. When we retrained with other seeds, some versions did use refine, and they collected the same return.

## V7_full_mission_learned_vs_rule  (`videos/V7_full_mission_learned_vs_rule.mp4`, video 58 s, voice-over about 50 s)

*Question:* Do the three agents work together?  
*Metric:* defects found, energy, whether Agent B is ever called

This is the complete system: the supervisor calls Agent A to explore and Agent B to refine. First the learned supervisor, then a rule supervisor driving the same learned Agents A and B. In this run, the A2C supervisor succeeded after 18 decisions, using 53 percent of the battery; the all-scripted system succeeded after 34 decisions, using 95 percent of the battery; the rule supervisor with learned Agents A and B succeeded after 32 decisions, using 94 percent of the battery. The learned hierarchy saves energy but never calls Agent B. With the rule supervisor, Agent B works inside the hierarchy, and that combination finds the most cracks.

## V8_detector_q0_to_q4  (`videos/V8_detector_q0_to_q4.mp4`, video 58 s, voice-over about 57 s)

*Question:* Why are real images harder?  
*Metric:* uncertainty and detection at view quality q0..q4

This clip shows why real images are harder. The same DeepCrack patch is shown at five view qualities, from four pixels wide up to sixty-four. The middle row is the crack the U-Net predicts; the bottom row is the spread of its dropout passes. Look at the crack patch: at four pixels the network misses the crack completely, yet the spread and the raw entropy printed below are both low, so it is confidently wrong. That is why the environment does not use these raw numbers: it ranks them and forces uncertainty to fall as view quality improves, so a blurry view always counts as uncertain. Since high and middle altitudes give the two worst qualities, the drone must fly low almost everywhere.

## V9_agentA_density35_seed2013  (`videos/V9_agentA_density35_seed2013.mp4`, video 58 s, voice-over about 46 s)

*Question:* Does Agent A cope with many more defects?  
*Metric:* success, battery

Here the bridge has thirty-five percent cracked cells instead of twenty. Cracked cells are more ambiguous for the detector, so the mean uncertainty falls more slowly. In this run, the DQN ran out of battery after 190 steps, using 100 percent of the battery; the dense raster succeeded after 123 steps, using 63 percent of the battery. On this bridge the DQN runs out of battery before it meets the threshold, while the dense raster finishes. Over a hundred bridges at this density the DQN's success rate drops, which shows the limit of a policy trained only at twenty percent.

## V10_agentA_battery0.6_seed2000  (`videos/V10_agentA_battery0.6_seed2000.mp4`, video 58 s, voice-over about 46 s)

*Question:* Is Agent A safe when the battery is low?  
*Metric:* battery-empty, success

This is our key robustness result. The drone starts with only sixty percent battery. Watch the battery bar on each side. In this run, the DQN succeeded after 81 steps, using 85 percent of the battery; the dense raster ran out of battery after 118 steps, using 100 percent of the battery. The dense raster needs more than sixty percent of a full battery for its fixed path, so it always runs out. The DQN sees its battery level in its observation and is rewarded for battery left, so it chooses a cheaper route and succeeds on most bridges.
