| system_component | setting | value | source_artifact |
|---|---|---|---|
| Simulation | SIM_SEED | 20260817 | SimulationManager.java |
| Simulation | maximum simulation seconds | 1500.0 | SimulationManager.java |
| Simulation | device count | 10 | SimulationManager.java |
| Failure predictor | sequence length | 10 | failure_predictor_meta.json |
| Failure predictor | feature count | 26 | failure_predictor_meta.json |
| Failure predictor | threshold | 0.18 | failure_predictor_meta.json |
| MAPPO | seed | 20260818 | mappo_config.json |
| MAPPO | device | cpu | mappo_config.json |
| MAPPO | training episodes | 600 | mappo_config.json |
| MAPPO | rollout episodes/update | 8 | mappo_config.json |
| MAPPO | PPO updates | 75 | derived from mappo_config.json |
| MAPPO | episode steps | 400 | mappo_config.json |
| MAPPO | gamma | 0.999 | mappo_config.json |
| MAPPO | gae lambda | 0.995 | mappo_config.json |
| MAPPO | risk source | oof | mappo_config.json |
| MAPPO | actors | 10 separate MLP(48 -> [128, 128] -> 4) | run_mappo_train.log |
| MAPPO | centralised critic | MLP(489+10 -> [256, 256] -> 1) | run_mappo_train.log |
| MAPPO | action space | STAY; MIGRATE_TO_NEIGHBOR_EDGE; MIGRATE_TO_CLOUD; PREEMPTIVE_REROUTE | run_mappo_train.log |
| Evaluation | fixed starts | [491, 521, 550, 580, 609, 639, 668, 698] | mappo_eval.json |
