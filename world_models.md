# World Models for Autonomous Driving

**Part of:** [Autonomous Driving AI Models](autonomous_driving_AI_models.md)

---

## Table of Contents

- [Overview](#overview)
- [Open-Source World Models](#open-source-world-models)
  - [Video-Generative World Models](#video-generative-world-models)
  - [Occupancy / 4D World Models](#occupancy--4d-world-models)
  - [Latent / Representation World Models (JEPA-style)](#latent--representation-world-models-jepa-style)
  - [Trajectory / Scene-Token World Models](#trajectory--scene-token-world-models)
- [Architecture Overview](#architecture-overview)
- [Comparison by Representation](#comparison-by-representation)
- [Key Capabilities for AD Planning](#key-capabilities-for-ad-planning)
- [Python Integration](#python-integration)
- [Benchmark Datasets](#benchmark-datasets)

---

## Overview

World models learn a compressed, predictive representation of the driving environment. They simulate how the environment evolves over time, enabling autonomous systems to:

- **Predict** future scene states, agent trajectories, and occlusions
- **Plan** by rolling out hypothetical action sequences ("mental simulation")
- **Generate** realistic synthetic driving scenarios for training and validation
- **Evaluate** risk across counterfactual futures before committing to an action

A world model can be thought of as a learned differentiable simulator that operates on sensor observations rather than hand-crafted physics.

---

## Open-Source World Models

### Video-Generative World Models

| Model | Params | Input | Output | Temporal Horizon | Licence | Source |
|-------|--------|-------|--------|-----------------|---------|--------|
| **DriveDreamer** | ~1 B | Front/surround cam + route | Future video frames | 4–8 s | Apache-2.0 | [github.com/JeffWang987/DriveDreamer](https://github.com/JeffWang987/DriveDreamer) |
| **DriveDreamer-2** | ~1 B | Video + text prompt | Controllable HD video | 6–10 s | Research | [github.com/f1yfisher/DriveDreamer2](https://github.com/f1yfisher/DriveDreamer2) |
| **WoVogen** | ~800 M | Camera + BEV map + action | Future scene video | 3 s | Apache-2.0 | [github.com/fudan-zvg/WoVoGen](https://github.com/fudan-zvg/WoVoGen) |
| **Vista** | ~1.5 B | Multi-cam + ego action | Action-conditioned video | 3–5 s | Apache-2.0 | [github.com/OpenDriveLab/Vista](https://github.com/OpenDriveLab/Vista) |
| **DrivingDiffusion** | ~900 M | Layout + camera | Surround-view video generation | Single frame | Apache-2.0 | [github.com/shalfun/DrivingDiffusion](https://github.com/shalfun/DrivingDiffusion) |
| **GenAD** | ~500 M | Camera + map | Generalised future prediction | 3 s | Apache-2.0 | [github.com/wzzheng/GenAD](https://github.com/wzzheng/GenAD) |
| **SubjectDrive** | ~1 B | Camera + agent boxes | Subject-conditioned video | 4 s | Research | Preprint |
| **DiMA** | 7 B | Video + ego telemetry | Diffusion-based future video | 5 s | Research | Preprint |

---

### Occupancy / 4D World Models

| Model | Params | Input | Output | Licence | Source |
|-------|--------|-------|--------|---------|--------|
| **OccWorld** | ~120 M | Surround cams | 4D occupancy flow prediction | Apache-2.0 | [github.com/wzzheng/OccWorld](https://github.com/wzzheng/OccWorld) |
| **MUVO** | ~150 M | Camera + LiDAR | Metric voxel occupancy + velocity | MIT | [github.com/robot-learning-freiburg/MUVO](https://github.com/robot-learning-freiburg/MUVO) |
| **OccSim** | ~200 M | LiDAR + camera | Occupancy simulation | Apache-2.0 | Research |
| **UniWorld** | ~200 M | Multi-cam | BEV + temporal 3D occupancy | Apache-2.0 | [github.com/chaytonmin/UniWorld](https://github.com/chaytonmin/UniWorld) |
| **Cam4DOcc** | ~80 M | Multi-cam | 4D occupancy from cameras only | Apache-2.0 | [github.com/haomo-ai/Cam4DOcc](https://github.com/haomo-ai/Cam4DOcc) |

---

### Latent / Representation World Models (JEPA-style)

| Model | Params | Input | Approach | Licence | Source |
|-------|--------|-------|---------|---------|--------|
| **V-JEPA** | 632 M | Video | Video joint embedding; no pixel generation | CC BY-NC 4.0 | [github.com/facebookresearch/jepa](https://github.com/facebookresearch/jepa) |
| **I-JEPA** | 307 M | Images | Image-level latent prediction | CC BY-NC 4.0 | [github.com/facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) |
| **DreamerV3** | ~200 M | Any sensory stream | RSSM latent world model + RL | Apache-2.0 | [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3) |
| **TDMPC2** | ~50 M | Proprioception / images | Temporal-difference MPC planning | MIT | [github.com/nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) |

---

### Trajectory / Scene-Token World Models

| Model | Params | Input | Output | Licence | Source |
|-------|--------|-------|--------|---------|--------|
| **ScenarioNet** | ~100 M | Waymo/nuPlan logs | Scenario simulation | Apache-2.0 | [github.com/metadriverse/scenarionet](https://github.com/metadriverse/scenarionet) |
| **MotionDiffuser** | ~150 M | Agent states | Future trajectory distribution | Apache-2.0 | [github.com/WXinlong/MotionDiffuser](https://github.com/WXinlong/MotionDiffuser) |
| **MTR (Motion Transformer)** | ~60 M | LiDAR + agent history | Multi-modal trajectory prediction | Apache-2.0 | [github.com/sshaoshuai/MTR](https://github.com/sshaoshuai/MTR) |
| **Wayformer** | ~50 M | Agent history | Joint-marginal trajectory prediction | Apache-2.0 | Research release |
| **GPT-Driver** | 7 B | Agent tokens + language | LLM trajectory token prediction | Apache-2.0 | [github.com/PointsCoder/GPT-Driver](https://github.com/PointsCoder/GPT-Driver) |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    World Model Architecture (AD)                  │
│                                                                   │
│  t=0 (current observations)                                       │
│  ┌─────────────────────────────────────────────────┐             │
│  │          Encoder (Perception)                   │             │
│  │  Cameras → BEV features / latent z_t            │             │
│  │  LiDAR  → voxel / point cloud features          │             │
│  └────────────────────┬────────────────────────────┘             │
│                        │                                          │
│                        ▼                                          │
│  ┌─────────────────────────────────────────────────┐             │
│  │          State Space / Dynamics Model            │             │
│  │                                                  │             │
│  │  Option A — Generative:  Diffusion / GAN/       │             │
│  │            Transformer → predict future pixels  │             │
│  │                                                  │             │
│  │  Option B — Latent RSSM: LSTM / Transformer     │             │
│  │            propagates z_t → z_{t+1}             │             │
│  │                                                  │             │
│  │  Option C — Occupancy:  3D voxel → 4D flow      │             │
│  └──────────────┬───────────────────────────────────┘            │
│                 │                                                  │
│                 ▼  t=1, 2, …, T (future rollout)                 │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  Predicted future scenes (video / voxels / token sequences)  ││
│  │                                                               ││
│  │        ↓ planning head                                        ││
│  │  Trajectory optimisation / risk scoring / action selection    ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

---

## Comparison by Representation

| Approach | Strengths | Weaknesses | Best For |
|----------|-----------|-----------|----------|
| **Video generation** | Human-interpretable; data augmentation | Computationally heavy; pixel-level detail | Simulation, edge-case synthesis |
| **4D occupancy** | Compact; 3D-aware; real-time capable | No colour/apperance; poor at long range | Planning, collision prediction |
| **Latent RSSM** | Fast inference; RL-compatible | Hard to interpret; distribution shift | Closed-loop RL training |
| **Trajectory tokens** | Language-model compatible; scalable | Loses scene context | Interaction modelling |
| **JEPA latent** | No pixel reconstruction; efficient | Requires offline distillation | Representation pretraining |

---

## Key Capabilities for AD Planning

| Capability | Description | Relevant Models |
|-----------|-------------|----------------|
| **Counterfactual rollout** | "What if ego brakes instead of swerves?" | DreamerV3, Vista |
| **Scenario generation** | Synthesise rare / safety-critical scenes | DriveDreamer-2, WoVogen |
| **Risk quantification** | Probability of collision over horizon | OccWorld, MUVO |
| **Occlusion prediction** | Infer hidden agents behind obstacles | OccSim, UniWorld |
| **Interaction modelling** | Multi-agent joint trajectory prediction | MTR, Wayformer |
| **Data augmentation** | Generate novel training samples | DrivingDiffusion, GenAD |

---

## Python Integration

See [`world_model_client_example.py`](world_model_client_example.py) for a multi-provider world model client demonstrating next-state prediction, risk assessment, trajectory simulation, and counterfactual analysis.

```python
from world_model_client_example import WorldModelClient, SceneState, AgentState

client = WorldModelClient(provider="claude")
scene = SceneState(
    timestamp_s=0.0, ego_speed_kmh=90.0, ego_heading_deg=270.0,
    road_type="highway", speed_limit_kmh=120.0, weather="rain",
    agents=[AgentState("V1","vehicle",0.0,30.0,70.0,270.0,"follow_lane")],
)
pred = client.predict_next_state(scene, horizon_s=4.0)
print(f"Risk: {pred.risk_score:.2f}  |  Action: {pred.recommended_action}")
for p in pred.ego_trajectory:
    print(f"  t={p.t_s:.0f}s  Δy={p.y_m:+.0f}m  {p.speed_kmh:.0f} km/h")
```

---

## Benchmark Datasets

| Dataset | Scenes | Key Use | Licence |
|---------|--------|---------|---------|
| **nuScenes** | 1000 | 6-cam future prediction | CC BY-NC 4.0 |
| **Waymo Open Motion** | 100 K scenarios | Agent trajectory prediction | Custom (free research) |
| **nuPlan** | 1300 hr | Reactive closed-loop planning | CC BY-NC 4.0 |
| **CARLA Town** | Unlimited | Synthetic scene generation | MIT |
| **OpenDV-2M** | 2 M clips | Diverse world model pretraining | Research |
| **Waymo Sim Agents** | 32 K segments | Multi-agent simulation | Custom |

---

*See also: [VLM Models](VLM_models.md) | [VLA Models](VLA_models.md) | [Foundation Models](foundation_models.md) | [Edge Deployment](edge_deployment.md)*
