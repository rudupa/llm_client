# Vision-Language-Action (VLA) Models for Autonomous Driving

**Part of:** [Autonomous Driving AI Models](autonomous_driving_AI_models.md)

---

## Table of Contents

- [Overview](#overview)
- [Open-Source VLA Models](#open-source-vla-models)
  - [Robotics & Generalised VLAs (transferable to AD)](#robotics--generalised-vlas-transferable-to-ad)
  - [Autonomous Driving VLAs](#autonomous-driving-vlas)
- [Architecture Overview](#architecture-overview)
  - [Action Representation Types](#action-representation-types)
- [Training Data Sources](#training-data-sources)
- [Key Metrics](#key-metrics)
- [Python Integration](#python-integration)
- [Model Size vs Latency (Jetson AGX Orin)](#model-size-vs-latency-jetson-agx-orin)

---

## Overview

Vision-Language-Action (VLA) models extend VLMs by adding an **action head** that outputs structured, executable control commands conditioned on visual observations, language instructions, and ego-vehicle state. They close the perception–reasoning–action loop in a single neural network.

In autonomous driving, VLAs translate surround-view camera frames, route instructions, and vehicle telemetry directly into:
- Trajectory waypoints
- Steering / throttle / brake commands
- High-level manoeuvre decisions (lane change, turn, stop)
- Natural-language justification of the selected action

---

## Open-Source VLA Models

### Robotics & Generalised VLAs (transferable to AD)

| Model | Params | Input Modalities | Action Space | Licence | Source |
|-------|--------|-----------------|-------------|---------|--------|
| **OpenVLA-7B** | 7 B | RGB + language instruction | 7-DoF continuous + gripper | MIT | [github.com/openvla/openvla](https://github.com/openvla/openvla) |
| **OpenVLA-OFT-7B** | 7 B | RGB + instruction | Fine-tuning derivative | MIT | [HuggingFace: openvla](https://huggingface.co/openvla) |
| **Octo (base)** | 93 M | RGB + wrist cam + language | Diffusion-based multi-task | Apache-2.0 | [github.com/octo-models/octo](https://github.com/octo-models/octo) |
| **Octo (small)** | 27 M | RGB + language | Lightweight variant | Apache-2.0 | [github.com/octo-models/octo](https://github.com/octo-models/octo) |
| **RoboVLMs** | 7 B | Image + text | Unified robot VLA framework | Apache-2.0 | [github.com/Robot-VLAs/RoboVLMs](https://github.com/Robot-VLAs/RoboVLMs) |
| **GR-2** | 7 B | Video + language | Video-conditioned manipulation | Apache-2.0 | [github.com/bytedance/GR-2](https://github.com/bytedance/GR-2) |
| **SpatialVLA** | 4 B | Spatial-aware images + text | 2D spatial action head | Apache-2.0 | [github.com/SpatialVLA/SpatialVLA](https://github.com/SpatialVLA/SpatialVLA) |
| **TinyVLA** | 1.1 B | RGB + instruction | Fast inference VLA | Apache-2.0 | [github.com/lesjie-wen/tiny-vla](https://github.com/lesjie-wen/tiny-vla) |
| **SmolVLA** | 450 M | RGB + text | On-device VLA < 1 B params | Apache-2.0 | [HuggingFace: SmolVLA](https://huggingface.co/lerobot) |
| **π₀ (pi-zero)** | 3 B | Dual-arm RGB + text | Flow-matching action diffusion | Research (weights restricted) | [physicalintelligence.company](https://physicalintelligence.company) |

---

### Autonomous Driving VLAs

| Model | Params | Input | Output | Licence | Source |
|-------|--------|-------|--------|---------|--------|
| **UniAD** | ~130 M | Surround cams + LiDAR | Waypoints + occupancy + tracking | Apache-2.0 | [github.com/OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD) |
| **VAD** | ~60 M | Multi-cam + HD map | Vectorised scene + trajectory | Apache-2.0 | [github.com/hustvl/VAD](https://github.com/hustvl/VAD) |
| **SparseDrive** | ~50 M | Surround cams | Sparse 3D occup + future traj | Apache-2.0 | [github.com/swc-17/SparseDrive](https://github.com/swc-17/SparseDrive) |
| **DriveCoT** | 7 B | Camera + language | Chain-of-thought → waypoints | Apache-2.0 | [github.com/NVlabs/DriveCoT](https://github.com/NVlabs/DriveCoT) |
| **DiMA** | 7 B | Video + ego data | Diffusion manoeuvre actions | MIT | Research preprint |
| **DriveDreamer-2 VLA** | 7 B | Video + text prompt | Future video + control signal | Research | [github.com/f1yfisher/DriveDreamer2](https://github.com/f1yfisher/DriveDreamer2) |
| **ThinkTwice** | 7 B | Front cam + language | Two-stage CoT + action | Apache-2.0 | [github.com/OpenDriveLab/ThinkTwice](https://github.com/OpenDriveLab/ThinkTwice) |
| **DriveX** | 7 B | Surround + text | Unified action reasoning | Research | Paper + weights on request |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      VLA Architecture (AD)                       │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────────┐│
│  │  Perception  │   │   Language   │   │     Action Head        ││
│  │             │   │   Backbone   │   │                        ││
│  │ Surround    │──▶│             │──▶│  Waypoint regression   ││
│  │ Cameras     │   │  LLaMA /     │   │  Steer/throttle/brake  ││
│  │ (6–12 cams) │   │  Mistral /   │   │  Lane-change token     ││
│  │             │   │  Qwen2       │   │  Trajectory diffusion  ││
│  │ LiDAR BEV   │   │             │   │                        ││
│  │ (optional)  │   │  +Adapters   │   │  Action tokens or      ││
│  │             │   │             │   │  continuous output      ││
│  │ HD Map      │──▶│             │──▶│                        ││
│  │ Ego state   │   │             │   │  Safety verifier       ││
│  │ Route instr │   │             │   │  (rule-based fallback) ││
│  └─────────────┘   └──────────────┘   └───────────────────────┘│
│                                               │                  │
│                                               ▼                  │
│                              ┌─────────────────────────────────┐│
│                              │         Vehicle Controller       ││
│                              │  (PID / MPC / lateral + long)   ││
│                              └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Action Representation Types

| Type | Description | Models Using It |
|------|-------------|----------------|
| **Waypoint regression** | Sequence of (x, y) future positions | UniAD, VAD, DriveLM |
| **Tokenised action** | Discrete action token prediction | OpenVLA, RT-2 |
| **Continuous regression** | Direct steer/throttle/brake scalars | Octo, TinyVLA |
| **Diffusion action head** | Denoising trajectory from noise | π₀, Octo-diffusion |
| **Chain-of-thought token** | Language reasoning → final action | DriveCoT, ThinkTwice |

---

## Training Data Sources

| Dataset | Frames / Hours | Modalities | Licence |
|---------|---------------|-----------|--------|
| **nuScenes** | 400 K frames, 1000 scenes | 6 cameras, LiDAR, radar | CC BY-NC 4.0 |
| **Waymo Open Dataset** | 1 000+ segments | 5 cameras, LiDAR | Custom (free research) |
| **CARLA Simulator** | Unlimited synthetic | Cameras, LiDAR, semantic | MIT |
| **nuPlan** | 1300 hr driving logs | 8 cameras + LiDAR | CC BY-NC 4.0 |
| **OpenX-Embodiment** | 1 M robot episodes | Multi-robot / multi-task | Apache-2.0 |
| **DriveX Dataset** | 3 M frames | 8 cams + GPS/IMU | Research |
| **NAVSIM** | 1100 scenes | nuPlan-based reactive | CC BY-NC 4.0 |
| **nuScenes-QA** | 34 K QA pairs | Language annotations | CC BY-NC 4.0 |

---

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **L2 displacement (1s/2s/3s)** | Waypoint error vs ground-truth | < 0.5 / 0.8 / 1.5 m |
| **Collision rate** | % of scenarios with collision | < 0.5 % |
| **Route completion** | % of routes fully completed | > 95 % |
| **PDMS (NAVSIM)** | Non-reactive closed-loop score | > 85 |
| **Reaction time** | End-to-end inference latency | < 50 ms at 20 Hz |

---

## Python Integration

See [`vla_client_example.py`](vla_client_example.py) for a multi-provider VLA implementation returning structured `DrivingAction` objects.

```python
from vla_client_example import VLAClient, EgoState

client = VLAClient(provider="gemini")  # or "claude" / "ollama"
ego = EgoState(speed_kmh=90.0, heading_deg=270.0,
               lane="center", turn_signal="left", acceleration_ms2=0.0)
action = client.predict_action(
    scene="Wet motorway, slow truck 25 m ahead in centre lane, left lane clear.",
    ego=ego,
)
print(action.maneuver, action.steering_angle_deg, action.reasoning)
```

---

## Model Size vs Latency (Jetson AGX Orin)

| Model | Params | FP16 ms | INT8 ms | INT4 ms |
|-------|--------|---------|---------|---------|
| SmolVLA | 450 M | 12 ms | 7 ms | 4 ms |
| TinyVLA | 1.1 B | 22 ms | 12 ms | 7 ms |
| Octo-small | 27 M | 3 ms | 2 ms | — |
| Octo-base | 93 M | 8 ms | 5 ms | — |
| SpatialVLA | 4 B | 95 ms | 50 ms | 28 ms |
| OpenVLA-7B | 7 B | 160 ms | 85 ms | 48 ms |
| DriveCoT-7B | 7 B | 180 ms | 95 ms | 55 ms |

---

*See also: [VLM Models](VLM_models.md) | [World Models](world_models.md) | [Foundation Models](foundation_models.md) | [Edge Deployment](edge_deployment.md)*
