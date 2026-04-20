# Autonomous Driving AI Models: Comprehensive Guide

A structured reference for open-source AI model families used in autonomous driving, from perception through planning to deployment.

---

## Model Family Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Autonomous Driving AI Model Stack                         │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                   FOUNDATION MODELS                                    │  │
│  │  Pre-trained on internet-scale data; provide world knowledge           │  │
│  │  → UniAD, DriveVLM, DriveLM, Dolphins, DriveMLM                       │  │
│  └────────────────────┬──────────────────────────────────────────────────┘  │
│                        │ fine-tuned / specialised                            │
│         ┌──────────────┼──────────────────┐                                 │
│         ▼              ▼                  ▼                                  │
│  ┌────────────┐ ┌────────────┐  ┌─────────────────┐                         │
│  │   VLM      │ │    VLA     │  │  WORLD MODELS   │                         │
│  │ Perception │ │  Planning  │  │  Prediction /   │                         │
│  │ & Scene    │ │ & Control  │  │  Simulation     │                         │
│  │ Understanding│ │           │  │                 │                         │
│  └────────────┘ └────────────┘  └─────────────────┘                         │
│         │              │                  │                                  │
│         └──────────────┴──────────────────┘                                 │
│                        │ compress + deploy                                   │
│                        ▼                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │         EDGE DEPLOYMENT (Teacher–Student, Quantisation)               │  │
│  │  NVIDIA Jetson Orin / DRIVE Thor   │   Qualcomm SA8775P               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Document Index

| Document | What it Covers | Key Models |
|----------|---------------|-----------|
| [VLM Models](VLM_models.md) | Vision-Language Models for scene understanding, hazard detection, multi-camera analysis | LLaVA, Qwen2-VL, InternVL2, Moondream, DriveLM-VLM |
| [VLA Models](VLA_models.md) | Vision-Language-Action Models for driving decisions and control commands | OpenVLA, Octo, GR-2, TinyVLA, OpenDriveVLA |
| [World Models](world_models.md) | Future state prediction, scene generation, temporal simulation | DriveDreamer, Vista, OccWorld, GenAD, MUVO |
| [Foundation Models](foundation_models.md) | Large pre-trained base models fine-tuned for AD; end-to-end planners | UniAD, DriveVLM, Dolphins, DriveMLM, DriveCoT |
| [Edge Deployment](edge_deployment.md) | Teacher–student distillation, quantisation, pruning, hardware targets | TensorRT, AIMET, AutoGPTQ, Jetson Orin, SA8775P |
| [AD Model Architectures](ad_model_architectures.md) | NVIDIA models, component internals (BEV, DETR heads, VLM connectors), architecture combinations, joint E2E training recipes | Cosmos, Hydra-MDP, BEVFusion, UniAD, BEVFormer, PPO/GRPO |

---

## Quick Comparison: Model Family by Role

| Family | Input | Output | AD Role | Latency Target | Min. Edge HW |
|--------|-------|--------|---------|---------------|-------------|
| **VLM** | Image(s) + text | Text (scene description, Q&A) | Perception, logging, SV | 50–300 ms | Jetson Orin NX 8G |
| **VLA** | Image(s) + text + state | Structured action / waypoints | Decision, low-level control | 20–80 ms | Jetson Orin NX 16G |
| **World Model** | History frames + actions | Future frames / occupancy | Prediction, simulation | 30–150 ms | Jetson AGX Orin 32G |
| **Foundation** | Multi-modal sensor suite | Planning trajectory / tokens | End-to-end planning | 50–200 ms | Jetson AGX Orin 64G |
| **Edge (Student)** | Same as parent | Same as parent | Real-time production | < 50 ms | Jetson Orin NX 8G |

---

## Open-Source Model Quick Reference

### Vision-Language Models (VLM)

| Model | Params | Licence | Specialised for AD | Link |
|-------|--------|---------|-------------------|------|
| Moondream 2 | 1.86B | Apache 2.0 | No (general) | [HuggingFace](https://huggingface.co/vikhyatk/moondream2) |
| InternVL2-1B | 1B | MIT | No | [HuggingFace](https://huggingface.co/OpenGVLab/InternVL2-1B) |
| InternVL2-4B | 4B | MIT | No | [HuggingFace](https://huggingface.co/OpenGVLab/InternVL2-4B) |
| InternVL2-8B | 8B | MIT | No | [HuggingFace](https://huggingface.co/OpenGVLab/InternVL2-8B) |
| Qwen2-VL-2B | 2B | Apache 2.0 | No | [HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| Qwen2-VL-7B | 7B | Apache 2.0 | No | [HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| PaliGemma 2 3B | 3B | Gemma Licence | No | [HuggingFace](https://huggingface.co/google/paligemma2-3b-pt-224) |
| LLaVA-NeXT-7B | 7B | Apache 2.0 | No | [HuggingFace](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) |
| Phi-3.5-vision | 4.2B | MIT | No | [HuggingFace](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) |
| DriveLM-Agent | 7B | Apache 2.0 | **Yes** | [GitHub](https://github.com/OpenDriveLab/DriveLM) |

*→ Full details: [VLM_models.md](VLM_models.md)*

---

### Vision-Language-Action Models (VLA)

| Model | Params | Licence | AD-ready | Link |
|-------|--------|---------|---------|------|
| OpenVLA | 7B | MIT | Partial | [GitHub](https://github.com/openvla/openvla) |
| TinyVLA | 1.1B | MIT | Partial | [GitHub](https://github.com/lesjie-wen/tinyvla) |
| Octo | 93M | Apache 2.0 | Partial | [GitHub](https://github.com/octo-models/octo) |
| OpenDriveVLA | 7B | Apache 2.0 | **Yes** | Research |
| π₀ (Pi Zero) | 3B | Research | Partial | [Physical Intelligence](https://www.physicalintelligence.company/blog/pi0) |

*→ Full details: [VLA_models.md](VLA_models.md)*

---

### World Models

| Model | Params | Licence | Output | Link |
|-------|--------|---------|--------|------|
| DriveDreamer-2 | ~1B | Apache 2.0 | Video frame prediction | [GitHub](https://github.com/f1yfisher/DriveDreamer2) |
| Vista | ~1B | CC-BY-NC 4.0 | Driving video generation | [GitHub](https://github.com/OpenDriveLab/Vista) |
| OccWorld | ~200M | Apache 2.0 | 4D occupancy flow | [GitHub](https://github.com/wzzheng/OccWorld) |
| GenAD | ~500M | Apache 2.0 | Generalised scenario gen | Research |
| MUVO | ~300M | MIT | Lidar+Camera world model | [GitHub](https://github.com/robot-learning-freiburg/MUVo) |
| WoVogen | ~1B | Apache 2.0 | World + video co-gen | Research |

*→ Full details: [world_models.md](world_models.md)*

---

### Foundation Models (AD End-to-End)

| Model | Params | Licence | Approach | Link |
|-------|--------|---------|---------|------|
| UniAD | ~170M | Apache 2.0 | Unified AD query transformer | [GitHub](https://github.com/OpenDriveLab/UniAD) |
| VAD | ~200M | Apache 2.0 | Vectorised AD | [GitHub](https://github.com/hustvl/VAD) |
| SparseDrive | ~100M | Apache 2.0 | Sparse representation | [GitHub](https://github.com/swc-17/SparseDrive) |
| DriveVLM | 7B+ | Research | VLM-based dual pipeline | Research |
| Dolphins | 3.5B | Research | Multi-modality AD VLM | [GitHub](https://github.com/SaFoLab-WISC/Dolphins) |
| DriveMLM | 7B | Apache 2.0 | Multi-modal language model | [GitHub](https://github.com/OpenGVLab/DriveMLM) |
| DriveCoT | 7B | Research | Chain-of-thought driving | Research |

*→ Full details: [foundation_models.md](foundation_models.md)*

---

### Edge Deployment Highlights

| Student Model | Teacher | Compression | Target SoC | Latency |
|--------------|---------|-------------|-----------|---------|
| Qwen2-VL-2B INT4 | Qwen2-VL-72B | 36× | Jetson Orin NX 8G | 35 ms |
| InternVL2-1B INT8 | InternVL2-8B | 8× | Jetson Orin NX 8G | 15 ms |
| TinyVLA-1.1B INT8 | OpenVLA-7B | 7× | Jetson AGX Orin | 22 ms |
| OccWorld-tiny INT8 | OccWorld-full | 6× | Jetson AGX Orin | 25 ms |

*→ Full details: [edge_deployment.md](edge_deployment.md)*

---

## AD Software Stack: Where Each Model Type Fits

```
Sensors (LiDAR, cameras, radar, GPS/IMU)
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  PERCEPTION LAYER                                      │
│  VLM: scene description, object detection, Q&A        │
│  Foundation: BEV feature extraction, detection        │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│  PREDICTION LAYER                                      │
│  World Models: future state, occupancy prediction     │
│  Foundation: agent trajectory prediction              │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│  PLANNING LAYER                                        │
│  VLA: waypoint generation, action tokens              │
│  Foundation: end-to-end planning with reasoning       │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│  CONTROL LAYER                                         │
│  VLA: throttle/brake/steer commands                   │
│  Classical PID / MPC controller fallback              │
└───────────────────────────────────────────────────────┘
                            │
                            ▼
                    Vehicle Actuators
```

---

## Key Datasets for Training & Evaluation

| Dataset | Scale | Modalities | Tasks | Licence |
|---------|-------|-----------|-------|---------|
| **nuScenes** | 1000 scenes, 1.4M bbox | Camera, LiDAR, radar | Detection, tracking, prediction | CC-BY-NC |
| **nuPlan** | 1500h driving | Camera, LiDAR, maps | Planning | CC-BY-NC |
| **Waymo Open Dataset** | 1000 segments, 12M bbox | Camera, LiDAR | Detection, tracking, motion | Waymo terms |
| **CARLA synthetic** | Unlimited (sim) | All sensor types | All tasks | MIT |
| **BDD100K** | 100K videos | Camera | Detection, tracking, lane | BSD |
| **Argoverse 2** | 1000 logs | Camera, LiDAR | Motion forecasting, detection | CC-BY-NC |
| **DriveLM-nuScenes** | nuScenes subset | Camera + QA pairs | VLM driving Q&A | Apache 2.0 |
| **DROID** | 76K trajectories | Camera, proprioception | VLA robot manipulation | Apache 2.0 |

---

## Getting Started with This Repository

```bash
# Clone and set up environment
git clone https://github.com/rudupa/llm_client
cd llm_client
python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt --index-url https://pypi.org/simple/

# Try the example scripts
python vlm_client_example.py        # VLM scene understanding
python vla_client_example.py        # VLA driving decisions
python world_model_client_example.py # World model predictions
```

---

## Related Documentation

- [VLM Models in Detail](VLM_models.md)
- [VLA Models in Detail](VLA_models.md)
- [World Models in Detail](world_models.md)
- [Foundation Models in Detail](foundation_models.md)
- [Edge Deployment & Teacher–Student](edge_deployment.md)
- [Design Document](DESIGN_DOCUMENT.md)
- [README](README.md)
