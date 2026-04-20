# Autonomous Driving Model Architectures: Internals, Combinations & Joint Training

A deep technical reference covering how AD AI models are built internally, how components are combined into full architectures, and how they are trained end-to-end — with emphasis on NVIDIA's model ecosystem.

---

## Table of Contents

- [1. The Four Architectural Paradigms](#1-the-four-architectural-paradigms)
- [2. NVIDIA Autonomous Driving Models](#2-nvidia-autonomous-driving-models)
  - [2.1 NVIDIA Cosmos (2025) — Physical World Foundation Model](#21-nvidia-cosmos-2025--physical-world-foundation-model)
  - [2.2 NVIDIA Hydra-MDP (2024) — Multi-task Driving Policy](#22-nvidia-hydra-mdp-2024--multi-task-driving-policy)
  - [2.3 NVIDIA BEVFusion (Camera-LiDAR Fusion)](#23-nvidia-bevfusion-camera-lidar-fusion)
  - [2.4 NVIDIA DRIVE Platform Models](#24-nvidia-drive-platform-models)
  - [2.5 NVIDIA Cosmos + DriveVLM Integration Pattern](#25-nvidia-cosmos--drivevlm-integration-pattern)
- [3. Component Internals: How Each Module Is Built](#3-component-internals-how-each-module-is-built)
  - [3.1 Sensor Encoders](#31-sensor-encoders)
    - [Multi-Modal Sensor Suite](#multi-modal-sensor-suite)
    - [Camera Encoder](#camera-encoder-vision-transformer-path)
  - [3.2 BEV Temporal Encoder (BEVFormer-style)](#32-bev-temporal-encoder-bevformer-style)
  - [3.3 Perception Heads (DETR-style)](#33-perception-heads-detr-style)
  - [3.4 Motion Prediction Head (MTR-style)](#34-motion-prediction-head-mtr-style)
  - [3.5 Planning Head (Waypoint Generation)](#35-planning-head-waypoint-generation)
  - [3.6 VLM Integration (Foundation Model Path)](#36-vlm-integration-foundation-model-path)
- [4. Architecture Combinations: Component Assembly](#4-architecture-combinations-component-assembly)
  - [Combination A: Camera-Only BEV E2E (BEVFormer + UniAD)](#combination-a-camera-only-bev-e2e-bevformer--uniad)
  - [Combination B: LiDAR-Camera Fusion (BEVFusion + CenterPoint + MTR)](#combination-b-lidar-camera-fusion-bevfusion--centerpoint--mtr)
  - [Combination C: VLM-Augmented Dual Pipeline (DriveVLM-style)](#combination-c-vlm-augmented-dual-pipeline-drivevlm-style)
  - [Combination D: Cosmos World Model + RL Policy (NVIDIA Reference Pipeline)](#combination-d-cosmos-world-model--rl-policy-nvidia-reference-pipeline)
  - [Combination E: Occupancy + Diffusion Planner (OccWorld + DriveDreamer)](#combination-e-occupancy--diffusion-planner-occworld--drivedreamer)
- [5. Joint End-to-End Training](#5-joint-end-to-end-training)
  - [5.1 Multi-Task Loss](#51-multi-task-loss)
  - [5.2 Imitation Learning (Behavioural Cloning)](#52-imitation-learning-behavioural-cloning)
  - [5.3 Reinforcement Learning Fine-Tuning](#53-reinforcement-learning-fine-tuning)
  - [5.4 Query Propagation (UniAD / VAD joint training detail)](#54-query-propagation-uniad--vad-joint-training-detail)
  - [5.5 VLM Joint Training (Visual Instruction Tuning)](#55-vlm-joint-training-visual-instruction-tuning)
  - [5.6 World Model Joint Pre-training (Cosmos Recipe)](#56-world-model-joint-pre-training-cosmos-recipe)
- [6. Other Industry E2E Models](#6-other-industry-e2e-models)
- [7. Architecture Selection Guide](#7-architecture-selection-guide)

---

## 1. The Four Architectural Paradigms

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  PARADIGM 1: Classic Modular Pipeline                                          │
│                                                                                │
│  Sensors → [Perception] → [Prediction] → [Planning] → [Control] → Actuators   │
│             hand-crafted  hand-crafted   hand-crafted  PID/MPC                 │
│             modules       modules        modules                               │
│                                                                                │
│  ✓ Interpretable   ✗ Error accumulates   ✗ No shared representation           │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  PARADIGM 2: Hybrid (Modular-E2E)                                              │
│                                                                                │
│  Sensors → [Shared Backbone] → [Task Heads: detect, segment, predict, plan]   │
│                     ↑                                                           │
│                Jointly trained, separate decoder heads, shared BEV repr        │
│                                                                                │
│  Examples: BEVFusion, BEVFormer, UniAD, VAD, SparseDrive                      │
│  ✓ Shared features  ✓ Multi-task  ✗ Planning head still auxiliary             │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  PARADIGM 3: Foundation-Model Augmented E2E                                    │
│                                                                                │
│  Sensors + Language → [VLM Backbone] → [World Model] → [VLA Action Head]      │
│                              ↑                                                  │
│          Pre-trained internet-scale; AD fine-tuned                             │
│                                                                                │
│  Examples: DriveVLM, Dolphins, DriveMLM, OpenVLA, LINGO-2, DriveLM            │
│  ✓ Common-sense reasoning  ✓ Language-guided  ✗ Latency on edge               │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  PARADIGM 4: Generative + Diffusion World Model                                │
│                                                                                │
│  Past Frames + State → [World Foundation Model] → Future Frames / Occupancy   │
│                                    ↑                                            │
│          Diffusion or autoregressive video generation                          │
│                                                                                │
│  Examples: NVIDIA Cosmos, DriveDreamer-2, Vista, WoVogen, GAIA-1              │
│  ✓ Unlimited synthetic data  ✓ Counterfactual sim  ✗ Inference cost           │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. NVIDIA Autonomous Driving Models

### 2.1 NVIDIA Cosmos (2025) — Physical World Foundation Model

**Type:** Generative World Foundation Model (Paradigm 4)
**Release:** January 2025 | **Licence:** NVIDIA Open Model Licence
**GitHub:** [github.com/NVIDIA/Cosmos](https://github.com/NVIDIA/Cosmos)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA Cosmos Architecture                             │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  Cosmos Tokenizer                                        │                │
│  │  Continuous Video → Discrete / Continuous latent tokens  │                │
│  │  8× temporal, 8× spatial compression                     │                │
│  │  Causal 3D CNN + VQ-VAE / FSQ                            │                │
│  └───────────────────────────┬─────────────────────────────┘                │
│                               │ latent tokens                                │
│         ┌─────────────────────┼──────────────────────────────┐              │
│         ▼                     ▼                              ▼              │
│  ┌──────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐   │
│  │  Cosmos-1.0  │  │  Cosmos-1.0           │  │  Cosmos-1.0             │   │
│  │  Autoregressive│  │  Diffusion (DiT)     │  │  Video2World            │   │
│  │  4B / 12B    │  │  7B / 14B            │  │  Conditional generation │   │
│  │  Next-token  │  │  Denoising in latent │  │  Action-conditioned     │   │
│  │  prediction  │  │  space (DDPM/Flow)   │  │  video prediction       │   │
│  └──────────────┘  └──────────────────────┘  └─────────────────────────┘   │
│                                                                               │
│  Use in AD:                                                                   │
│  • Generate synthetic training data for rare/dangerous scenarios              │
│  • Simulate future sensor observations (camera video prediction)              │
│  • World model simulator for RL policy training                               │
│  • Action-conditioned rollout: "if I brake here, what happens?"               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Internal Architecture — Cosmos Diffusion (DiT backbone):**

```
Input: Past N frames (tokenised) + action condition + text prompt
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Cosmos Tokenizer            │
         │  3D Causal Conv encoder      │
         │  Output: latent tensor       │
         │  shape [B, T/8, H/8, W/8, C] │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Diffusion Transformer (DiT) │
         │                              │
         │  • 3D RoPE positional embed  │
         │  • Full spatiotemporal attn  │
         │  • T5 text encoder for cond  │
         │  • Timestep (noise level) emb│
         │  • Adaptive LayerNorm (adaLN)│
         │  • N × DiT blocks            │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Cosmos Tokenizer Decoder    │
         │  Latent → pixel-space frames │
         └──────────────────────────────┘
                        │
                        ▼
                Future video frames
```

---

### 2.2 NVIDIA Hydra-MDP (2024) — Multi-task Driving Policy

**Type:** Hybrid Modular-E2E (Paradigm 2) with imitation learning
**Paper:** "Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation"

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Hydra-MDP Architecture                                    │
│                                                                               │
│  Multi-camera images + LiDAR + HD Map + Ego state                            │
│           │                                                                   │
│           ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  BEV Feature Extractor                                               │     │
│  │  • Camera branch: BEVFormer / LSS lift to BEV                        │     │
│  │  • LiDAR branch: VoxelNet → BEV pillars                              │     │
│  │  • Fusion: element-wise add or cross-attention                       │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│           │                                                                   │
│           ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                       Temporal Encoder                                │    │
│  │  GRU / Transformer — fuses T past BEV frames into one state repr     │    │
│  └───────────────────────────────────────┬──────────────────────────────┘    │
│                                          │                                    │
│         ┌──────────────────────┬─────────┴───────────┬──────────────────┐   │
│         ▼                      ▼                      ▼                  ▼   │
│  ┌─────────────┐  ┌─────────────────────┐  ┌────────────────┐  ┌──────────┐ │
│  │  Detection  │  │  Motion Prediction  │  │  Map Segment.  │  │ Planning │ │
│  │  Head       │  │  Head               │  │  Head          │  │  Head    │ │
│  │  DETR-style │  │  Multi-modal traj.  │  │  Semantic BEV  │  │ Waypoints│ │
│  └─────────────┘  └─────────────────────┘  └────────────────┘  └──────────┘ │
│                                                       ↑                       │
│                 Multi-target Hydra Distillation ──────┘                      │
│                 (multiple teacher models, one student)                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key innovation — Hydra Distillation:**
Multiple specialist teacher models (one per task) distill into a single multi-head student. The planning head is supervised by both imitation loss against expert trajectories *and* distillation loss from a specialist trajectory teacher.

---

### 2.3 NVIDIA BEVFusion (Camera-LiDAR Fusion)

**Type:** Perception backbone (used inside larger E2E systems)
**Paper:** "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Repr."

```
Camera Images (6 views)       LiDAR Point Cloud
        │                             │
        ▼                             ▼
 ┌─────────────────┐        ┌──────────────────────┐
 │  Image Encoder  │        │  VoxelNet / SECOND    │
 │  Swin-T or R50  │        │  Sparse 3D Conv       │
 │  FPN neck       │        │  → BEV pillar features│
 └────────┬────────┘        └──────────┬───────────┘
          │                            │
          ▼ LSS view transform         ▼
 ┌─────────────────────────────────────────────────┐
 │  Camera BEV feature [H×W×C_cam]                 │
 └──────────────────────┬──────────────────────────┘
                        │
              Channel-wise concat
                        │
                        ▼
 ┌─────────────────────────────────────────────────┐
 │  BEV Encoder (Residual ConvNet / Transformer)   │
 │  Fused BEV feature [H×W×(C_cam+C_lid)]         │
 └──────────────────────┬──────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
   3D Object Detection           BEV Map Segmentation
   (CenterPoint head)            (Semantic heads)
```

---

### 2.4 NVIDIA DRIVE Platform Models

| Component | Model / Technology | Role |
|-----------|-------------------|------|
| **DriveWorks Perception** | CNN + Transformer stereo/mono depth + detection | Production ADAS perception |
| **DriveWorks DNN** | TensorRT-optimised detection + segmentation nets | Real-time on DRIVE Orin SoC |
| **DriveAGX Planner** | Rule-based + IL waypoint network | Motion planning on DRIVE |
| **DriveGPT** (internal) | GPT-4 class LLM fine-tuned on driving logs | Scene reasoning, disambiguation |
| **NVIDIA TAO Toolkit** | Transfer learning + pruning framework | Domain adaptation of DINO/ViT |
| **DRIVE Sim / Omniverse** | USD-based deterministic + stochastic simulation | Synthetic training data generation |
| **Cosmos (Cosmos-Transfer)** | Diffusion world model | eDRIVE Sim extension, sensor sim |

---

### 2.5 NVIDIA Cosmos + DriveVLM Integration Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  NVIDIA Reference Integration: Cosmos World Model + VLM Planning             │
│                                                                              │
│  TRAINING TIME:                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Cosmos Video2World                                                   │  │
│  │  Pre-trained on 20M hours of video                                   │  │
│  │  Fine-tuned on: nuPlan / Waymo + CARLA synthetic                     │  │
│  │  → Learns physics, agent behaviours, sensor dynamics                  │  │
│  └────────────────────────────┬─────────────────────────────────────────┘  │
│                                │  generates rollouts                        │
│                                ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  RL Policy Training (PPO / GRPO)                                     │  │
│  │  Reward: comfort + safety + efficiency                               │  │
│  │  Environment: Cosmos world model (differentiable simulator)          │  │
│  └────────────────────────────┬─────────────────────────────────────────┘  │
│                                │  trained policy                            │
│  INFERENCE TIME:               ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  VLM Scene Reasoner (Qwen2-VL / DriveGPT)                           │  │
│  │  • Interprets complex scene: "construction zone with flagman"        │  │
│  │  • Sets high-level intent token                                      │  │
│  └────────────────────────────┬─────────────────────────────────────────┘  │
│                                │  scene intent                              │
│                                ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  VLA Planning Head (RL-trained student model)                        │  │
│  │  • Receives BEV features + intent token                              │  │
│  │  • Outputs waypoint trajectory [x₁, y₁, ... x₆, y₆]                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Internals: How Each Module Is Built

### 3.1 Sensor Encoders

#### Multi-Modal Sensor Suite

An autonomous vehicle carries multiple **types** (modalities) of sensors — each capturing a different physical phenomenon. Together they form the **multi-modal sensor suite**. No single sensor covers all conditions, so the AI must fuse them all.

| Sensor | Modality | What it measures | Strength | Weakness |
|--------|----------|-----------------|----------|---------|
| **Camera** | Visible light | Colour, texture, lanes, signs | Rich semantic detail, low cost | Fails in rain / fog / darkness |
| **LiDAR** | Laser pulses (ToF) | Precise 3D point cloud, distances | Accurate depth, works in dark | Expensive; sparse returns in heavy rain |
| **Radar** | Radio waves | Velocity + distance of objects | All-weather, measures speed directly | Low resolution, no colour/texture |
| **GPS / GNSS** | Satellite signals | Global absolute position | Absolute localisation | Fails in tunnels; ~1–3 m accuracy |
| **IMU** | Accelerometers + gyroscopes | Ego acceleration and rotation rate | High-frequency ego-motion | Drifts over time without correction |
| **HD Map** | Pre-built prior | Road structure, lane topology | Free geometry and semantic prior | Must be maintained and kept current |

**Why fusion matters:** Cameras can't see in fog; LiDAR struggles in heavy rain; radar can't read road signs. Fusing all modalities gives the AI a complete, redundant picture of the environment. This is why models like BEVFusion explicitly combine camera and LiDAR features into a unified Bird's-Eye View (BEV) before any perception or planning.

```
Camera (6×) ──→ ViT / ResNet encoder ──→ 2D feature maps ──→ BEV lift (LSS)
                                                                     │
LiDAR ────────→ VoxelNet sparse conv ──→ BEV pillar features ────────┤
                                                                     ▼
Radar (opt.) ──→ point / grid encoder ──────────────────────→ Fused BEV [H×W×C]
                                                                     │
GPS + IMU ─────→ ego pose / velocity ──────────────────────→ positional conditioning
```

---

#### Camera Encoder (Vision Transformer Path)

```
Raw image [3 × H × W]
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  Patch Embedding                                            │
│  16×16 patches → linear projection → [N_p × D_model]       │
│  + 2D sinusoidal positional embeddings                      │
└────────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────┐
│  ViT Transformer Blocks × L                                 │
│  Each block:                                                │
│    LayerNorm → Multi-head Self-Attention → residual         │
│    LayerNorm → MLP (FFN, 4× expand) → residual             │
│  Output: patch features [N_p × D_model]                     │
└────────────────────────────────┬───────────────────────────┘
                                 │
                          FPN neck (multi-scale)
                                 │
                                 ▼
                    Multi-scale feature maps
                    [P2: H/4, P3: H/8, P4: H/16, P5: H/32]
```

#### Camera → BEV Lift via Lift-Splat-Shoot (LSS)

```
Per-pixel depth distribution D(d | pixel, camera)
        ×
Per-pixel semantic feature F(pixel)
        ↓
Per-voxel feature accumulation (splat) into 3D frustum
        ↓
BEV pool along Z axis (shoot) → [H_bev × W_bev × C]
```

This is camera-only BEV and is used in BEVDet, BEVDepth, BEVFusion camera branch.

#### LiDAR Encoder (VoxelNet/PointPillars)

```
Raw point cloud [N_points × 4 (x,y,z,intensity)]
        │
        ▼
Voxelise: assign points to (x,y,z) voxels
        │
        ▼
Voxel Feature Encoding (VFE):
  Per-voxel: PointNet-style MLP on ~35 points
  → fixed-size voxel descriptor [C_v]
        │
        ▼
Sparse 3D Convolutional backbone (SubM SparseConv)
  → produces BEV feature map or 3D feature volume
        │
        ▼
BEV feature map [H_bev × W_bev × C_lid]
```

---

### 3.2 BEV Temporal Encoder (BEVFormer-style)

Multi-camera temporal fusion using deformable attention:

```
Current BEV query Q_t   [H×W×C]
Past BEV features B_{t-1..t-T}
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│  Temporal Self-Attention                                      │
│  • Warp B_{t-1} to current ego frame (rotation + translation)│
│  • Deformable Attn: Q_t attends to K reference points        │
│    sampled from warped past BEV features                      │
│  • Learns "what to remember from the past"                    │
└──────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│  Spatial Cross-Attention                                      │
│  • BEV query attends to 2D perspective image features         │
│  • Reference points: project BEV grid into each camera view  │
│  • Multi-scale deformable attention over FPN features         │
└──────────────────────────────────────────────────────────────┘
                │
                ▼
         Updated BEV features [H×W×C]
```

---

### 3.3 Perception Heads (DETR-style)

All modern AD perception heads use query-based detection (no NMS):

```
Learned object queries Q_obj [N_q × D]
+ BEV features K, V [H×W×D]
        │
        ▼
Transformer decoder:
  Q_obj → Self-Attn (queries interact) → Cross-Attn with BEV → FFN
  Repeat × 6 decoder layers
        │
        ▼
Per-query prediction heads:
  • Class: Linear(D) → softmax  [N_q × num_classes]
  • 3D Box: Linear(D) → (x,y,z,l,w,h,θ) [N_q × 7]
  • Velocity: Linear(D) → (vx, vy) [N_q × 2]
  • Track ID: cosine similarity against memory bank
```

---

### 3.4 Motion Prediction Head (MTR-style)

```
Per-agent features from detection head
+ Map polyline features (PointNet encoded road graph)
+ Scene context (global attention)
        │
        ▼
Motion Transformer (MTR):
  Intention point priors (K = 64 trajectory endpoints)
  → Cross-attention: trajectory query ↔ scene features
  → Iterative refinement (motion queries updated per layer)
        │
        ▼
Output per agent:
  K trajectory hypotheses, each with:
    • T=8 (4s) waypoints [K × T × 2]
    • Confidence score [K]
```

---

### 3.5 Planning Head (Waypoint Generation)

**Imitation Learning (IL) Planner:**

```
BEV features + Detected agents (from perception) + HD Map + Ego state
        │
        ▼
Planning Transformer:
  Ego query Q_ego attends to:
    • Object queries (agents)
    • Map queries (road topology)
    • Goal point embedding (target waypoint or command)
        │
        ▼
GRU decoder: autoregressively generates waypoints
  hidden_0 = f(ego_state)
  for t in [1..T]:
      waypoint_t = MLP(hidden_t)
      hidden_{t+1} = GRU(waypoint_t, hidden_t)
        │
        ▼
  Planned trajectory [T × 2] (x, y in BEV)
```

**Occupancy-based Implicit Planner (UniAD-style):**
Instead of direct waypoint regression, score multiple candidate trajectories against a learned occupancy cost map.

---

### 3.6 VLM Integration (Foundation Model Path)

```
Sensor images → Visual encoder (ViT / SigLIP / CLIP)
                      │
                      ▼ visual tokens [N_vis × D]
Text tokens (instruction / scene description)
                      │
                      ▼ text tokens [N_text × D]
                      │
              Token concatenation
                      │
                      ▼
┌────────────────────────────────────────────────────────────────┐
│  Causal LLM Decoder (QwenLM / LLaMA / Mistral backbone)        │
│  Full self-attention over [visual + text] tokens               │
│  Predicts next text token autoregressively                     │
│                                                                │
│  For VLA: additional action token vocabulary appended          │
│  Action tokens: discretised (x, y, θ, speed) via VQ codebook  │
└────────────────────────────────────────────────────────────────┘
                      │
    ┌─────────────────┼──────────────────┐
    ▼                 ▼                  ▼
Scene description   Chain-of-thought   Action tokens
(text output)       reasoning          → detokenise → waypoints
```

---

## 4. Architecture Combinations: Component Assembly

### Combination A: Camera-Only BEV E2E (BEVFormer + UniAD)

```
6× Cameras
    │ ResNet / ViT encoder
    ▼
Multi-scale image features [6 × P2..P5]
    │ Spatial cross-attention (BEVFormer)
    ▼
BEV feature map [200×200×256]
    │ Temporal self-attention (history fusion)
    │
    ├──→ [Track Head] Object queries → tracked agents
    │           │
    │           ▼ agent features fed as context
    ├──→ [Map Head] Map queries → BEV semantic lanes
    │           │
    │           ▼ map features fed as context
    ├──→ [Motion Head] Per-agent future trajectories
    │           │
    │           ▼ occupancy flow from trajectories
    ├──→ [Occupancy Head] 2D BEV occupancy map (future T frames)
    │           │
    │           ▼ cost map for planning
    └──→ [Planning Head] Ego waypoints (IL + occupancy cost)
```

**Shared information flow** (UniAD's key contribution): each downstream head queries the output of upstream heads via cross-attention. Tracking informs motion; motion informs occupancy; occupancy informs planning.

---

### Combination B: LiDAR-Camera Fusion (BEVFusion + CenterPoint + MTR)

```
LiDAR point cloud ──→ VoxelNet ──→ BEV [200×200×C_lid]
                                         │
                                    Fusion (concat + conv)
                                         │
6× Cameras ──→ ResNet/FPN ──→ LSS ──→ BEV [200×200×C_cam]
                                         │
                                 Fused BEV [200×200×(C_lid+C_cam)]
                                         │
                         ┌───────────────┼──────────────────┐
                         ▼               ▼                  ▼
                  3D Detection     Map Seg.          Motion Prediction
                  (CenterPoint)    (Seg head)        (MTR Transformer)
                                                          │
                                                    Trajectories → Planning
```

---

### Combination C: VLM-Augmented Dual Pipeline (DriveVLM-style)

```
                 Human instruction / scene question
                           │
                           ▼
Cameras ──→ CLIP/SigLIP encoder ──→ visual tokens
                           │
                           ▼
                  VLM Backbone (Qwen2-VL 7B)
                           │
              ┌────────────┴────────────────┐
              ▼                             ▼
  Chain-of-thought reasoning         Scene description
  "Construction zone, reduce speed"  "Pedestrian crossing left"
              │
              ▼
    Intent token / command embedding
              │
              ▼
 Classical BEV Planner ────────→ Waypoint trajectory
 (fast, geometric, deterministic)
 (receives intent as soft-constraint)
```

**Why dual?** The VLM handles rare/complex semantics. The geometric planner handles low-latency reactive control. The two run asynchronously — VLM updates at 2–5 Hz, planner runs at 20 Hz.

---

### Combination D: Cosmos World Model + RL Policy (NVIDIA Reference Pipeline)

```
TRAINING:
  Historical driving video (curated fleet data)
           │
           ▼
  Cosmos pre-training (video prediction)
           │
           ▼
  Cosmos fine-tune on AD domain (nuPlan, Waymo, CARLA)
           │
           ▼  environment simulator
  RL agent (PPO): action → Cosmos step → next frame → reward
           │
           ▼
  Policy network: Transformer  [ego_state, BEV_tokens] → waypoints

INFERENCE (on-vehicle):
  Real sensors → Student policy (distilled from RL teacher)
  Cosmos: offline  │  Student policy: online (real-time)
```

---

### Combination E: Occupancy + Diffusion Planner (OccWorld + DriveDreamer)

```
Sensor History (T frames)
        │
        ▼
BEV Feature Extractor (BEVFormer)
        │
        ▼
Occupancy Encoder: voxel grid [X×Y×Z×C] → latent z_occ
        │
        ▼
World Model (Transformer or Diffusion):
  z_occ_t + action_t → z_occ_{t+1}
  (jointly models geometry + agent motion)
        │
        ▼
Occupancy Decoder: z_occ → [X×Y×Z×num_classes]
  Predicts: free space, vehicles, pedestrians, road surface
        │
        ▼
Trajectory optimiser:
  Sample N candidate trajectories
  Score each against predicted occupancy (collision cost)
  Select lowest-cost trajectory
```

---

## 5. Joint End-to-End Training

### 5.1 Multi-Task Loss

All Paradigm 2/3 models train with a weighted sum across task heads:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{det} + \lambda_2 \mathcal{L}_{track} + \lambda_3 \mathcal{L}_{map} + \lambda_4 \mathcal{L}_{motion} + \lambda_5 \mathcal{L}_{occ} + \lambda_6 \mathcal{L}_{plan}$$

| Loss | Type | Target |
|------|------|--------|
| $\mathcal{L}_{det}$ | Focal + L1 (Hungarian match) | 3D bounding box |
| $\mathcal{L}_{track}$ | Contrastive / cosine embed | Re-ID consistency |
| $\mathcal{L}_{map}$ | Binary cross-entropy | BEV semantic lanes |
| $\mathcal{L}_{motion}$ | Winner-takes-all min-FDE / NLL | Future agent trajectories |
| $\mathcal{L}_{occ}$ | Focal loss over voxel grid | Occupancy + velocity flow |
| $\mathcal{L}_{plan}$ | L2 vs. expert trajectory + collision penalty | Ego planned waypoints |

**Gradient flow:** gradients from downstream planning loss backpropagate all the way through the shared backbone. This is the key advantage of E2E training — the perception backbone learns features useful *for planning*, not just detection.

---

### 5.2 Imitation Learning (Behavioural Cloning)

Most AD training starts with **behavioural cloning (BC)** on logged expert demonstrations:

```
Dataset: (sensor_observations_t, expert_waypoints_t)  ← human driving logs
                         │
                         ▼
Objective: min L2(model_waypoints, expert_waypoints) over dataset
                         │
         ┌───────────────┴────────────────────┐
         ▼                                    ▼
  Distributional shift problem           Compounding errors
  (model sees states expert never saw)   (small errors grow over time)
         │
         ▼ Mitigations:
  • DAgger: online data collection with interventions
  • Closed-loop rollout training (use model's own state distribution)
  • Model predictive IL: multi-step prediction loss (T=6 future steps)
```

---

### 5.3 Reinforcement Learning Fine-Tuning

Applied after BC pre-training to overcome compounding errors:

```
BC pre-trained model (good initialisation)
        │
        ▼
RL environment: CARLA simulator or Cosmos world model
        │
        ▼
PPO / GRPO training loop:
  ┌─────────────────────────────────────────────┐
  │  1. Rollout: model drives in sim (N steps)  │
  │  2. Collect (state, action, reward) tuples  │
  │  3. Compute advantages (GAE)                │
  │  4. Update policy: PPO clip objective       │
  │  5. Update value fn: MSE vs. returns        │
  └─────────────────────────────────────────────┘

Reward function r(t):
  r = w₁·progress - w₂·collision - w₃·discomfort - w₄·traffic_violation
      + w₅·smoothness + w₆·lane_keeping
```

**GRPO for VLA (LLM-style):** Group Relative Policy Optimisation — treats action tokens like reasoning tokens; rewards depend on whether the generated trajectory is safe and efficient vs. the group mean.

---

### 5.4 Query Propagation (UniAD / VAD joint training detail)

The key to training multi-head E2E models is **structured query propagation** — each head uses the outputs of previous heads as context, and this chain is differentiable end-to-end:

```
BEV features                            (gradient flows ←──────────────)
   ↓                                                                    │
Track Head: object queries Q_det ──→ track embeddings                  │
                                             │ (as context)             │
                                             ▼                          │
Motion Head: agent query Q_agent ←── Q_det + past traj              ←──┘
                                             │
                                             ▼
Occupancy Head: Q_occ ←── motion output (rollout occupancy flow)
                                             │
                                             ▼
Planning Head: Q_plan ←── occupancy cost map + map features
                                             │
                                             ▼
          Planned waypoints → L_plan (vs. expert) → backprop all the way up
```

During backpropagation, the planning gradient reaches the detection head, training it to detect objects that *matter for planning*, not just maximise detection mAP.

---

### 5.5 VLM Joint Training (Visual Instruction Tuning)

Foundation model-based pipelines use a **3-stage training recipe**:

```
STAGE 1: Vision-Language Alignment (frozen LLM)
  Task: image captioning, VQA on general data
  Train: projection layer (MLP connector) only
  Data: LLaVA-558K / CC3M captions
  Duration: ~1 epoch

STAGE 2: Visual Instruction Fine-Tuning (unfreeze LLM)
  Task: multi-turn instruction following, scene Q&A
  Train: full model (ViT + connector + LLM)
  Data: LLaVA-665K / ShareGPT4V
  Duration: 1–2 epochs

STAGE 3: Domain Fine-Tuning (AD-specific)
  Task: driving scene description, hazard Q&A, CoT reasoning
  Train: LoRA adapters or full fine-tune
  Data: DriveLM-nuScenes, nuPlan commentary logs, CARLA synthetic QA
  Duration: 3–5 epochs

  Optional VLA extension:
    Add action token vocabulary to tokenizer
    Add action prediction head (MLP on last token)
    Train on (observation, expert_action) pairs with BC loss
```

---

### 5.6 World Model Joint Pre-training (Cosmos Recipe)

```
Stage 1: Tokenizer pre-training
  Train Cosmos Tokenizer (3D causal CNN + VQ-VAE) on video reconstruction
  Objective: LPIPS + L1 + adversarial
  Data: 20M hrs internet video + driving logs

Stage 2: World model pre-training (Autoregressive or Diffusion)
  Objective: predict next latent token (AR) or denoise (diffusion)
  Data: same tokenized video corpus

Stage 3: Physical grounding fine-tune
  Data: AD-domain — nuPlan, Waymo, CARLA (physics-consistent)
  Added supervision: ego trajectory loss, depth consistency loss

Stage 4: Action-conditioned fine-tune
  Condition on ego action tokens (speed, steer, accel)
  Objective: accurately predict sensor observations given action
  Enables: planning-as-inference (MBRL)
```

---

## 6. Other Industry E2E Models

| Model | Organisation | Key Approach | Components |
|-------|-------------|-------------|-----------|
| **LINGO-2** | Wayve | VLM with driving commentary as reward signal | ViT + LLM + IL + RL |
| **GAIA-1** | Wayve | Generative world model for video prediction | GPT-style autoregressive video |
| **DriveX** | Waymo | Internal VLM-planner integration | Proprietary |
| **Senna** | Various | LLM-based scene reasoning to VPA planner | Llama + IL planner |
| **GenAD** | OpenDriveLab | Generalised AD world model | Diffusion + BEV |
| **DriveCoT** | Shanghai AI Lab | Chain-of-thought driving decisions | Qwen + CoT fine-tune |
| **Tesla FSD v12** | Tesla | Full neural net, no hand-coded rules | E2E Conv+Transformer, BC+RL |
| **Think2Drive** | PKU | VLM for long-horizon CARLA planning | GPT-4V + MPC |
| **DriveLLM** | SAIC | LLM knowledge injection for planning | ChatGLM + BEV planner |

---

## 7. Architecture Selection Guide

| Scenario | Recommended Paradigm | Reason |
|----------|---------------------|--------|
| Production ADAS (L2/L2+) | Hybrid E2E (Paradigm 2) | Proven, fast, debuggable |
| L4 robotaxi urban | Foundation-augmented (Paradigm 3) | Handles long-tail |
| Simulation / synthetic data | Generative world model (Paradigm 4) | Unlimited data |
| Edge / embedded SoC | Distilled student from Paradigm 2 | Latency + power |
| Research / rapid iteration | VLM-augmented dual pipeline | Leverage pre-trained VLMs |
| RL training environment | Cosmos world model | Differentiable, scalable |

---

*See also: [VLM Models](VLM_models.md) | [VLA Models](VLA_models.md) | [World Models](world_models.md) | [Foundation Models](foundation_models.md) | [Edge Deployment](edge_deployment.md) | [AD Overview](autonomous_driving_AI_models.md)*
