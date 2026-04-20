# Edge Deployment of AD AI Models: Teacher–Student Architecture

**Part of:** [Autonomous Driving AI Models](autonomous_driving_AI_models.md)

---

## Table of Contents

- [Overview](#overview)
- [Teacher–Student Architecture](#teacherstudent-architecture)
  - [Distillation Loss Components](#distillation-loss-components)
- [Edge Hardware Targets](#edge-hardware-targets)
  - [Automotive-Grade SoCs](#automotive-grade-socs)
- [Compression Techniques](#compression-techniques)
  - [1. Knowledge Distillation](#1-knowledge-distillation)
  - [2. Quantisation](#2-quantisation)
  - [3. Pruning](#3-pruning)
  - [4. Architecture Optimisation](#4-architecture-optimisation)
- [Teacher–Student Pairs (AD-Specific)](#teacherstudent-pairs-ad-specific)
- [Deployment Toolchains](#deployment-toolchains)
  - [NVIDIA Jetson / DRIVE](#nvidia-jetson--drive)
  - [Qualcomm (SA8775P / Snapdragon Ride)](#qualcomm-sa8775p--snapdragon-ride)
  - [Generic / Cross-Platform](#generic--cross-platform)
- [Multi-Tier Deployment Architecture](#multi-tier-deployment-architecture)
- [Distillation Workflow (Step-by-Step)](#distillation-workflow-step-by-step)
- [Open-Source Distillation Frameworks](#open-source-distillation-frameworks)
- [Performance Reference: Jetson AGX Orin 64G](#performance-reference-jetson-agx-orin-64g)

---

## Overview

Autonomous vehicles require real-time AI inference under strict constraints:

| Constraint | Typical Target |
|-----------|---------------|
| Latency | < 50 ms per inference cycle (20 Hz) |
| Power budget | 15–75 W (embedded SoC) |
| Memory | 8–32 GB LPDDR5 |
| Storage | 64–256 GB eMMC / NVMe |
| Reliability | 24/7 all-weather, safety-SIL 2/3 |

Large foundation models (7B–70B parameters) cannot run at these specs without compression. **Teacher–Student distillation** is the primary method for transferring the capability of a large teacher model into a compact, edge-runnable student model.

---

## Teacher–Student Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Teacher–Student Pipeline                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              TEACHER (cloud / training cluster)               │  │
│  │                                                               │  │
│  │  Large Foundation Model (7B–70B)                             │  │
│  │  Full precision FP32/BF16                                    │  │
│  │  Trained on massive diverse dataset                          │  │
│  │  Produces: soft labels, feature maps, intermediate logits    │  │
│  └───────────────────────────┬──────────────────────────────────┘  │
│                               │  Distillation signals              │
│             ┌─────────────────┼──────────────────────┐             │
│             │  Soft labels    │  Feature alignment    │             │
│             │  (KL-div loss)  │  (MSE on layers)      │             │
│             │                 │                        │             │
│             ▼                 ▼                        ▼             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              STUDENT (edge SoC / embedded GPU)               │   │
│  │                                                               │   │
│  │  Compact Model (0.5B–3B)                                     │   │
│  │  INT8 / INT4 quantised                                       │   │
│  │  Sparse / pruned architecture                                │   │
│  │  TensorRT / ONNX / CoreML optimised                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Distillation Loss Components

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{task} + \beta \cdot \mathcal{L}_{KD} + \gamma \cdot \mathcal{L}_{feat}$$

| Loss Term | Formula | Purpose |
|-----------|---------|---------|
| $\mathcal{L}_{task}$ | Cross-entropy / regression vs. ground truth | Task accuracy |
| $\mathcal{L}_{KD}$ | $\text{KL}(p_S \| p_T^{1/\tau})$ at temperature $\tau$ | Soft-label transfer |
| $\mathcal{L}_{feat}$ | $\| f_S(x) - f_T(x) \|_2^2$ | Intermediate feature alignment |

---

## Edge Hardware Targets

### Automotive-Grade SoCs

| Platform | AI Tops | Memory | Power | Deployment Target |
|----------|---------|--------|-------|------------------|
| **NVIDIA Jetson Orin NX 8G** | 20 TOPS | 8 GB | 10–20 W | Entry parking/ADAS |
| **NVIDIA Jetson Orin NX 16G** | 35 TOPS | 16 GB | 10–25 W | Level 2+ ADAS |
| **NVIDIA Jetson AGX Orin 32G** | 80–100 TOPS | 32 GB | 15–60 W | L3 AV compute node |
| **NVIDIA Jetson AGX Orin 64G** | 80–275 TOPS (all engines) | 64 GB | 15–60 W | L3/L4 AV central |
| **NVIDIA DRIVE Thor** | 2000 TOPS | 128 GB | 150 W | L4/L5 AV SoC (2025+) |
| **Qualcomm SA8650P** | 73 TOPS | 16 GB | 15 W | L2+/L3 ADAS |
| **Qualcomm SA8775P** | 123 TOPS | 32 GB | 30 W | L3/L4 full-stack |
| **Hailo-8** | 26 TOPS | Host memory | 2.5 W | Vision accelerator insert |
| **Hailo-8L** | 13 TOPS | Host memory | 1.5 W | Entry ADAS add-on |
| **Mobileye EyeQ6H** | ~30 TOPS | Integrated | ~8 W | Camera-based ADAS |
| **Texas Instruments TDA4VM** | 8 TOPS | 8 GB | 15 W | Automotive vision pipeline |

---

## Compression Techniques

### 1. Knowledge Distillation

```
Teacher (7B FP16) ──▶ Student (1B INT8)
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
  Response-based    Feature-based    Relation-based
  (Soft labels)     (Layer align)    (Attention map)
```

Key methods:
- **Response KD**: Match output logits with temperature scaling (Hinton et al.)
- **Feature KD**: Align intermediate representations (FitNets, PKD)
- **Attention KD**: Transfer attention matrices (TinyBERT, MiniLM)
- **Contrastive KD**: CRD — maximise student-teacher mutual information

---

### 2. Quantisation

| Level | Size Reduction | Accuracy Drop | Tools |
|-------|---------------|---------------|-------|
| **FP16** | 2× vs FP32 | ~0% | PyTorch autocast, TensorRT |
| **INT8 (PTQ)** | 4× vs FP32 | 0.5–2% | TensorRT, ONNX Runtime, AIMET |
| **INT8 (QAT)** | 4× | < 0.5% | PyTorch FX, TensorRT QAT |
| **INT4 (GPTQ)** | 8× | 1–3% | AutoGPTQ, llama.cpp |
| **INT4 (AWQ)** | 8× | 0.5–2% | AutoAWQ, TensorRT-LLM |
| **INT4 (GGUF Q4_K_M)** | 8× | 1.5–3% | llama.cpp, Ollama |
| **2-bit (QuIP#)** | 16× | 3–6% | Research stage |

---

### 3. Pruning

| Type | Description | Size Reduction | Accuracy Drop |
|------|-------------|---------------|---------------|
| **Unstructured (magnitude)** | Zero individual weights | 50–90% | 1–5% (sparse) |
| **Structured (channel)** | Remove entire filters | 30–70% | 2–5% |
| **Head pruning (Transformers)** | Remove attention heads | 20–50% | 1–3% |
| **Depth pruning** | Remove Transformer layers | 20–40% | 1–4% |
| **SparseGPT** | One-shot weight pruning | 50% | ~1% |
| **Wanda** | Weight magnitude × activation norm | 50% | ~1.5% |

---

### 4. Architecture Optimisation

| Technique | Benefit | Examples |
|-----------|---------|---------|
| **MobileNet-style depthwise conv** | 8–9× less compute | MobileViT, EfficientNet |
| **Mixture-of-Experts (MoE)** | Large model, sparse activation | Mixtral, Qwen2-57B-A14B |
| **Grouped Query Attention (GQA)** | Fewer KV cache params | LLaMA 3, Mistral |
| **FlashAttention 3** | Memory-efficient attention | All Transformer inference |
| **Speculative decoding** | 2–4× LLM throughput | Draft model + verifier |
| **Continuous batching** | GPU utilisation for LLM serving | vLLM, TensorRT-LLM |

---

## Teacher–Student Pairs (AD-Specific)

| Task | Teacher | Student | Compression | Edge Target |
|------|---------|---------|-------------|------------|
| Scene description | Qwen2-VL-72B | Qwen2-VL-2B INT4 | 36× | Jetson Orin NX 8G |
| Object detection | InternVL2-8B | InternVL2-1B INT8 | 8× | Jetson Orin NX 8G |
| Depth estimation | Depth-Anything-v2-L | Depth-Anything-v2-S INT8 | 14× | Hailo-8 + host |
| VLA action | OpenVLA-7B | TinyVLA-1.1B INT8 | 7× | Jetson AGX Orin |
| Planning reasoning | DeepSeek-R1-14B | DeepSeek-R1-Distill-7B INT4 | 2× | Jetson AGX Orin 64G |
| Segmentation | SAM2-large | EfficientSAM INT8 | 22× | Jetson Orin NX 16G |
| Occupancy | OccWorld-full | OccWorld-tiny INT8 | 6× | Jetson AGX Orin |
| BEV detection | BEVFusion-L (LiDAR+cam) | CenterPoint-cam INT8 | 10× | Jetson AGX Orin |

---

## Deployment Toolchains

### NVIDIA Jetson / DRIVE

```
PyTorch FP32/BF16 model
        │
        ▼
ONNX Export (torch.onnx.export)
        │
        ▼
TensorRT engine (trtexec or Python API)
        │
     ┌──┴────────────────────┐
     │  Calibration (INT8)   │
     │  or QAT layers (QAT)  │
     └──────────────────────┘
        │
        ▼
.trt engine file → deploy on Jetson / DRIVE
```

**Key NVIDIA tools:**

| Tool | Purpose |
|------|---------|
| TensorRT 10 | Engine optimisation + INT8/INT4 |
| TensorRT-LLM | Optimised LLM serving (streaming) |
| DeepStream | Multi-stream video pipeline |
| TAO Toolkit | Transfer learning + pruning |
| AIMET | Quantisation + compression framework |
| cuDLA | DLA compiler for Jetson |

---

### Qualcomm (SA8775P / Snapdragon Ride)

| Tool | Purpose |
|------|---------|
| Qualcomm AI Hub | Cloud-based model optimisation |
| QNN SDK | Quantised neural network inference |
| SNPE (deprecated → QNN) | Hexagon DSP acceleration |
| AI Model Efficiency Toolkit (AIMET) | PTQ / QAT for Qualcomm targets |

---

### Generic / Cross-Platform

| Tool | Framework | Use Case |
|------|-----------|---------|
| **llama.cpp** | C++ | GGUF quantised LLM/VLM on CPU or GPU |
| **Ollama** | llama.cpp wrapper | Local model serving |
| **ONNX Runtime** | C++/Python | Cross-platform INT8 inference |
| **OpenVINO** | Intel | Intel-CPU optimised inference |
| **CoreML** | Swift/Python | Apple Silicon deployment |
| **ExecuTorch** | PyTorch | Mobile / embedded PyTorch deployment |
| **MLC LLM** | TVM | Universal LLM deployment (any HW) |
| **vLLM** | Python | PagedAttention serving for cloud inference |

---

## Multi-Tier Deployment Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      VEHICLE COMPUTE                         │
│                                                              │
│  ┌──────────────────────────────────────┐                   │
│  │  AV CENTRAL COMPUTE NODE             │                   │
│  │  (DRIVE Thor / AGX Orin 64G)         │                   │
│  │                                      │                   │
│  │  Student Models (INT8/INT4):         │                   │
│  │  • BEV Perception (50 ms)            │                   │
│  │  • Occupancy World Model (30 ms)     │                   │
│  │  • VLA Action Head (20 ms)           │                   │
│  │  • Safety Monitor (always on)        │                   │
│  └───────────────────┬──────────────────┘                   │
│                       │ CAN / Ethernet                       │
│  ┌────────────────────┴─────────────────┐                   │
│  │  SENSOR FUSION NODES                 │                   │
│  │  (Hailo-8 / TDA4VM)                  │                   │
│  │  • Camera ISP + detection (5 ms)     │                   │
│  │  • LiDAR clustering (8 ms)           │                   │
│  └──────────────────────────────────────┘                   │
└───────────────────────────────────────────5G/WiFi───────────┘
                                                │
┌──────────────────────────────────────────────▼──────────────┐
│                      CLOUD BACKEND                           │
│                                                              │
│  Teacher Models (FP16/BF16):                                 │
│  • Qwen2-VL-72B  → generates new training labels            │
│  • DeepSeek-R1   → complex scenario reasoning               │
│  • World Model   → rare edge-case scenario generation        │
│                                                              │
│  OTA Model Updates → compressed student model weights        │
└──────────────────────────────────────────────────────────────┘
```

---

## Distillation Workflow (Step-by-Step)

```
Step 1: SELECT teacher + student architecture pair
         Teacher: Qwen2-VL-7B  │ Student: Qwen2-VL-2B

Step 2: COLLECT distillation dataset
         - nuScenes, Waymo Open, CARLA synthetic
         - Teacher inference → store soft labels + features

Step 3: DISTILLATION TRAINING
         Loss = λ₁·CE(y_pred, y_gt)
               + λ₂·KL(student_logits ∥ teacher_logits / τ)
               + λ₃·MSE(student_features, teacher_features)
         Epochs: 50–100 on combined dataset

Step 4: POST-TRAINING QUANTISATION (PTQ)
         Calibration: 512–1024 representative samples
         Target: INT8 (W8A8) or INT4 (W4A16)

Step 5: VALIDATION on held-out test split
         Regression test: nuScenes mAP delta < 3%
         Latency test: < 50 ms on target SoC

Step 6: ONNX/TensorRT EXPORT
         Batch = 1 (streaming)  │  FP16 backbone + INT8 heads

Step 7: SAFETY VALIDATION
         ISO 26262 / SOTIF testing
         ODD (Operational Design Domain) coverage

Step 8: OTA DEPLOY to vehicle fleet
```

---

## Open-Source Distillation Frameworks

| Framework | Focus | Models Supported | Source |
|-----------|-------|-----------------|--------|
| **TinyBERT** | BERT-family distillation | Transformer NLP | [HuggingFace TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) |
| **pkd-bert** | Patient KD for BERT | Any Transformer | [github.com/intersun/PKD-for-BERT-Model-Compression](https://github.com/intersun/PKD-for-BERT-Model-Compression) |
| **LLaMA-Factory** | Fine-tune + distillation for LLMs | LLaMA, Mistral, Qwen | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| **Axolotl** | LLM fine-tune / distillation | All PEFT-compatible | [github.com/OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) |
| **TorchDistill** | General KD framework in PyTorch | Any PyTorch model | [github.com/yoshitomo-matsubara/torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) |
| **MiniLLM** | Sequence-level LLM KD | GPT-style LLMs | [github.com/microsoft/LMOps/tree/main/minillm](https://github.com/microsoft/LMOps/tree/main/minillm) |
| **Distill-CLIP** | CLIP-family vision KD | ViT + CLIP | Research |
| **AutoGPTQ** | INT4 GPTQ quantisation | All LLM families | [github.com/AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) |
| **AutoAWQ** | INT4 AWQ quantisation | LLaMA, Mistral, Qwen | [github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) |
| **NVIDIA TAO** | Transfer learning + pruning | Detection, segmentation | [developer.nvidia.com/tao-toolkit](https://developer.nvidia.com/tao-toolkit) |
| **AIMET** | Quantisation aware training | PyTorch / TF models | [github.com/quic/aimet](https://github.com/quic/aimet) |

---

## Performance Reference: Jetson AGX Orin 64G

| Model | Task | Precision | Latency | Throughput |
|-------|------|-----------|---------|-----------|
| InternVL2-1B | VQA | INT8 | 15 ms | 67 fps |
| InternVL2-4B | VQA | INT8 | 40 ms | 25 fps |
| Qwen2-VL-2B | Scene desc | INT4 | 35 ms | 28 fps |
| Moondream 2 | Caption | INT4 | 8 ms | 125 fps |
| TinyVLA-1.1B | VLA action | INT8 | 22 ms | 45 fps |
| SAM2-base+ | Segmentation | INT8 | 12 ms | 83 fps |
| Depth-Anything-v2-S | Depth | INT8 | 5 ms | 200 fps |
| DeepSeek-R1-Distill-7B | Reasoning | INT4 | 180 ms | 5.5 fps |
| OccWorld-tiny | 4D occupancy | INT8 | 25 ms | 40 fps |

---

*See also: [VLM Models](VLM_models.md) | [VLA Models](VLA_models.md) | [World Models](world_models.md) | [Foundation Models](foundation_models.md)*
