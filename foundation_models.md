# Foundation Models for Autonomous Driving

**Part of:** [Autonomous Driving AI Models](autonomous_driving_AI_models.md)

---

## Table of Contents

- [Overview](#overview)
- [Language Foundation Models (LLMs) Used in AD](#language-foundation-models-llms-used-in-ad)
- [Vision Foundation Models (Image/Video)](#vision-foundation-models-imagevideo)
- [Multi-Modal Foundation Models (AD-Focused)](#multi-modal-foundation-models-ad-focused)
- [Foundation Model Taxonomy for AD](#foundation-model-taxonomy-for-ad)
- [Pre-Training Strategies](#pre-training-strategies)
- [Key Open Benchmarks](#key-open-benchmarks)
- [Choosing a Foundation Model for Your AD Stack](#choosing-a-foundation-model-for-your-ad-stack)

---

## Overview

Foundation models for autonomous driving are large-scale, pre-trained networks that encode generalised representations of the physical world from massive, heterogeneous data. Unlike task-specific networks, they are:

- **Pre-trained** on internet-scale or driving-scale data (images, video, language, sensor data)
- **Transferable** to many downstream AD tasks via fine-tuning or prompting
- **Unified** across perception, prediction, and planning — replacing brittle pipelines of specialist models

The AD foundation model landscape spans three lineages:

| Lineage | Core Training | AD Use |
|---------|--------------|--------|
| **Language FM** | Web text | Reasoning, planning, QA |
| **Vision FM** | Image/video | Perception, scene encoding |
| **Multi-modal FM** | Image + text (+ audio, sensor) | Full-stack understanding |

---

## Language Foundation Models (LLMs) Used in AD

| Model | Params | Context | Licence | Key AD Application |
|-------|--------|---------|---------|-------------------|
| **LLaMA 3.1 8B** | 8 B | 128 K tokens | Meta Llama 3 Community | Planning reasoning, QA |
| **LLaMA 3.1 70B** | 70 B | 128 K tokens | Meta Llama 3 Community | Fleet data analysis |
| **LLaMA 3.3 70B** | 70 B | 128 K tokens | Meta Llama 3 Community | Code + chain-of-thought |
| **Mistral-7B-v0.3** | 7 B | 32 K tokens | Apache-2.0 | Lightweight planning backbone |
| **Mistral-NeMo 12B** | 12 B | 128 K tokens | Apache-2.0 | NVIDIA edge deployment |
| **Mixtal 8×7B** | 47 B (MoE) | 32 K tokens | Apache-2.0 | Expert-routing for diverse tasks |
| **Qwen2.5 7B** | 7 B | 128 K tokens | Apache-2.0 | Structured JSON reasoning |
| **Qwen2.5 72B** | 72 B | 128 K tokens | Apache-2.0 | Expert reasoning; map analysis |
| **Phi-4 14B** | 14 B | 16 K tokens | MIT | Math/physics reasoning; safety |
| **Gemma 2 9B** | 9 B | 8 K tokens | Gemma Licence | Edge-optimised by Google |
| **Gemma 2 27B** | 27 B | 8 K tokens | Gemma Licence | On-car server |
| **DeepSeek-V3 (671B MoE)** | 671 B | 128 K tokens | MIT | Best open reasoning (2025) |
| **DeepSeek-R1 (671B)** | 671 B | 128 K tokens | MIT | Chain-of-thought planning |
| **DeepSeek-R1-Distill-7B** | 7 B | 64 K tokens | MIT | Distilled CoT planner on edge |
| **DeepSeek-R1-Distill-14B** | 14 B | 64 K tokens | MIT | Balanced CoT on AGX Orin |

---

## Vision Foundation Models (Image/Video)

| Model | Params | Pretraining | Licence | AD Application |
|-------|--------|-------------|---------|---------------|
| **DINOv2 ViT-L/14** | 307 M | Self-supervised (LVD-142M) | Apache-2.0 | Dense features; 3D-aware detection |
| **DINOv2 ViT-G/14** | 1.1 B | Self-supervised | Apache-2.0 | Strongest dense features |
| **SAM 2 (base+)** | 80 M | SA-1B + SA-V (video) | Apache-2.0 | Real-time video segmentation |
| **SAM 2 (large)** | 224 M | SA-1B + SA-V | Apache-2.0 | High-quality mask propagation |
| **EfficientSAM** | 10 M | SAM distillation | Apache-2.0 | Edge-deployable segmentation |
| **Grounding DINO** | 341 M | COCO + grounded pretraining | Apache-2.0 | Open-vocab object detection |
| **OWLv2** | 350 M | Web image-text | Apache-2.0 | Open-vocab detection |
| **EVA-CLIP-18B** | 18 B | 2 B image-text pairs | MIT | Best image encoder |
| **SigLIP-SO400M** | 400 M | WebLI 10 B pairs | Apache-2.0 | Encoders for VLMs/VLAs |
| **V-JEPA 2** | 1.2 B | Video self-supervised | CC BY-NC 4.0 | Fastest video representation |
| **VideoMAE-H** | 633 M | Masked video autoencoding | CC BY-NC 4.0 | Temporal scene understanding |
| **Depth-Anything v2** | 25–335 M | Synthetic + real depth | Apache-2.0 | Monocular depth estimation |
| **UniDepth** | 344 M | Metric depth generalisation | Apache-2.0 | Camera-agnostic metric depth |
| **Marigold** | ~1 B | Diffusion-based depth | Apache-2.0 | High-quality single-image depth |

---

## Multi-Modal Foundation Models (AD-Focused)

| Model | Params | Inputs | AD Tasks | Licence | Source |
|-------|--------|--------|---------|---------|--------|
| **UniAD** | ~130 M | Cam + LiDAR | Perception → prediction → planning | Apache-2.0 | [github.com/OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD) |
| **BEV-Planner** | ~80 M | Surround cams | BEV perception + motion planning | Apache-2.0 | [github.com/HorizonRobotics/BEV-Planner](https://github.com/HorizonRobotics/BEV-Planner) |
| **DriveX** | 7 B | Multi-cam + map | Unified: detect, predict, plan | Research | preprint |
| **OpenDriveVLA** | 7 B | Camera + route | Full multi-modal driving agent | Apache-2.0 | [github.com/OpenDriveLab](https://github.com/OpenDriveLab) |
| **Dolphins (FM)** | 7 B | Dashcam video | Conversational driving FM | Apache-2.0 | [github.com/SaFoLab-WISC/Dolphins](https://github.com/SaFoLab-WISC/Dolphins) |
| **ELM** | 7 B | Surround + spatial data | Embodied spatial reasoning | Apache-2.0 | [github.com/OpenDriveLab/ELM](https://github.com/OpenDriveLab/ELM) |
| **NAVSIM baselines** | Various | Camera + HD map route | Log-replay closed-loop benchmark | Apache-2.0 | [github.com/autonomousvision/navsim](https://github.com/autonomousvision/navsim) |
| **Panda-70M** | Pre-training corpus | Diverse video | Broad temporal pretraining | Apache-2.0 | Dataset only |

---

## Foundation Model Taxonomy for AD

```
                    Foundation Models
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   Language FM        Vision FM        Multi-modal FM
        │                 │                 │
  ┌─────────┐       ┌─────────┐       ┌──────────────┐
  │ Plan,   │       │ Detect, │       │ Perceive +   │
  │ Reason, │       │ Segment │       │ Plan in      │
  │ QA      │       │ Depth   │       │ one pass     │
  └────┬────┘       └────┬────┘       └──────┬───────┘
       │                 │                   │
       └─────────────────┴───────────────────┘
                         │
               Fine-tune / Prompt
                         │
          ┌──────────────┼─────────────────┐
          │              │                 │
       VLM task       VLA task       World Model
      (Sec/VLM)      (Sec/VLA)        (Sec/WM)
```

---

## Pre-Training Strategies

| Strategy | Description | Representative Models |
|----------|-------------|----------------------|
| **Supervised (image-text)** | Contrastive pairing of image + caption | CLIP, SigLIP, EVA-CLIP |
| **Masked autoencoding** | Predict masked patches / frames | MAE, VideoMAE |
| **Self-supervised JEPA** | Predict latent representations, no pixels | I-JEPA, V-JEPA |
| **Causal LM** | Next-token prediction on text/tokens | LLaMA, Qwen, Mistral |
| **Multi-modal causal LM** | Interleaved image+text next-token | LLaVA, Qwen2-VL |
| **Diffusion pre-training** | Denoise latent representations | DriveDreamer, Marigold |
| **RL from human feedback** | RLHF/DPO alignment + safety | All chat-aligned models |

---

## Key Open Benchmarks

| Benchmark | Task | Metric | Leader (open, 2025) |
|-----------|------|--------|---------------------|
| **nuScenes detection** | 3D object detection | mAP / NDS | InternImage-XL + BEVFusion |
| **nuScenes prediction** | Agent trajectory | minADE5 | MTR, Wayformer |
| **nuPlan planning** | Closed-loop driving score | PDMS | PDM-Closed, NAVSIM |
| **NAVSIM** | Non-reactive planning | PDMS | SparseDrive, UniAD |
| **KITTI depth** | Monocular depth | AbsRel | Depth-Anything v2 |
| **nuScenes-QA** | VQA on driving data | Accuracy | InternVL2-76B |
| **Waymo Motion** | Trajectory prediction | mAP | MTR++ |

---

## Choosing a Foundation Model for Your AD Stack

```
Task                    Recommended Model (open-source)
─────────────────────────────────────────────────────────────────
Scene description       Qwen2-VL-7B or InternVL2-8B
Open-vocab detection    Grounding DINO + SAM2
Monocular depth         Depth-Anything-v2-Large
Dense features (3D)     DINOv2 ViT-L
Video understanding     V-JEPA 2 or LLaVA-Video-7B
Planning reasoning      DeepSeek-R1-Distill-14B (edge)
                        DeepSeek-R1-671B (cloud)
End-to-end AD           UniAD, SparseDrive
Embodied VLA            OpenVLA (cloud), TinyVLA (edge)
Edge camera encoder     MobileViT, EfficientViT, InternViT-300M
```

---

*See also: [VLM Models](VLM_models.md) | [VLA Models](VLA_models.md) | [World Models](world_models.md) | [Edge Deployment](edge_deployment.md)*
