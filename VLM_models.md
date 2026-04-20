# Vision-Language Models (VLM) for Autonomous Driving

**Part of:** [Autonomous Driving AI Models](autonomous_driving_AI_models.md)

---

## Table of Contents

- [Overview](#overview)
- [Open-Source VLM Models](#open-source-vlm-models)
  - [General-Purpose VLMs (usable for AD)](#general-purpose-vlms-usable-for-ad)
  - [Autonomous Driving–Specific VLMs](#autonomous-driving-specific-vlms)
- [Architecture Overview](#architecture-overview)
  - [Common Vision Encoders](#common-vision-encoders)
- [AD-Relevant Tasks & Benchmark Performance](#ad-relevant-tasks--benchmark-performance)
- [Python Integration](#python-integration)
- [Deployment Sizes (Quantised)](#deployment-sizes-quantised)

---

## Overview

Vision-Language Models (VLMs) fuse image (or video) encoders with large language model decoders, enabling open-vocabulary scene understanding, driver-query answering, and rich semantic annotation — core perception capabilities for autonomous driving.

In an AD pipeline VLMs are used for:
- Scene description and semantic segmentation narration
- Traffic sign / road-marking recognition
- Hazard and edge-case detection
- Driver-assistance Q&A (e.g. "is it safe to merge?")
- Data labelling and synthetic annotation

---

## Open-Source VLM Models

### General-Purpose VLMs (usable for AD)

| Model | Params | Context / Resolution | Key Capabilities | Licence | Ollama Tag |
|-------|--------|---------------------|-----------------|---------|------------|
| **Moondream 2** | 1.86 B | 378 × 378 | Compact edge VLM; captioning, VQA, detection | Apache-2.0 | `moondream` |
| **Phi-3.5-Vision** | 4.2 B | 4× tiles (HD crops) | Multi-frame video; dense OCR; small footprint | MIT | — |
| **Qwen2-VL 2B** | 2 B | Dynamic res, up to 2K | Multi-image, video, documents; strong OCR | Apache-2.0 | `qwen2-vl:2b` |
| **Qwen2-VL 7B** | 7 B | Dynamic res, up to 2K | Best-in-class 7 B vision; multi-image reasoning | Apache-2.0 | `qwen2-vl:7b` |
| **Qwen2-VL 72B** | 72 B | Dynamic res, up to 4K | GPT-4V level; video understanding | Apache-2.0 | — |
| **InternVL2-1B** | 0.9 B | 448 px | Smallest InternVL; on-device inference | MIT | — |
| **InternVL2-4B** | 4 B | 448 px | Balanced; SigLIP vision encoder | MIT | — |
| **InternVL2-8B** | 8 B | 448 px | Strong OCR; chart / diagram reading | MIT | — |
| **InternVL2-26B** | 26 B | 448 px | Multi-image; video frame VQA | MIT | — |
| **InternVL2-76B** | 76 B | 448 px | State-of-the-art open VLM (2024) | MIT | — |
| **LLaVA-1.5-7B** | 7 B | 336 × 336 | Strong VQA baseline; widely deployed | Apache-2.0 | `llava:7b` |
| **LLaVA-1.5-13B** | 13 B | 336 × 336 | Improved reasoning vs 7 B | Apache-2.0 | `llava:13b` |
| **LLaVA-NeXT-34B** | 34 B | Up to 672×672 anyres | High-res; multi-image reasoning | Apache-2.0 | — |
| **LLaVA-NeXT-72B** | 72 B | Up to 672×672 anyres | Top open VLM before Qwen2-VL era | Apache-2.0 | — |
| **LLaVA-Video-7B** | 7 B | Video frames (32 f) | Temporal video understanding | Apache-2.0 | — |
| **LLaVA-Video-72B** | 72 B | Video frames (32 f) | Best open video VLM (early 2025) | Apache-2.0 | — |
| **CogVLM2-19B** | 19 B | 1344 × 1344 | High-res; Chinese + English; deep spatial reasoning | Apache-2.0 | — |
| **MiniCPM-V 2.6** | 8 B | 1.8 M tokens (any-res) | 180+ language; video; real-time OCR | Apache-2.0 | — |
| **PaliGemma 3B** | 3 B | 224 / 448 / 896 px | Google fine-tune base; strong domain transfer | Gemma licence | — |
| **PaliGemma2 10B** | 10 B | Up to 896 px | Improved spatial grounding | Gemma licence | — |
| **BLIP-3 / xGen-MM** | 4 B | 256–1024 px | Interleaved multi-image; Salesforce release | Apache-2.0 | — |
| **Idefics3-8B** | 8 B | Any-res tiling | HuggingFace; DocVQA strong | MIT | — |

---

### Autonomous Driving–Specific VLMs

| Model | Params | Input | AD Capabilities | Source |
|-------|--------|-------|----------------|--------|
| **DriveLM** | 7 B | Multi-cam images + route | Graph-structured perception-prediction-planning VQA | [github.com/OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM) |
| **DriveVLM** | 7 B | Surround cameras | Chain-of-thought driving reasoning; scene understanding → action | [github.com/HKUST-Aerial-Robotics/DriveVLM](https://github.com/HKUST-Aerial-Robotics/DriveVLM) |
| **Dolphins** | 7 B | Dashcam + route | Conversational AD; describes and justifies decisions | [github.com/SaFoLab-WISC/Dolphins](https://github.com/SaFoLab-WISC/Dolphins) |
| **DriveGPT4** | 7 B | Video frames | Temporal event narration; driving Q&A | [github.com/Pointcloud-Inc/DriveGPT4](https://github.com/Pointcloud-Inc/DriveGPT4) |
| **DriveMLM** | 7 B | Front camera + HD map | Decision-making language model with rule-based alignment | [github.com/OpenDriveLab/DriveMLM](https://github.com/OpenDriveLab/DriveMLM) |
| **ELM** | 7 B | Multi-modal surround | Embodied language model; sensor-spatial grounding | [github.com/OpenDriveLab/ELM](https://github.com/OpenDriveLab/ELM) |
| **NuScenes-QA fine-tunes** | 7 B | nuScenes cameras | Domain-specific spatial QA on real AD datasets | Various HuggingFace repos |
| **MAPLM** | 3 B | BEV + camera | Map understanding + visual Q&A | [github.com/NVIDIA/MAPLM](https://github.com/NVIDIA/MAPLM) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     VLM Architecture                     │
│                                                          │
│  ┌─────────────────┐      ┌──────────────────────────┐  │
│  │  Vision Encoder  │      │   Language Model (LLM)  │  │
│  │                  │      │                          │  │
│  │  Surround Cams   │──────▶  Projection / Adapter   │  │
│  │  (CLIP/SigLIP/  │──────▶  (MLP or Q-Former)      │  │
│  │   DINOv2/EVA)    │      │                          │  │
│  │                  │      │  Mistral / LLaMA /       │  │
│  │  Optional:        │      │  Phi / Qwen2 backbone   │  │
│  │  LiDAR voxels     │      │                          │  │
│  │  Radar BEV map    │      │  Token interleaving for  │  │
│  └─────────────────┘      │  multi-image / video      │  │
│                             └──────────────────────────┘  │
│                                         │                  │
│                                         ▼                  │
│                          ┌──────────────────────────────┐ │
│                          │     Structured Output         │ │
│                          │  Scene description / VQA /   │ │
│                          │  Bounding boxes / Hazard tags │ │
│                          └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Common Vision Encoders

| Encoder | Params | Strengths |
|---------|--------|-----------|
| **CLIP ViT-L/14** | 307 M | Strong zero-shot; broad semantic | 
| **SigLIP ViT-SO400M** | 400 M | Better localisation; multi-lingual |
| **DINOv2 ViT-L** | 307 M | Dense features; 3D-aware |
| **EVA-CLIP** | 18 B (largest) | Best open image encoder (2024) |
| **InternViT-6B** | 6 B | Highest-res patches; InternVL series |

---

## AD-Relevant Tasks & Benchmark Performance

| Task | Best Open VLM (2025) | Notes |
|------|---------------------|-------|
| Scene description | Qwen2-VL-72B | 5-cam surround view |
| Hazard detection VQA | InternVL2-76B | Outperforms GPT-4V on nuScenes-QA |
| Traffic sign OCR | MiniCPM-V 2.6 | Sub-1 s on RTX 4090 |
| Lane / road marking Q&A | DriveLM-7B | AD-specific training |
| Driving decision explanation | DriveVLM | Step-by-step CoT |
| Real-time edge (< 30 ms) | Moondream 2 / InternVL2-1B | INT4 on Jetson Orin |

---

## Python Integration

See [`vlm_client_example.py`](vlm_client_example.py) for a working multi-provider VLM client with real driving-scene analysis examples.

```python
from vlm_client_example import VLMClient, load_image

client = VLMClient(provider="ollama")   # or "claude" / "gemini"
resp = client.chat(
    "List all traffic hazards in this scene.",
    image=load_image("scene.jpg"),
    system="You are an AV perception system.",
    temperature=0.1,
)
print(resp.text)
```

---

## Deployment Sizes (Quantised)

| Model | FP16 VRAM | INT8 VRAM | INT4 VRAM | Edge Target |
|-------|-----------|-----------|-----------|------------|
| Moondream 2 | 4 GB | 2 GB | 1.5 GB | Jetson Nano / RPi 5 |
| Qwen2-VL 2B | 5 GB | 3 GB | 2 GB | Jetson Orin NX 8 |
| InternVL2-4B | 8 GB | 4.5 GB | 2.5 GB | Jetson Orin NX 16 |
| Phi-3.5-Vision | 9 GB | 5 GB | 3 GB | Jetson Orin NX 16 |
| LLaVA-1.5-7B | 14 GB | 7.5 GB | 4 GB | Jetson AGX Orin |
| Qwen2-VL 7B | 15 GB | 8 GB | 4.5 GB | Jetson AGX Orin |
| InternVL2-8B | 16 GB | 9 GB | 5 GB | Jetson AGX Orin |

---

*See also: [VLA Models](VLA_models.md) | [World Models](world_models.md) | [Foundation Models](foundation_models.md) | [Edge Deployment](edge_deployment.md)*
