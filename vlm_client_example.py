"""
VLM (Vision-Language Model) Client — Autonomous Driving Scene Understanding

VLMs process images + text together, enabling scene perception tasks critical
for autonomous driving: object detection, hazard assessment, condition rating.

Providers:
  claude  → Anthropic Claude (claude-3-7-sonnet-20250219, vision-capable)
  gemini  → Google Gemini   (gemini-2.0-flash, vision-capable)
  ollama  → Ollama with moondream (~1.7 GB) or llava (~4.7 GB)

Install:
    pip install -r requirements.txt
    ollama pull moondream          # smallest vision model (~1.7 GB)
    # ollama pull llava            # larger alternative (~4.7 GB)

NOTE: For the ollama provider the model MUST be vision-capable.
      Text-only models (qwen2, llama3.2) do NOT support image inputs.

Usage with a real image:
    from vlm_client_example import VLMClient, load_image
    client = VLMClient()
    resp = client.chat("Describe this scene.", image=load_image("scene.jpg"))
"""

import base64
import os
import struct
import zlib
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — change PROVIDER to switch backends
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER = "ollama"   # "claude" | "gemini" | "ollama"

MODELS = {
    "claude": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-2.0-flash",
    "ollama": "moondream",    # vision model; alternative: llava
}

OLLAMA_BASE_URL = "http://localhost:11434"


# ─────────────────────────────────────────────────────────────────────────────
#  Image utilities (stdlib only — no Pillow needed for test image)
# ─────────────────────────────────────────────────────────────────────────────

def create_test_scene_image() -> bytes:
    """
    Generate a synthetic road-scene PNG using stdlib only (no Pillow needed).
    Returns 320×240 RGB PNG: gradient sky (top), asphalt road (bottom),
    dashed yellow centre line and white lane-edge markings.
    """
    W, H = 320, 240
    rows = []
    for y in range(H):
        row = []
        for x in range(W):
            if y < H // 2:
                # Sky: gradient from grey-blue to blue
                r, g, b = 100, 150, min(255, 165 + y)
            elif abs(x - W // 2) < 4 and (y // 15) % 2:
                # Dashed centre line: yellow
                r, g, b = 255, 215, 0
            elif abs(x - W // 4) < 3 or abs(x - 3 * W // 4) < 3:
                # Lane-edge markings: white
                r, g, b = 200, 200, 200
            else:
                # Road: dark grey with perspective darkening towards horizon
                v = max(35, 75 - (y - H // 2) * 20 // (H // 2))
                r, g, b = v, v, v
            row += [r, g, b]
        rows.append(bytes(row))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # Each scanline: filter byte 0x00 (None) + W * 3 RGB bytes
    raw = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def load_image(path: str) -> bytes:
    """Load any image file (JPEG, PNG, etc.) as raw bytes."""
    return Path(path).read_bytes()


def _b64(img: bytes) -> str:
    return base64.b64encode(img).decode()


# ─────────────────────────────────────────────────────────────────────────────
#  Response type
# ─────────────────────────────────────────────────────────────────────────────

class VLMResponse:
    """Normalised response returned by all providers."""

    def __init__(self, text: str, input_tokens: int = 0, output_tokens: int = 0):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __str__(self) -> str:
        return self.text


# ─────────────────────────────────────────────────────────────────────────────
#  VLM Client
# ─────────────────────────────────────────────────────────────────────────────

class VLMClient:
    """
    Multi-provider Vision-Language Model client.
    Exposes a single chat() method that accepts an optional image (raw bytes).
    """

    def __init__(self, provider: str = PROVIDER):
        self.provider = provider.lower()
        self.model = MODELS[self.provider]
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "claude":
            from anthropic import Anthropic
            return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        if self.provider == "gemini":
            from google import genai
            return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        if self.provider == "ollama":
            import ollama as _ollama
            return _ollama

        raise ValueError(f"Unknown provider: {self.provider!r}")

    def chat(
        self,
        prompt: str,
        image: bytes = None,
        system: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> VLMResponse:
        """
        Send a prompt (+ optional image) to the VLM.

        prompt:      text question or instruction
        image:       raw image bytes (PNG/JPEG); None = text-only
        system:      optional system prompt
        max_tokens:  max response tokens
        temperature: sampling temperature (low = more deterministic)
        """
        if self.provider == "claude":
            return self._claude(prompt, image, system, max_tokens, temperature)
        if self.provider == "gemini":
            return self._gemini(prompt, image, system, max_tokens, temperature)
        if self.provider == "ollama":
            return self._ollama(prompt, image, system, max_tokens, temperature)

    # ── Claude ────────────────────────────────────────────────────────────────

    def _claude(self, prompt, image, system, max_tokens, temperature):
        content = []
        if image:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _b64(image)},
            })
        content.append({"type": "text", "text": prompt})

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        return VLMResponse(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _gemini(self, prompt, image, system, max_tokens, temperature):
        from google.genai import types

        parts = []
        if image:
            parts.append(types.Part.from_bytes(data=image, mime_type="image/png"))
        parts.append(types.Part(text=prompt))

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system,
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=config,
        )
        usage = resp.usage_metadata
        return VLMResponse(
            text=resp.text,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
        )

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _ollama(self, prompt, image, system, max_tokens, temperature):
        import ollama

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user_msg = {"role": "user", "content": prompt}
        if image:
            user_msg["images"] = [image]
        messages.append(user_msg)

        resp = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return VLMResponse(
            text=resp["message"]["content"],
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Autonomous Driving Examples
# ─────────────────────────────────────────────────────────────────────────────

def analyze_driving_scene(image: bytes = None):
    """Full scene analysis: road layout, agents, signs, weather, hazards."""
    client = VLMClient()
    img = image or create_test_scene_image()

    resp = client.chat(
        prompt=(
            "Analyze this driving scene and describe:\n"
            "1. Road layout and lane markings\n"
            "2. Vehicles (type, position, estimated distance)\n"
            "3. Pedestrians or cyclists\n"
            "4. Traffic signs or signals\n"
            "5. Weather and visibility conditions\n"
            "6. Immediate hazards or safety concerns"
        ),
        image=img,
        system=(
            "You are an autonomous driving perception system. "
            "Be precise and concise. Prioritise safety-critical details."
        ),
    )
    print(f"=== Scene Analysis [{PROVIDER}/{MODELS[PROVIDER]}] ===")
    print(resp.text)
    print(f"Tokens — in: {resp.input_tokens}  out: {resp.output_tokens}\n")


def detect_road_hazards(image: bytes = None):
    """Identify and classify road hazards with severity and recommended response."""
    client = VLMClient()
    img = image or create_test_scene_image()

    resp = client.chat(
        prompt=(
            "List all road hazards visible in this scene.\n"
            "For each: type | severity (low/medium/high/critical) | "
            "location (left/centre/right, near/mid/far) | recommended response."
        ),
        image=img,
        system="You are an AV safety perception system. Be conservative — never understate risk.",
        temperature=0.1,
    )
    print(f"=== Hazard Detection [{PROVIDER}/{MODELS[PROVIDER]}] ===")
    print(resp.text)
    print()


def assess_driving_conditions(image: bytes = None):
    """Rate weather, lighting, and road surface conditions for AV planning."""
    client = VLMClient()
    img = image or create_test_scene_image()

    resp = client.chat(
        prompt=(
            "Assess the driving conditions in this image:\n"
            "- Weather:    clear / rain / fog / snow / overcast\n"
            "- Visibility: estimate in metres\n"
            "- Lighting:   day / dusk / night / artificial\n"
            "- Road surface: dry / wet / icy / damaged\n"
            "- Speed adjustment: recommended % of posted limit\n"
            "Be concise."
        ),
        image=img,
        temperature=0.1,
    )
    print(f"=== Driving Conditions [{PROVIDER}/{MODELS[PROVIDER]}] ===")
    print(resp.text)
    print()


def multi_turn_scene_qa(image: bytes = None):
    """Ask a sequence of autonomous-driving questions about the same scene."""
    client = VLMClient()
    img = image or create_test_scene_image()
    system = "You are an expert AV perception assistant. Answer in 1–2 sentences."

    questions = [
        "What type of road environment is this?",
        "Is it safe to proceed at the current speed?",
        "What is the single most important object the ego vehicle must monitor?",
    ]

    print(f"=== Multi-Turn Scene Q&A [{PROVIDER}/{MODELS[PROVIDER]}] ===")
    for q in questions:
        resp = client.chat(prompt=q, image=img, system=system, max_tokens=128)
        print(f"Q: {q}\nA: {resp.text}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Provider: {PROVIDER}  |  Model: {MODELS[PROVIDER]}")
    print("NOTE: Ollama requires a vision-capable model.")
    print("      Run: ollama pull moondream   (or: ollama pull llava)\n")

    # To use a real image file, pass it explicitly:
    #   analyze_driving_scene(load_image("scene.jpg"))

    analyze_driving_scene()
    detect_road_hazards()
    assess_driving_conditions()
    multi_turn_scene_qa()
