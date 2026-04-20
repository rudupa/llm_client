"""
VLA (Vision-Language-Action) Client — Autonomous Driving Decision Making

VLA models condition structured driving actions on visual perception and language
instructions, closing the loop from perception → reasoning → control.

This client wraps LLM/VLM providers with structured JSON action-output prompting,
implementing a VLA-style interface using general-purpose models.
For dedicated VLA model research see: OpenVLA, π₀ (pi-zero), RT-2, GR-2.

Providers:
  claude  → Anthropic Claude (vision-capable when image is supplied)
  gemini  → Google Gemini   (vision-capable when image is supplied)
  ollama  → qwen2:0.5b for state-based actions; moondream/llava for vision path

Install:
    pip install -r requirements.txt
    # For vision-conditioned VLA on Ollama:
    # ollama pull moondream

Output:
    DrivingAction dataclass with maneuver, steering angle, longitudinal command,
    target speed, confidence score, and one-sentence reasoning.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER = "ollama"   # "claude" | "gemini" | "ollama"

MODELS = {
    "claude": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-2.0-flash",
    "ollama": "qwen2:0.5b",    # text-based state → action; swap to moondream for vision
}


# ─────────────────────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EgoState:
    """Current ego-vehicle dynamic state."""
    speed_kmh: float          # current speed
    heading_deg: float        # compass heading (0 = North, 90 = East)
    lane: str                 # "left" | "center" | "right"
    turn_signal: str          # "none" | "left" | "right"
    acceleration_ms2: float   # positive = accelerating


@dataclass
class DrivingAction:
    """
    Structured driving action output from the VLA model.

    maneuver:           high-level intent
    steering_angle_deg: wheel angle, negative = left, positive = right
    longitudinal:       throttle/brake command
    target_speed_kmh:   desired speed after executing the action
    confidence:         model confidence 0.0 – 1.0
    reasoning:          one-sentence explanation
    """
    maneuver: str
    steering_angle_deg: float
    longitudinal: str
    target_speed_kmh: float
    confidence: float
    reasoning: str

    def display(self):
        print(f"  Maneuver      : {self.maneuver}")
        print(f"  Steering      : {self.steering_angle_deg:+.1f}°")
        print(f"  Longitudinal  : {self.longitudinal}")
        print(f"  Target speed  : {self.target_speed_kmh:.0f} km/h")
        print(f"  Confidence    : {self.confidence:.2f}")
        print(f"  Reasoning     : {self.reasoning}")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VLA_SYSTEM = """\
You are a Vision-Language-Action model for autonomous driving.
Given a scene description and ego-vehicle state, output a JSON driving action.
Always respond with ONLY a valid JSON object matching this exact schema:
{
  "maneuver": "follow_lane|lane_change_left|lane_change_right|turn_left|turn_right|brake|stop|accelerate",
  "steering_angle_deg": <float, -30 to 30>,
  "longitudinal": "accelerate|maintain|decelerate|emergency_stop",
  "target_speed_kmh": <float, 0 to 130>,
  "confidence": <float, 0.0 to 1.0>,
  "reasoning": "<one sentence>"
}
Prioritise safety. Do not include any text outside the JSON object."""


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from model output, handling code fences."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    raw = m.group(1) if m else text
    m2 = re.search(r"\{[\s\S]*\}", raw)
    return json.loads(m2.group(0) if m2 else raw)


def _safe_action(ego: EgoState, reason: str) -> DrivingAction:
    """Return a safe fallback action when model output cannot be parsed."""
    return DrivingAction(
        maneuver="brake",
        steering_angle_deg=0.0,
        longitudinal="decelerate",
        target_speed_kmh=max(0.0, ego.speed_kmh - 15.0),
        confidence=0.05,
        reasoning=f"Safe fallback — {reason}",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  VLA Client
# ─────────────────────────────────────────────────────────────────────────────

class VLAClient:
    """
    Multi-provider VLA client.

    predict_action() accepts a plain-text scene description + EgoState and
    optionally raw image bytes for vision-conditioned action prediction.
    All providers return a parsed DrivingAction with a safe fallback on error.
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

    def predict_action(
        self,
        scene: str,
        ego: EgoState,
        image: Optional[bytes] = None,
        temperature: float = 0.1,
    ) -> DrivingAction:
        """
        Predict a structured driving action.

        scene:       text description of the current driving scene
        ego:         current ego-vehicle state
        image:       optional raw image bytes for vision-conditioned path
        temperature: low values for deterministic, safety-critical output
        """
        prompt = (
            f"Scene:\n{scene}\n\n"
            f"Ego state:\n"
            f"  Speed         : {ego.speed_kmh:.1f} km/h\n"
            f"  Heading       : {ego.heading_deg:.0f}°\n"
            f"  Lane          : {ego.lane}\n"
            f"  Turn signal   : {ego.turn_signal}\n"
            f"  Acceleration  : {ego.acceleration_ms2:+.1f} m/s²\n\n"
            "Predict the optimal driving action as JSON."
        )

        raw = self._call(prompt, image, temperature)
        try:
            d = _parse_json(raw)
            return DrivingAction(
                maneuver=d.get("maneuver", "follow_lane"),
                steering_angle_deg=float(d.get("steering_angle_deg", 0.0)),
                longitudinal=d.get("longitudinal", "maintain"),
                target_speed_kmh=float(d.get("target_speed_kmh", ego.speed_kmh)),
                confidence=float(d.get("confidence", 0.5)),
                reasoning=d.get("reasoning", ""),
            )
        except Exception as e:
            return _safe_action(ego, str(e))

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _call(self, prompt: str, image: Optional[bytes], temperature: float) -> str:
        if self.provider == "claude":
            return self._claude_call(prompt, image, temperature)
        if self.provider == "gemini":
            return self._gemini_call(prompt, image, temperature)
        return self._ollama_call(prompt, image, temperature)

    # ── Claude ────────────────────────────────────────────────────────────────

    def _claude_call(self, prompt, image, temperature):
        import base64
        content = []
        if image:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png",
                           "data": base64.b64encode(image).decode()},
            })
        content.append({"type": "text", "text": prompt})
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=temperature,
            system=_VLA_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _gemini_call(self, prompt, image, temperature):
        from google.genai import types
        parts = []
        if image:
            parts.append(types.Part.from_bytes(data=image, mime_type="image/png"))
        parts.append(types.Part(text=prompt))
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=temperature,
                system_instruction=_VLA_SYSTEM,
                response_mime_type="application/json",
            ),
        )
        return resp.text

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _ollama_call(self, prompt, image, temperature):
        import ollama
        messages = [{"role": "system", "content": _VLA_SYSTEM}]
        user_msg = {"role": "user", "content": prompt}
        if image:
            user_msg["images"] = [image]
        messages.append(user_msg)
        resp = ollama.chat(
            model=self.model,
            messages=messages,
            format="json",
            options={"temperature": temperature, "num_predict": 512},
        )
        return resp["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
#  Autonomous Driving Examples
# ─────────────────────────────────────────────────────────────────────────────

def highway_cruise():
    """Steady-state motorway driving — slow vehicle in lane ahead."""
    client = VLAClient()
    ego = EgoState(speed_kmh=110.0, heading_deg=270.0, lane="center",
                   turn_signal="none", acceleration_ms2=0.0)
    scene = (
        "Three-lane motorway, dry conditions, clear weather, 120 km/h limit. "
        "Vehicle ahead in centre lane cruising at ~85 km/h, 40 m gap. "
        "Left lane is clear for at least 200 m."
    )
    action = client.predict_action(scene, ego)
    print(f"=== Highway Cruise [{PROVIDER}] ===")
    action.display()
    print()


def urban_intersection():
    """Signalised intersection — amber light with active pedestrian crossing."""
    client = VLAClient()
    ego = EgoState(speed_kmh=45.0, heading_deg=0.0, lane="right",
                   turn_signal="right", acceleration_ms2=-0.5)
    scene = (
        "Urban road, 50 km/h limit. Traffic light 25 m ahead shows amber. "
        "Two pedestrians are actively crossing on the right-turn path. "
        "Ego vehicle intends to turn right."
    )
    action = client.predict_action(scene, ego)
    print(f"=== Urban Intersection [{PROVIDER}] ===")
    action.display()
    print()


def emergency_obstacle():
    """Sudden large debris in lane at medium speed — emergency response."""
    client = VLAClient()
    ego = EgoState(speed_kmh=65.0, heading_deg=90.0, lane="center",
                   turn_signal="none", acceleration_ms2=0.2)
    scene = (
        "Rural two-lane road, 60 km/h limit. Large piece of truck tyre "
        "in centre lane, 12 m ahead. Right shoulder is clear. "
        "Oncoming vehicle visible at ~150 m in the opposing lane."
    )
    action = client.predict_action(scene, ego)
    print(f"=== Emergency Obstacle [{PROVIDER}] ===")
    action.display()
    print()


def parking_approach():
    """Low-speed parallel parking manoeuvre into an empty bay."""
    client = VLAClient()
    ego = EgoState(speed_kmh=8.0, heading_deg=180.0, lane="right",
                   turn_signal="right", acceleration_ms2=-0.1)
    scene = (
        "Car park, 10 km/h internal limit. Empty parallel bay 5 m ahead on the right. "
        "Bay is 2 m longer than ego vehicle. No pedestrians or vehicles in vicinity."
    )
    action = client.predict_action(scene, ego)
    print(f"=== Parking Approach [{PROVIDER}] ===")
    action.display()
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Provider: {PROVIDER}  |  Model: {MODELS[PROVIDER]}\n")

    highway_cruise()
    urban_intersection()
    emergency_obstacle()
    parking_approach()
