"""
World Model Client — Autonomous Driving Scene Prediction & Simulation

World models learn a compressed temporal representation of the environment and
predict future scene states, agent trajectories, and risk from current observations.

This client wraps LLM providers with structured temporal-prediction prompting,
implementing a world-model-style interface using general-purpose language models.
For production AD world models see: GAIA-1 (Wayve), DriveDreamer, UniSim, Vista.

Providers:
  claude  → Anthropic Claude
  gemini  → Google Gemini
  ollama  → Any Ollama model (qwen2:0.5b works; larger models give richer output)

Install:
    pip install -r requirements.txt

Key capabilities demonstrated:
  predict_next_state()       — forecast scene state over a time horizon
  assess_scenario_risk()     — risk breakdown by category (collision, pedestrian…)
  simulate_trajectory()      — ego path simulation given a specific action
  counterfactual_analysis()  — compare outcomes of two candidate actions
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER = "ollama"   # "claude" | "gemini" | "ollama"

MODELS = {
    "claude": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-2.0-flash",
    "ollama": "qwen2:0.5b",    # text-based world modeling; larger = richer output
}


# ─────────────────────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """State of a traffic agent at a single point in time."""
    agent_id: str
    agent_type: str           # vehicle | pedestrian | cyclist | motorcycle
    x_m: float                # lateral offset (m), positive = right of ego lane
    y_m: float                # longitudinal distance (m), positive = ahead of ego
    speed_kmh: float
    heading_deg: float        # compass heading, 0 = North, clockwise
    intent: str = "unknown"   # follow_lane | turning | stopping | crossing


@dataclass
class SceneState:
    """Complete description of the driving scene at a single timestamp."""
    timestamp_s: float
    ego_speed_kmh: float
    ego_heading_deg: float
    road_type: str            # highway | urban | rural | parking
    speed_limit_kmh: float
    weather: str              # clear | rain | fog | snow | overcast
    agents: List[AgentState] = field(default_factory=list)
    traffic_light: str = "none"   # none | red | amber | green
    description: str = ""


@dataclass
class TrajectoryPoint:
    """Predicted ego-vehicle pose at a future time offset."""
    t_s: float        # seconds from now
    x_m: float        # lateral displacement from current position
    y_m: float        # longitudinal displacement from current position
    speed_kmh: float
    heading_deg: float


@dataclass
class WorldPrediction:
    """Structured output from the world model."""
    horizon_s: float
    ego_trajectory: List[TrajectoryPoint]
    risk_score: float           # 0.0 = safe, 1.0 = imminent collision
    critical_events: List[str]
    predicted_agent_positions: List[dict]
    recommended_action: str
    confidence: float


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

_WM_SYSTEM = """\
You are an autonomous driving world model.
Given a current scene state, predict future states, trajectories, and risks.
Be precise, conservative, and safety-focused.
Always respond with ONLY valid JSON — no extra text, no markdown fences."""


def _parse_json(text: str) -> dict:
    """Extract the first JSON object or array from model output."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    raw = m.group(1) if m else text
    m2 = re.search(r"[\[{][\s\S]*[\]}]", raw)
    return json.loads(m2.group(0) if m2 else raw)


def _scene_to_text(s: SceneState) -> str:
    """Serialise a SceneState to a compact text representation."""
    agent_lines = "".join(
        f"  {a.agent_type} '{a.agent_id}': "
        f"({a.x_m:+.0f}m lat, {a.y_m:.0f}m ahead), "
        f"{a.speed_kmh:.0f} km/h, hdg {a.heading_deg:.0f}°, intent={a.intent}\n"
        for a in s.agents
    ) or "  (none)\n"
    return (
        f"t={s.timestamp_s:.1f}s\n"
        f"Ego: {s.ego_speed_kmh:.0f} km/h, heading {s.ego_heading_deg:.0f}°\n"
        f"Road: {s.road_type}, limit {s.speed_limit_kmh:.0f} km/h\n"
        f"Weather: {s.weather}  |  Traffic light: {s.traffic_light}\n"
        f"Agents:\n{agent_lines}"
        + (f"Context: {s.description}\n" if s.description else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
#  World Model Client
# ─────────────────────────────────────────────────────────────────────────────

class WorldModelClient:
    """
    Multi-provider World Model client.

    Uses LLM backends with structured temporal-prediction prompting to simulate
    the core capabilities of an autonomous driving world model:
      - next-state prediction
      - risk assessment
      - trajectory simulation
      - counterfactual analysis
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

    # ── Unified call ──────────────────────────────────────────────────────────

    def _call(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> str:
        if self.provider == "claude":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=_WM_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        if self.provider == "gemini":
            from google.genai import types
            cfg = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                system_instruction=_WM_SYSTEM,
                **({"response_mime_type": "application/json"} if json_mode else {}),
            )
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=cfg,
            )
            return resp.text

        if self.provider == "ollama":
            import ollama
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": _WM_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            if json_mode:
                kwargs["format"] = "json"
            resp = ollama.chat(**kwargs)
            return resp["message"]["content"]

    # ── World model capabilities ───────────────────────────────────────────────

    def predict_next_state(
        self, scene: SceneState, horizon_s: float = 3.0
    ) -> WorldPrediction:
        """
        Forecast ego trajectory, agent positions, and risk over `horizon_s` seconds.
        Returns a WorldPrediction with sampled trajectory waypoints at ~1 s intervals.
        """
        schema = (
            '{\n'
            '  "horizon_s": <float>,\n'
            '  "ego_trajectory": [{"t_s":<float>,"x_m":<float>,"y_m":<float>,'
            '"speed_kmh":<float>,"heading_deg":<float>}],\n'
            '  "risk_score": <float 0-1>,\n'
            '  "critical_events": ["<event>"],\n'
            '  "predicted_agent_positions": [{"id":"<id>","t_s":<float>,"x_m":<float>,"y_m":<float>}],\n'
            '  "recommended_action": "<string>",\n'
            '  "confidence": <float 0-1>\n'
            '}'
        )
        prompt = (
            f"Current scene:\n{_scene_to_text(scene)}\n"
            f"Predict the scene over {horizon_s:.0f} s.\n"
            f"Include ego trajectory waypoints at 1-second intervals.\n\n"
            f"Respond with JSON matching:\n{schema}"
        )
        raw = self._call(prompt)
        try:
            d = _parse_json(raw)
            traj = [
                TrajectoryPoint(
                    t_s=float(p.get("t_s", 0)),
                    x_m=float(p.get("x_m", 0)),
                    y_m=float(p.get("y_m", 0)),
                    speed_kmh=float(p.get("speed_kmh", scene.ego_speed_kmh)),
                    heading_deg=float(p.get("heading_deg", scene.ego_heading_deg)),
                )
                for p in d.get("ego_trajectory", [])
            ]
            return WorldPrediction(
                horizon_s=float(d.get("horizon_s", horizon_s)),
                ego_trajectory=traj,
                risk_score=float(d.get("risk_score", 0.5)),
                critical_events=d.get("critical_events", []),
                predicted_agent_positions=d.get("predicted_agent_positions", []),
                recommended_action=d.get("recommended_action", "maintain"),
                confidence=float(d.get("confidence", 0.5)),
            )
        except Exception as e:
            return WorldPrediction(
                horizon_s=horizon_s,
                ego_trajectory=[],
                risk_score=0.9,
                critical_events=[f"Prediction error: {e}"],
                predicted_agent_positions=[],
                recommended_action="brake",
                confidence=0.0,
            )

    def assess_scenario_risk(self, scene: SceneState) -> dict:
        """
        Evaluate overall scenario risk with a per-category breakdown.
        Returns dict with: overall_risk, collision_risk, pedestrian_risk,
        road_condition_risk, visibility_risk, top_risk_factors, mitigation.
        """
        schema = (
            '{"overall_risk":<float 0-1>,"collision_risk":<float 0-1>,'
            '"pedestrian_risk":<float 0-1>,"road_condition_risk":<float 0-1>,'
            '"visibility_risk":<float 0-1>,'
            '"top_risk_factors":["<factor>"],"mitigation":"<action>"}'
        )
        prompt = (
            f"Scene:\n{_scene_to_text(scene)}\n"
            f"Perform a risk assessment.\nRespond with JSON:\n{schema}"
        )
        raw = self._call(prompt, max_tokens=512)
        try:
            return _parse_json(raw)
        except Exception as e:
            return {"overall_risk": 0.9, "mitigation": "brake", "error": str(e)}

    def simulate_trajectory(
        self, scene: SceneState, action: str, horizon_s: float = 5.0
    ) -> List[TrajectoryPoint]:
        """
        Simulate the ego trajectory if the given action is executed.
        Returns a list of TrajectoryPoints at 1-second intervals.
        """
        prompt = (
            f"Scene:\n{_scene_to_text(scene)}\n"
            f"Simulate ego trajectory if it executes: '{action}'.\n"
            f"Provide {int(horizon_s)+1} waypoints at 1-second intervals (t=0 to t={int(horizon_s)}).\n\n"
            f"Respond with a JSON array:\n"
            f'[{{"t_s":<float>,"x_m":<float>,"y_m":<float>,"speed_kmh":<float>,"heading_deg":<float>}}]'
        )
        raw = self._call(prompt, max_tokens=512)
        try:
            m = re.search(r"\[[\s\S]*\]", raw)
            data = json.loads(m.group(0) if m else raw)
            return [
                TrajectoryPoint(
                    t_s=float(p["t_s"]),
                    x_m=float(p["x_m"]),
                    y_m=float(p["y_m"]),
                    speed_kmh=float(p["speed_kmh"]),
                    heading_deg=float(p["heading_deg"]),
                )
                for p in data
            ]
        except Exception:
            return []

    def counterfactual_analysis(
        self, scene: SceneState, action_a: str, action_b: str
    ) -> dict:
        """
        Compare the predicted outcomes of two candidate actions.
        Returns dict with outcome, risk_score, and time_to_clear for each,
        plus recommended action and one-sentence reasoning.
        """
        schema = (
            '{"action_a":{"outcome":"<string>","risk_score":<float>,"time_to_clear_s":<float>},'
            '"action_b":{"outcome":"<string>","risk_score":<float>,"time_to_clear_s":<float>},'
            '"recommended":"A"|"B","reasoning":"<one sentence>"}'
        )
        prompt = (
            f"Scene:\n{_scene_to_text(scene)}\n"
            f"Action A: '{action_a}'\n"
            f"Action B: '{action_b}'\n\n"
            f"Compare the predicted outcomes and recommend the safer action.\n"
            f"Respond with JSON:\n{schema}"
        )
        raw = self._call(prompt, max_tokens=512)
        try:
            return _parse_json(raw)
        except Exception as e:
            return {"recommended": "A", "reasoning": f"Parse error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
#  Sample scenes
# ─────────────────────────────────────────────────────────────────────────────

def _highway_scene() -> SceneState:
    return SceneState(
        timestamp_s=0.0,
        ego_speed_kmh=105.0,
        ego_heading_deg=270.0,
        road_type="highway",
        speed_limit_kmh=120.0,
        weather="clear",
        agents=[
            AgentState("V1", "vehicle",  0.0, 35.0,  85.0, 270.0, "follow_lane"),
            AgentState("V2", "vehicle", -3.5, 15.0, 110.0, 270.0, "follow_lane"),
        ],
        description="Three-lane motorway. Ego intends to overtake V1 in the centre lane.",
    )


def _urban_scene() -> SceneState:
    return SceneState(
        timestamp_s=0.0,
        ego_speed_kmh=42.0,
        ego_heading_deg=0.0,
        road_type="urban",
        speed_limit_kmh=50.0,
        weather="rain",
        agents=[
            AgentState("P1", "pedestrian", -2.0, 18.0, 5.0, 90.0, "crossing"),
            AgentState("P2", "pedestrian",  1.0, 18.0, 4.0, 90.0, "crossing"),
            AgentState("V1", "vehicle",     0.0, 55.0, 38.0, 0.0, "follow_lane"),
        ],
        traffic_light="amber",
        description="Rainy urban intersection. Amber light, pedestrians actively crossing.",
    )


def _fog_emergency_scene() -> SceneState:
    return SceneState(
        timestamp_s=0.0,
        ego_speed_kmh=70.0,
        ego_heading_deg=90.0,
        road_type="rural",
        speed_limit_kmh=80.0,
        weather="fog",
        agents=[
            AgentState("V1", "vehicle", 0.0, 15.0, 0.0, 90.0, "stopping"),
        ],
        description="Dense fog, stationary vehicle 15 m ahead. Right lane is clear.",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Autonomous Driving Examples
# ─────────────────────────────────────────────────────────────────────────────

def example_predict_next_state():
    """Predict future scene state and ego trajectory on a motorway."""
    client = WorldModelClient()
    pred = client.predict_next_state(_highway_scene(), horizon_s=4.0)

    print(f"=== Next State Prediction [{PROVIDER}] ===")
    print(f"  Horizon      : {pred.horizon_s:.0f} s")
    print(f"  Risk score   : {pred.risk_score:.2f}")
    print(f"  Confidence   : {pred.confidence:.2f}")
    print(f"  Action       : {pred.recommended_action}")
    print(f"  Critical events:")
    for e in pred.critical_events:
        print(f"    - {e}")
    print(f"  Ego trajectory:")
    for p in pred.ego_trajectory:
        print(f"    t={p.t_s:.0f}s  Δy={p.y_m:+.0f}m  {p.speed_kmh:.0f} km/h")
    print()


def example_risk_assessment():
    """Risk breakdown for a rainy urban intersection with crossing pedestrians."""
    client = WorldModelClient()
    risk = client.assess_scenario_risk(_urban_scene())

    print(f"=== Risk Assessment [{PROVIDER}] ===")
    for k, v in risk.items():
        print(f"  {k}: {v}")
    print()


def example_simulate_trajectory():
    """Simulate ego trajectory under two competing actions and compare."""
    client = WorldModelClient()
    scene = _highway_scene()

    for action in ["maintain speed and follow lane", "accelerate and change to left lane"]:
        traj = client.simulate_trajectory(scene, action, horizon_s=5.0)
        print(f"=== Trajectory Simulation: '{action}' [{PROVIDER}] ===")
        for p in traj:
            print(f"  t={p.t_s:.0f}s  Δy={p.y_m:+.1f}m  {p.speed_kmh:.0f} km/h  hdg={p.heading_deg:.0f}°")
        print()


def example_counterfactual():
    """Compare emergency brake vs. evasive lane change in a fog scenario."""
    client = WorldModelClient()
    result = client.counterfactual_analysis(
        scene=_fog_emergency_scene(),
        action_a="emergency brake to full stop",
        action_b="steer right into clear lane while decelerating",
    )

    print(f"=== Counterfactual Analysis [{PROVIDER}] ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Provider: {PROVIDER}  |  Model: {MODELS[PROVIDER]}\n")

    example_predict_next_state()
    example_risk_assessment()
    example_simulate_trajectory()
    example_counterfactual()
