import requests
import json
from datetime import datetime, timezone
# ─── Constants ───────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


# ─── Prompt Builder ───────────────────────────────────────────────────────────
def build_prompt(consensus_map: dict, thermal_detections: list, fleet_status: dict) -> str:
    scenario = consensus_map.get("scenario", "Unknown scenario")

    red_zones = [
        zone for zone in consensus_map.get("zones", [])
        if zone.get("status") == "RED"
    ]

    red_zone_lines = []
    for z in red_zones:
        flagged_by = ", ".join(z.get("flagged_by", []))
        red_zone_lines.append(
            f"  - Zone '{z['id']}' at coords {z.get('coords', 'N/A')} "
            f"flagged by: {flagged_by}"
        )
    red_zone_text = "\n".join(red_zone_lines) if red_zone_lines else "  None"

    thermal_lines = []
    for t in thermal_detections:
        thermal_lines.append(
            f"  - ID '{t['id']}': confidence={t.get('confidence', 0):.0%}, "
            f"bearing={t.get('bearing_deg', 'N/A')}°, "
            f"label={t.get('label', 'unknown')}"
        )
    thermal_text = "\n".join(thermal_lines) if thermal_lines else "  None"

    vehicle_lines = []
    for vid, vdata in fleet_status.items():
        vehicle_lines.append(
            f"  - {vid}: type={vdata.get('type', 'unknown')}, "
            f"battery={vdata.get('battery_pct', 0)}%, "
            f"status={vdata.get('status', 'unknown')}"
        )
    vehicle_text = "\n".join(vehicle_lines) if vehicle_lines else "  None"

    prompt = f"""You are an AI decision engine for a drone-based search-and-rescue system.
Analyze the following operational data and provide a recommendation.

SCENARIO: {scenario}

RED ZONES (high-priority areas flagged by detection models):
{red_zone_text}

THERMAL DETECTIONS (possible survivor candidates):
{thermal_text}

AVAILABLE VEHICLES:
{vehicle_text}

Based on this data, respond with a JSON object in exactly this format:
{{
  "recommendation": "<one clear action in plain English>",
  "reasons": [
    "<reason 1 drawn directly from the model data>",
    "<reason 2 drawn directly from the model data>",
    "<reason 3 drawn directly from the model data>"
  ],
  "confidence": <integer 0-100>
}}

Rules:
- recommendation must be ONE sentence describing exactly what to do next.
- reasons must each reference specific data points (zone IDs, detection IDs, battery %, etc.).
- confidence reflects how certain you are given the available data (0=no data, 100=overwhelming evidence).
- Output ONLY the JSON object. No preamble, no explanation, no markdown fences.
"""
    return prompt


# ─── Ollama Call ─────────────────────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


# ─── Response Parser ──────────────────────────────────────────────────────────
def parse_response(text: str) -> tuple[str, list[str], int]:
    # Strip markdown fences if the model includes them despite instructions
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    # Find the first { ... } block
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        return _safe_defaults("No JSON block found in response")

    json_str = clean[start:end]

    try:
        parsed = json.loads(json_str)
        recommendation = str(parsed.get("recommendation", "No recommendation provided."))
        reasons = parsed.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        reasons = [str(r) for r in reasons[:3]]
        while len(reasons) < 3:
            reasons.append("No additional reason provided.")
        confidence = int(parsed.get("confidence", 50))
        confidence = max(0, min(100, confidence))
        return recommendation, reasons, confidence

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return _safe_defaults(f"JSON parse error: {e}")


def _safe_defaults(reason: str) -> tuple[str, list[str], int]:
    return (
        "Unable to generate recommendation — fallback to manual operator review.",
        [
            f"Parse failure: {reason}",
            "Recommend dispatching nearest available vehicle to highest-confidence thermal detection.",
            "Operator should cross-reference RED zones manually.",
        ],
        0,
    )


# ─── Orchestrator ─────────────────────────────────────────────────────────────
def get_recommendation(
    consensus_map: dict,
    thermal_detections: list,
    fleet_status: dict,
) -> tuple[str, list[str], int, str]:
    prompt = build_prompt(consensus_map, thermal_detections, fleet_status)
    raw_text = call_ollama(prompt)
    recommendation, reasons, confidence = parse_response(raw_text)
    timestamp = datetime.now(timezone.utc).isoformat()
    return recommendation, reasons, confidence, timestamp


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    consensus_map = {
        "scenario": "Post-earthquake urban search and rescue — downtown grid sector 4",
        "zones": [
            {
                "id": "ZONE-7",
                "status": "RED",
                "coords": (34.052, -118.243),
                "flagged_by": ["YOLOv8-collapse", "DepthNet-rubble", "AudioNet-cry"],
            }
        ],
    }

    thermal_detections = [
        {
            "id": "THERM-001",
            "label": "survivor_candidate",
            "confidence": 0.87,
            "bearing_deg": 142,
        }
    ]

    fleet_status = {
        "DRONE-ALPHA": {
            "type": "quadcopter",
            "battery_pct": 75,
            "status": "standby",
        }
    }

    print("=" * 60)
    print("BEACON — Ollama Recommendation Engine")
    print("=" * 60)

    recommendation, reasons, confidence, timestamp = get_recommendation(
        consensus_map, thermal_detections, fleet_status
    )

    print(f"\nTimestamp : {timestamp}")
    print(f"Confidence: {confidence}/100")
    print(f"\nRecommendation:\n  {recommendation}")
    print("\nReasons:")
    for i, r in enumerate(reasons, 1):
        print(f"  {i}. {r}")
    print("=" * 60)