import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from rothermel import RothermelModel, FireEnvironment, FireBehavior
from mattson import MattsonModel, SubjectCategory, MattsonResult
from noaa_feed import NOAAFeed, WeatherConditions
from fairsite_wrapper import predict_fire_spread

@dataclass
class ConsensusZone:
    lat: float
    lon: float
    radius_km: float
    color: str           # red, orange, yellow
    priority: int        # 1 = highest
    models_agreeing: List[str]
    threat_level: str
    probability: float
    deploy: str          # drone, rover, sub
    label: str

@dataclass
class ConsensusMap:
    location_lat: float
    location_lon: float
    zones: List[ConsensusZone]
    weather: Optional[WeatherConditions]
    fire_behavior: Optional[FireBehavior]
    sar_result: Optional[MattsonResult]
    active_models: List[str]
    timestamp: str

    def summary(self):
        lines = [
            f"\n{'='*60}",
            f"BEACON CONSENSUS PRIORITY MAP",
            f"{'='*60}",
            f"Location: ({self.location_lat}, {self.location_lon})",
            f"Active Models: {', '.join(self.active_models)}",
            f"",
            f"PRIORITY ZONES:",
        ]
        for zone in sorted(self.zones, key=lambda z: z.priority):
            lines.append(
                f"\n  [{zone.color.upper()}] Zone {zone.priority} — {zone.label}"
            )
            lines.append(f"  Models: {', '.join(zone.models_agreeing)}")
            lines.append(f"  Radius: {zone.radius_km:.1f}km from center")
            lines.append(f"  Threat: {zone.threat_level.upper()}")
            lines.append(f"  Deploy: {zone.deploy.upper()}")
        return "\n".join(lines)


class BeaconConsensus:
    """
    Multi-model consensus system.
    Combines all predictive models into one unified priority map.
    
    Priority rules:
    - RED:    3+ models agree OR any single critical threat
    - ORANGE: 2 models agree OR high threat
    - YELLOW: 1 model flags OR medium threat
    
    Vehicle assignment:
    - Drone:  aerial zones, thermal search, fire perimeter
    - Rover:  ground zones, debris, magnetometer
    - Sub:    underwater zones, flood areas
    """

    def __init__(self):
        self.rothermel = RothermelModel()
        self.mattson = MattsonModel()
        self.noaa = NOAAFeed()

    def run(
        self,
        lat: float,
        lon: float,
        scenario: str = "wildfire",
        subject_category: SubjectCategory = SubjectCategory.HIKER,
        hours_missing: float = 0,
        terrain: str = "forest",
        fuel_model: int = 4,
        slope: float = 10,
        run_farsite: bool = True
    ) -> ConsensusMap:

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        active_models = []
        model_outputs = {}
        zones = []
        weather = None
        fire_behavior = None
        sar_result = None

        # ── STEP 1: Pull live weather ──────────────────────────────
        print("[CONSENSUS] Fetching live weather from NOAA...")
        try:
            weather_data = self.noaa.get_fire_weather(lat, lon)
            weather = weather_data["conditions"]
            active_models.append("NOAA")
            print(f"[CONSENSUS] Weather: {weather.wind_speed_mph:.1f}mph, "
                  f"{weather.humidity:.0f}% humidity")
        except Exception as e:
            print(f"[CONSENSUS] NOAA unavailable: {e}")
            weather_data = {
                "wind_speed": 10.0,
                "wind_direction": 270.0,
                "moisture_1hr": 0.10,
                "moisture_10hr": 0.15,
                "moisture_100hr": 0.20,
                "moisture_live": 0.80,
            }

        # ── STEP 2: Run Rothermel ──────────────────────────────────
        if scenario in ["wildfire", "all"]:
            print("[CONSENSUS] Running Rothermel fire spread model...")
            try:
                env = FireEnvironment(
                    fuel_model=fuel_model,
                    wind_speed=weather_data["wind_speed"],
                    wind_direction=weather_data["wind_direction"],
                    slope=slope, aspect=180,
                    moisture_1hr=weather_data["moisture_1hr"],
                    moisture_10hr=weather_data["moisture_10hr"],
                    moisture_100hr=weather_data["moisture_100hr"],
                    moisture_live=weather_data["moisture_live"]
                )
                fire_behavior = self.rothermel.calculate(env)
                threat = self.rothermel.threat_level(fire_behavior)
                active_models.append("Rothermel")
                model_outputs["rothermel"] = {
                    "threat": threat,
                    "spread_mph": fire_behavior.spread_rate_mph,
                    "flame_length": fire_behavior.flame_length,
                    "direction": fire_behavior.direction
                }
                print(f"[CONSENSUS] Fire: {fire_behavior.spread_rate_mph:.2f}mph, "
                      f"threat={threat}")
            except Exception as e:
                print(f"[CONSENSUS] Rothermel error: {e}")

        # ── STEP 3: Run FARSITE ────────────────────────────────────
        if scenario in ["wildfire", "all"] and run_farsite:
            print("[CONSENSUS] Running FARSITE simulation...")
            try:
                farsite_zones = predict_fire_spread(
                    ignition_lat=lat,
                    ignition_lon=lon,
                    wind_speed=weather_data["wind_speed"],
                    wind_direction=weather_data["wind_direction"],
                    fuel_moisture=weather_data["moisture_1hr"] * 100,
                    duration_hours=6
                )
                active_models.append("FARSITE")
                model_outputs["farsite"] = farsite_zones
                print(f"[CONSENSUS] FARSITE: {len(farsite_zones)} priority zones generated")
            except Exception as e:
                print(f"[CONSENSUS] FARSITE error: {e}")

        # ── STEP 4: Run Mattson ────────────────────────────────────
        if scenario in ["sar", "all"] or hours_missing > 0:
            print("[CONSENSUS] Running Mattson SAR probability model...")
            try:
                sar_result = self.mattson.calculate(
                    lat, lon, subject_category, hours_missing, terrain
                )
                active_models.append("Mattson")
                model_outputs["mattson"] = sar_result
                print(f"[CONSENSUS] SAR Zone 1: "
                      f"{sar_result.search_areas[0].probability*100:.0f}% probability")
            except Exception as e:
                print(f"[CONSENSUS] Mattson error: {e}")

        # ── STEP 5: Build consensus zones ─────────────────────────
        print("[CONSENSUS] Building unified priority map...")

        # Count how many models flagged each threat level
        fire_threat = model_outputs.get("rothermel", {}).get("threat", "none")
        has_farsite = "farsite" in model_outputs
        has_sar = "mattson" in model_outputs

        fire_models = []
        if fire_threat in ["critical", "high"]:
            fire_models.append("Rothermel")
        if has_farsite:
            fire_models.append("FARSITE")

        # RED zone — highest priority
        red_models = []
        if fire_threat == "critical":
            red_models.append("Rothermel")
        if has_farsite and fire_threat in ["critical", "high"]:
            red_models.append("FARSITE")
        if has_sar and sar_result.search_areas[0].probability > 0.45:
            red_models.append("Mattson")

        if red_models or fire_threat == "critical":
            spread_radius = max(
                fire_behavior.spread_rate_mph * 2 if fire_behavior else 2.0,
                sar_result.search_areas[0].max_distance_km if sar_result else 2.0,
                2.0
            )
            zones.append(ConsensusZone(
                lat=lat, lon=lon,
                radius_km=min(spread_radius, 15.0),
                color="red",
                priority=1,
                models_agreeing=red_models or ["Rothermel"],
                threat_level="critical",
                probability=0.85,
                deploy="drone",
                label="CRITICAL — Immediate deployment required"
            ))

        # ORANGE zone
        orange_models = []
        if fire_threat in ["high", "medium"]:
            orange_models.append("Rothermel")
        if has_sar and sar_result.search_areas[1].probability > 0.15:
            orange_models.append("Mattson")
        if has_farsite:
            orange_models.append("FARSITE")

        if orange_models:
            zones.append(ConsensusZone(
                lat=lat, lon=lon,
                radius_km=min(
                    (fire_behavior.spread_rate_mph * 4 if fire_behavior else 5.0),
                    25.0
                ),
                color="orange",
                priority=2,
                models_agreeing=orange_models,
                threat_level="high",
                probability=0.65,
                deploy="rover",
                label="HIGH — Secondary deployment"
            ))

        # YELLOW zone — monitoring
        yellow_models = list(active_models)
        zones.append(ConsensusZone(
            lat=lat, lon=lon,
            radius_km=min(
                (fire_behavior.spread_rate_mph * 8 if fire_behavior else 10.0),
                40.0
            ),
            color="yellow",
            priority=3,
            models_agreeing=yellow_models,
            threat_level="medium",
            probability=0.35,
            deploy="drone",
            label="MODERATE — Monitor and prepare"
        ))

        return ConsensusMap(
            location_lat=lat,
            location_lon=lon,
            zones=zones,
            weather=weather,
            fire_behavior=fire_behavior,
            sar_result=sar_result,
            active_models=active_models,
            timestamp=timestamp
        )


if __name__ == "__main__":
    beacon = BeaconConsensus()

    print("🔺 BEACON MULTI-MODEL CONSENSUS SYSTEM")
    print("Combining NOAA + Rothermel + FARSITE + Mattson\n")

    # Scenario 1 — Wildfire + lost hiker
    result = beacon.run(
        lat=40.1, lon=-121.4,
        scenario="all",
        subject_category=SubjectCategory.HIKER,
        hours_missing=3,
        terrain="forest",
        fuel_model=4,
        slope=20
    )

    print(result.summary())