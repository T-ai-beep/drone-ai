import math
from dataclasses import dataclass
from typing import Dict, List
from shakemap import ShakeMap, ShakeMapResult, EarthquakeEvent

@dataclass
class BuildingType:
    code: str
    name: str
    slight_threshold: float    # PGA (g) for slight damage
    moderate_threshold: float  # PGA (g) for moderate damage
    extensive_threshold: float # PGA (g) for extensive damage
    complete_threshold: float  # PGA (g) for complete collapse
    occupancy_per_sqft: float  # people per sqft
    casualty_rate_complete: float  # casualty rate at complete damage

# HAZUS building types with fragility parameters
BUILDING_TYPES = {
    "W1":  BuildingType("W1",  "Wood Single Family",      0.10, 0.25, 0.50, 0.80, 0.004, 0.05),
    "W2":  BuildingType("W2",  "Wood Commercial",         0.12, 0.30, 0.60, 1.00, 0.003, 0.04),
    "S1L": BuildingType("S1L", "Steel Moment Frame Low",  0.15, 0.35, 0.70, 1.20, 0.005, 0.03),
    "S1M": BuildingType("S1M", "Steel Moment Frame Mid",  0.12, 0.28, 0.55, 0.95, 0.005, 0.04),
    "S1H": BuildingType("S1H", "Steel Moment Frame High", 0.10, 0.25, 0.50, 0.85, 0.005, 0.05),
    "C1L": BuildingType("C1L", "Concrete Frame Low",      0.12, 0.28, 0.55, 0.90, 0.005, 0.06),
    "C1M": BuildingType("C1M", "Concrete Frame Mid",      0.10, 0.22, 0.45, 0.75, 0.005, 0.07),
    "C1H": BuildingType("C1H", "Concrete Frame High",     0.08, 0.18, 0.38, 0.65, 0.005, 0.08),
    "C2L": BuildingType("C2L", "Concrete Shear Wall Low", 0.15, 0.35, 0.70, 1.10, 0.005, 0.05),
    "URM": BuildingType("URM", "Unreinforced Masonry",    0.06, 0.14, 0.28, 0.50, 0.005, 0.15),
    "MH":  BuildingType("MH",  "Mobile Home",             0.08, 0.18, 0.35, 0.60, 0.003, 0.08),
}

@dataclass
class DamageState:
    building_type: str
    pga: float
    prob_slight: float
    prob_moderate: float
    prob_extensive: float
    prob_complete: float
    expected_damage_state: str
    structural_loss_ratio: float

@dataclass
class CasualtyEstimate:
    indoor_casualties: int
    indoor_fatalities: int
    outdoor_casualties: int
    total_affected: int
    severity_1: int  # minor injuries
    severity_2: int  # hospitalized
    severity_3: int  # life threatening
    severity_4: int  # fatalities

@dataclass
class HAZUSResult:
    event: EarthquakeEvent
    site_lat: float
    site_lon: float
    pga: float
    mmi: float
    damage_states: Dict[str, DamageState]
    casualties: CasualtyEstimate
    fire_ignition_probability: float
    utility_damage: Dict[str, float]
    threat_level: str
    beacon_priority: int

    def summary(self):
        lines = [
            f"PGA: {self.pga:.4f}g  MMI: {self.mmi:.1f}",
            f"Threat: {self.threat_level.upper()}",
            f"",
            f"STRUCTURAL DAMAGE:"
        ]
        for code, state in self.damage_states.items():
            lines.append(
                f"  {code:5s} ({BUILDING_TYPES[code].name[:20]:20s}): "
                f"{state.expected_damage_state:10s} "
                f"[complete: {state.prob_complete*100:.0f}%]"
            )
        lines.extend([
            f"",
            f"CASUALTY ESTIMATES:",
            f"  Minor injuries:      {self.casualties.severity_1}",
            f"  Hospitalized:        {self.casualties.severity_2}",
            f"  Life threatening:    {self.casualties.severity_3}",
            f"  Fatalities:          {self.casualties.severity_4}",
            f"  Total affected:      {self.casualties.total_affected}",
            f"",
            f"SECONDARY HAZARDS:",
            f"  Fire ignition prob:  {self.fire_ignition_probability*100:.0f}%",
            f"  Power system:        {self.utility_damage['power']*100:.0f}% damaged",
            f"  Water system:        {self.utility_damage['water']*100:.0f}% damaged",
            f"  Road network:        {self.utility_damage['roads']*100:.0f}% damaged",
        ])
        return "\n".join(lines)


class HAZUSModel:
    """
    FEMA HAZUS Earthquake Loss Estimation Model.
    
    Reference: FEMA 2020. Hazus Earthquake Model Technical Manual.
    Hazus 4.2. Federal Emergency Management Agency.
    
    Uses fragility curves to estimate structural damage probability
    at each shaking intensity level, then calculates casualties
    and secondary hazards.
    """

    def __init__(self):
        self.shakemap = ShakeMap()

    def _fragility_curve(
        self,
        pga: float,
        threshold: float,
        beta: float = 0.6
    ) -> float:
        """
        Lognormal fragility curve.
        P(damage >= state) = Phi(ln(PGA/threshold) / beta)
        Beta = 0.6 is standard HAZUS value.
        """
        if pga <= 0 or threshold <= 0:
            return 0.0
        z = math.log(pga / threshold) / beta
        # Standard normal CDF approximation
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _damage_state(
        self,
        building: BuildingType,
        pga: float
    ) -> DamageState:
        """Calculate damage state probabilities for a building type."""
        p_slight = self._fragility_curve(pga, building.slight_threshold)
        p_moderate = self._fragility_curve(pga, building.moderate_threshold)
        p_extensive = self._fragility_curve(pga, building.extensive_threshold)
        p_complete = self._fragility_curve(pga, building.complete_threshold)

        # Determine expected damage state
        if p_complete > 0.50:
            expected = "COMPLETE"
            loss_ratio = 0.85
        elif p_extensive > 0.50:
            expected = "EXTENSIVE"
            loss_ratio = 0.45
        elif p_moderate > 0.50:
            expected = "MODERATE"
            loss_ratio = 0.15
        elif p_slight > 0.50:
            expected = "SLIGHT"
            loss_ratio = 0.03
        else:
            expected = "NONE"
            loss_ratio = 0.0

        return DamageState(
            building_type=building.code,
            pga=pga,
            prob_slight=p_slight,
            prob_moderate=p_moderate,
            prob_extensive=p_extensive,
            prob_complete=p_complete,
            expected_damage_state=expected,
            structural_loss_ratio=loss_ratio
        )

    def _estimate_casualties(
        self,
        damage_states: Dict[str, DamageState],
        population: int = 10000
    ) -> CasualtyEstimate:
        """
        Estimate casualties from damage states.
        Uses HAZUS casualty model — time of day dependent.
        Using 2AM (worst case, max indoor population).
        """
        total_s1 = 0  # minor
        total_s2 = 0  # hospitalized
        total_s3 = 0  # life threatening
        total_s4 = 0  # fatalities

        # HAZUS casualty rates by damage state (Table 13.3)
        casualty_rates = {
            "COMPLETE":  {"s1": 0.05, "s2": 0.01, "s3": 0.001, "s4": 0.0001},
            "EXTENSIVE": {"s1": 0.01, "s2": 0.001, "s3": 0.0001, "s4": 0.00001},
            "MODERATE":  {"s1": 0.005, "s2": 0.0001, "s3": 0.00001, "s4": 0.0},
            "SLIGHT":    {"s1": 0.0005, "s2": 0.0, "s3": 0.0, "s4": 0.0},
            "NONE":      {"s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0},
        }

        for code, state in damage_states.items():
            rates = casualty_rates.get(state.expected_damage_state, casualty_rates["NONE"])
            pop_share = population / len(damage_states)

            total_s1 += int(pop_share * rates["s1"])
            total_s2 += int(pop_share * rates["s2"])
            total_s3 += int(pop_share * rates["s3"])
            total_s4 += int(pop_share * rates["s4"])

        return CasualtyEstimate(
            indoor_casualties=total_s1 + total_s2 + total_s3,
            indoor_fatalities=total_s4,
            outdoor_casualties=int(total_s1 * 0.1),
            total_affected=total_s1 + total_s2 + total_s3 + total_s4,
            severity_1=total_s1,
            severity_2=total_s2,
            severity_3=total_s3,
            severity_4=total_s4
        )

    def _fire_ignition_probability(self, pga: float) -> float:
        """
        Probability of post-earthquake fire ignition.
        Based on HAZUS fire following earthquake model.
        Higher PGA = more broken gas lines = more fires.
        """
        if pga < 0.10:
            return 0.01
        elif pga < 0.20:
            return 0.05
        elif pga < 0.40:
            return 0.15
        elif pga < 0.60:
            return 0.35
        else:
            return 0.60

    def _utility_damage(self, pga: float) -> Dict[str, float]:
        """Estimate utility system damage ratios."""
        return {
            "power": min(pga * 2.5, 1.0),
            "water": min(pga * 3.0, 1.0),
            "gas": min(pga * 2.0, 1.0),
            "roads": min(pga * 1.5, 1.0),
            "bridges": min(pga * 3.5, 1.0),
        }

    def _threat_from_damage(
        self,
        damage_states: Dict[str, DamageState]
    ) -> tuple:
        complete_count = sum(
            1 for s in damage_states.values()
            if s.expected_damage_state == "COMPLETE"
        )
        extensive_count = sum(
            1 for s in damage_states.values()
            if s.expected_damage_state == "EXTENSIVE"
        )

        if complete_count >= 3:
            return "critical", 1
        elif complete_count >= 1 or extensive_count >= 3:
            return "high", 2
        elif extensive_count >= 1:
            return "medium", 3
        return "low", 4

    def calculate(
        self,
        event: EarthquakeEvent,
        site_lat: float,
        site_lon: float,
        vs30: float = 360.0,
        population: int = 10000
    ) -> HAZUSResult:
        """
        Run full HAZUS earthquake damage assessment.
        
        Args:
            event: Earthquake parameters
            site_lat, site_lon: Assessment location
            vs30: Site soil condition (760=rock, 360=stiff soil, 180=soft soil)
            population: Population in assessment area
        """
        # Get ground shaking from ShakeMap
        shake = self.shakemap.calculate(event, site_lat, site_lon, vs30)
        pga = shake.pga

        # Calculate damage states for all building types
        damage_states = {}
        for code, building in BUILDING_TYPES.items():
            damage_states[code] = self._damage_state(building, pga)

        # Estimate casualties
        casualties = self._estimate_casualties(damage_states, population)

        # Secondary hazards
        fire_prob = self._fire_ignition_probability(pga)
        utility = self._utility_damage(pga)

        # Overall threat
        threat, priority = self._threat_from_damage(damage_states)

        return HAZUSResult(
            event=event,
            site_lat=site_lat,
            site_lon=site_lon,
            pga=pga,
            mmi=shake.mmi,
            damage_states=damage_states,
            casualties=casualties,
            fire_ignition_probability=fire_prob,
            utility_damage=utility,
            threat_level=threat,
            beacon_priority=priority
        )

    def beacon_priority_zones(
        self,
        event: EarthquakeEvent,
        center_lat: float,
        center_lon: float
    ) -> dict:
        """Generate Beacon priority zones from HAZUS output."""
        zones = {}
        radii = [5, 20, 50]
        colors = ["red", "orange", "yellow"]

        for i, (radius, color) in enumerate(zip(radii, colors)):
            site_lat = center_lat + (radius / 111.0)
            result = self.calculate(event, site_lat, center_lon)

            zones[color] = {
                "priority": i + 1,
                "radius_km": radius,
                "pga": result.pga,
                "mmi": result.mmi,
                "threat": result.threat_level,
                "casualties": result.casualties.total_affected,
                "fire_risk": result.fire_ignition_probability,
                "deploy": "drone" if i == 0 else "rover",
                "label": f"MMI {result.mmi:.1f} — {result.threat_level}"
            }

        return zones


if __name__ == "__main__":
    model = HAZUSModel()

    print("🏚️  FEMA HAZUS EARTHQUAKE DAMAGE MODEL")
    print("Fragility curves + casualty estimation\n")

    scenarios = [
        ("2023 Morocco M6.8 — Village near epicenter",
         EarthquakeEvent(6.8, 18.5, 31.12, -8.38, "strike_slip"),
         31.20, -8.30, 180, 5000),

        ("2021 Kermadec M8.1 — New Zealand North Island",
         EarthquakeEvent(8.1, 10.0, -29.72, -177.28, "reverse"),
         -38.0, 176.0, 360, 50000),

        ("Hypothetical Dallas M6.5 — Downtown",
        EarthquakeEvent(6.5, 15.0, 32.78, -96.80, "strike_slip"),
        32.82, -96.80, 360, 100000),
    ]

    for name, event, lat, lon, vs30, pop in scenarios:
        print(f"\n{'='*58}")
        print(f"SCENARIO: {name}")
        print(f"{'='*58}")
        result = model.calculate(event, lat, lon, vs30, pop)
        print(result.summary())