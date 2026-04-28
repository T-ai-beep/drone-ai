import math
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class ChemicalClass(Enum):
    TOXIC_GAS = "toxic_gas"
    FLAMMABLE_GAS = "flammable_gas"
    TOXIC_LIQUID = "toxic_liquid"
    FLAMMABLE_LIQUID = "flammable_liquid"
    EXPLOSIVE = "explosive"


class ReleaseType(Enum):
    INSTANTANEOUS = "instantaneous"
    CONTINUOUS = "continuous"
    EVAPORATING_POOL = "evaporating_pool"


@dataclass
class Chemical:
    name: str
    cas_number: str
    chemical_class: ChemicalClass
    molecular_weight: float
    boiling_point_c: float
    vapor_pressure_kpa: float
    idlh_ppm: float
    erpg2_ppm: float
    erpg3_ppm: float
    lfl_percent: float
    ufl_percent: float
    density_ratio: float
    decay_rate_per_s: float   # atmospheric decay rate 1/s


CHEMICALS = {
    "chlorine": Chemical(
        "Chlorine", "7782-50-5", ChemicalClass.TOXIC_GAS,
        70.9, -34.0, 674.0, 10, 3, 20, 0.0, 0.0, 2.47,
        0.0005   # reacts with moisture, ~20min half life outdoors
    ),
    "ammonia": Chemical(
        "Ammonia", "7664-41-7", ChemicalClass.TOXIC_GAS,
        17.0, -33.4, 1003.0, 300, 200, 1000, 15.0, 28.0, 0.59,
        0.0001   # relatively stable
    ),
    "hydrogen_sulfide": Chemical(
        "Hydrogen Sulfide", "7783-06-4", ChemicalClass.TOXIC_GAS,
        34.1, -60.3, 1880.0, 50, 27, 50, 4.0, 44.0, 1.19,
        0.0002
    ),
    "sulfur_dioxide": Chemical(
        "Sulfur Dioxide", "7446-09-5", ChemicalClass.TOXIC_GAS,
        64.1, -10.0, 325.0, 100, 0.3, 3.0, 0.0, 0.0, 2.26,
        0.0003
    ),
    "propane": Chemical(
        "Propane", "74-98-6", ChemicalClass.FLAMMABLE_GAS,
        44.1, -42.1, 853.0, 2100, 2100, 2100, 2.1, 9.5, 1.52,
        0.00001
    ),
    "methane": Chemical(
        "Methane", "74-82-8", ChemicalClass.FLAMMABLE_GAS,
        16.0, -161.5, 24800.0, 5000, 5000, 5000, 5.0, 15.0, 0.55,
        0.00001
    ),
    "hydrofluoric_acid": Chemical(
        "Hydrofluoric Acid", "7664-39-3", ChemicalClass.TOXIC_GAS,
        20.0, 19.5, 122.0, 30, 20, 50, 0.0, 0.0, 0.69,
        0.0004
    ),
    "benzene": Chemical(
        "Benzene", "71-43-2", ChemicalClass.FLAMMABLE_LIQUID,
        78.1, 80.1, 12.7, 500, 150, 1000, 1.2, 7.8, 2.77,
        0.00005
    ),
}


@dataclass
class ReleaseScenario:
    chemical_name: str
    release_type: ReleaseType
    quantity_kg: float
    release_rate_kg_s: float
    duration_min: float
    release_height_m: float
    source_lat: float
    source_lon: float


@dataclass
class AtmosphericConditions:
    wind_speed_ms: float
    wind_direction: float
    stability_class: str
    temperature_c: float
    humidity: float
    mixing_height_m: float


@dataclass
class PlumeThreatZone:
    zone_name: str
    chemical: str
    concentration_ppm: float
    guideline: str
    distance_m: float
    width_m: float
    threat_level: str
    health_effects: str
    affected_area_m2: float


@dataclass
class ALOHAResult:
    scenario: ReleaseScenario
    atmospheric: AtmosphericConditions
    threat_zones: List[PlumeThreatZone]
    max_downwind_distance_m: float
    wind_direction_deg: float
    flammable: bool
    fire_risk: bool
    explosion_risk: bool
    evacuation_radius_m: float
    beacon_deploy: str

    def summary(self):
        lines = [
            f"Chemical:      {self.scenario.chemical_name}",
            f"Released:      {self.scenario.quantity_kg:.0f}kg",
            f"Wind:          {self.atmospheric.wind_speed_ms*2.237:.1f}mph "
            f"@ {self.atmospheric.wind_direction:.0f}°",
            f"Stability:     Class {self.atmospheric.stability_class}",
            f"",
            f"THREAT ZONES (downwind):"
        ]
        for zone in self.threat_zones:
            lines.append(
                f"  [{zone.threat_level.upper():8s}] {zone.zone_name}: "
                f"{zone.distance_m:.0f}m downwind, "
                f"{zone.width_m:.0f}m wide — {zone.guideline}"
            )
        lines.extend([
            f"",
            f"Max Hazard Distance: {self.max_downwind_distance_m:.0f}m",
            f"Evacuation Radius:   {self.evacuation_radius_m:.0f}m",
            f"Flammable Risk:      {'YES' if self.fire_risk else 'NO'}",
            f"Explosion Risk:      {'YES' if self.explosion_risk else 'NO'}",
            f"Deploy:              {self.beacon_deploy.upper()}"
        ])
        return "\n".join(lines)


class ALOHAModel:
    """
    ALOHA - Areal Locations of Hazardous Atmospheres.
    NOAA/EPA chemical plume dispersion model.
    Gaussian plume with atmospheric decay.
    """

    PG_COEFFS = {
        "A": {"ay": 0.3658, "by": 0.9024, "az": 0.192, "bz": 0.936},
        "B": {"ay": 0.2751, "by": 0.9024, "az": 0.156, "bz": 0.922},
        "C": {"ay": 0.2090, "by": 0.9024, "az": 0.116, "bz": 0.905},
        "D": {"ay": 0.1471, "by": 0.9024, "az": 0.079, "bz": 0.881},
        "E": {"ay": 0.1046, "by": 0.9024, "az": 0.063, "bz": 0.871},
        "F": {"ay": 0.0722, "by": 0.9024, "az": 0.053, "bz": 0.814},
    }

    def _sigma_y(self, x_m: float, stability: str) -> float:
        c = self.PG_COEFFS.get(stability, self.PG_COEFFS["D"])
        return c["ay"] * (x_m ** c["by"])

    def _sigma_z(self, x_m: float, stability: str, mixing_height: float) -> float:
        c = self.PG_COEFFS.get(stability, self.PG_COEFFS["D"])
        return min(c["az"] * (x_m ** c["bz"]), mixing_height * 0.8)

    def _centerline_concentration(
        self,
        Q_gs: float,
        x_m: float,
        release_height: float,
        wind_ms: float,
        stability: str,
        mixing_height: float,
        decay_rate: float
    ) -> float:
        """Centerline ground-level concentration in g/m3 with decay."""
        if x_m <= 0 or wind_ms <= 0:
            return 0.0

        sy = self._sigma_y(x_m, stability)
        sz = self._sigma_z(x_m, stability, mixing_height)

        if sy <= 0 or sz <= 0:
            return 0.0

        # Travel time for decay
        travel_time = x_m / wind_ms
        decay = math.exp(-decay_rate * travel_time)

        # Ground level centerline with ground reflection
        exp_z = math.exp(-0.5 * (release_height / sz) ** 2)

        C = (Q_gs / (math.pi * sy * sz * wind_ms)) * exp_z * decay

        return max(0.0, C)

    def _gm3_to_ppm(self, C_gm3: float, chemical: Chemical, temp_c: float) -> float:
        """Convert g/m3 to ppm."""
        if C_gm3 <= 0:
            return 0.0
        T_K = temp_c + 273.15
        return (C_gm3 * 24450.0 * T_K / 298.0) / chemical.molecular_weight

    def _ppm_to_gm3(self, ppm: float, chemical: Chemical, temp_c: float) -> float:
        """Convert ppm to g/m3."""
        if ppm <= 0:
            return 0.0
        T_K = temp_c + 273.15
        return (ppm * chemical.molecular_weight) / (24450.0 * T_K / 298.0)

    def _find_hazard_distance(
        self,
        Q_gs: float,
        target_ppm: float,
        chemical: Chemical,
        atm: AtmosphericConditions,
        release_height: float
    ) -> float:
        """Find downwind distance where concentration drops to target_ppm."""
        if target_ppm <= 0:
            return 0.0

        target_gm3 = self._ppm_to_gm3(target_ppm, chemical, atm.temperature_c)
        u = max(atm.wind_speed_ms, 0.5)

        # Check if we ever reach target concentration
        C_near = self._centerline_concentration(
            Q_gs, 10.0, release_height, u,
            atm.stability_class, atm.mixing_height_m, chemical.decay_rate_per_s
        )

        if C_near < target_gm3:
            return 0.0  # never reaches this concentration

        # Scan to find crossover point
        distances = [
            10, 25, 50, 100, 200, 300, 500, 750,
            1000, 1500, 2000, 3000, 5000, 7500,
            10000, 15000, 20000, 30000, 50000
        ]

        prev_C = C_near
        prev_x = 10.0

        for x in distances[1:]:
            C = self._centerline_concentration(
                Q_gs, x, release_height, u,
                atm.stability_class, atm.mixing_height_m, chemical.decay_rate_per_s
            )

            if prev_C >= target_gm3 and C < target_gm3:
                # Interpolate
                frac = (prev_C - target_gm3) / max(prev_C - C, 1e-30)
                return prev_x + frac * (x - prev_x)

            prev_C = C
            prev_x = x

        # If still above target at max distance
        if prev_C >= target_gm3:
            return distances[-1]

        return 0.0

    def _zone_width(self, distance_m: float, stability: str) -> float:
        sy = self._sigma_y(distance_m, stability)
        return 2 * 2.15 * sy

    def calculate(
        self,
        scenario: ReleaseScenario,
        atm: AtmosphericConditions
    ) -> ALOHAResult:

        chemical = CHEMICALS.get(scenario.chemical_name)
        if not chemical:
            raise ValueError(f"Unknown chemical: {scenario.chemical_name}")

        # Source strength in g/s
        if scenario.release_type == ReleaseType.INSTANTANEOUS:
            duration_s = max(scenario.duration_min * 60, 60)
            Q_gs = scenario.quantity_kg * 1000 / duration_s
        else:
            Q_gs = scenario.release_rate_kg_s * 1000

        u = max(atm.wind_speed_ms, 0.5)
        threat_zones = []

        # ERPG-3
        if chemical.erpg3_ppm > 0:
            d3 = self._find_hazard_distance(
                Q_gs, chemical.erpg3_ppm, chemical, atm, scenario.release_height_m
            )
            if d3 > 0:
                threat_zones.append(PlumeThreatZone(
                    zone_name="ERPG-3 Zone",
                    chemical=chemical.name,
                    concentration_ppm=chemical.erpg3_ppm,
                    guideline="ERPG-3",
                    distance_m=d3,
                    width_m=self._zone_width(d3, atm.stability_class),
                    threat_level="critical",
                    health_effects="Life threatening — immediate evacuation required",
                    affected_area_m2=0.5 * d3 * self._zone_width(d3, atm.stability_class)
                ))

        # ERPG-2
        if chemical.erpg2_ppm > 0 and chemical.erpg2_ppm != chemical.erpg3_ppm:
            d2 = self._find_hazard_distance(
                Q_gs, chemical.erpg2_ppm, chemical, atm, scenario.release_height_m
            )
            if d2 > 0:
                threat_zones.append(PlumeThreatZone(
                    zone_name="ERPG-2 Zone",
                    chemical=chemical.name,
                    concentration_ppm=chemical.erpg2_ppm,
                    guideline="ERPG-2",
                    distance_m=d2,
                    width_m=self._zone_width(d2, atm.stability_class),
                    threat_level="high",
                    health_effects="Irreversible health effects — shelter or evacuate",
                    affected_area_m2=0.5 * d2 * self._zone_width(d2, atm.stability_class)
                ))

        # IDLH
        if chemical.idlh_ppm > 0:
            d_idlh = self._find_hazard_distance(
                Q_gs, chemical.idlh_ppm, chemical, atm, scenario.release_height_m
            )
            if d_idlh > 0:
                threat_zones.append(PlumeThreatZone(
                    zone_name="IDLH Zone",
                    chemical=chemical.name,
                    concentration_ppm=chemical.idlh_ppm,
                    guideline="IDLH",
                    distance_m=d_idlh,
                    width_m=self._zone_width(d_idlh, atm.stability_class),
                    threat_level="medium",
                    health_effects="Dangerous without respiratory protection",
                    affected_area_m2=0.5 * d_idlh * self._zone_width(d_idlh, atm.stability_class)
                ))

        max_dist = max((z.distance_m for z in threat_zones), default=100.0)
        flammable = chemical.lfl_percent > 0
        fire_risk = flammable and scenario.quantity_kg > 10
        explosion_risk = flammable and chemical.vapor_pressure_kpa > 100
        evac_radius = max(max_dist * 1.2, 500.0)

        if chemical.density_ratio > 1.0:
            deploy = "drone — heavy gas stays near ground, aerial thermal detection optimal"
        else:
            deploy = "drone — lighter than air, aerial monitoring required"

        return ALOHAResult(
            scenario=scenario,
            atmospheric=atm,
            threat_zones=sorted(threat_zones, key=lambda z: z.distance_m, reverse=True),
            max_downwind_distance_m=max_dist,
            wind_direction_deg=atm.wind_direction,
            flammable=flammable,
            fire_risk=fire_risk,
            explosion_risk=explosion_risk,
            evacuation_radius_m=evac_radius,
            beacon_deploy=deploy
        )

    def beacon_priority_zones(self, result: ALOHAResult) -> dict:
        zones = {}
        color_map = {"critical": "red", "high": "orange", "medium": "yellow"}
        for zone in result.threat_zones:
            color = color_map.get(zone.threat_level, "yellow")
            zones[color] = {
                "priority": {"red": 1, "orange": 2, "yellow": 3}[color],
                "distance_m": zone.distance_m,
                "width_m": zone.width_m,
                "guideline": zone.guideline,
                "health_effects": zone.health_effects,
                "deploy": "drone",
                "label": f"{zone.zone_name} — {zone.distance_m:.0f}m downwind"
            }
        return zones


if __name__ == "__main__":
    model = ALOHAModel()

    print("☣️  ALOHA CHEMICAL PLUME DISPERSION MODEL")
    print("NOAA/EPA Areal Locations of Hazardous Atmospheres\n")

    scenarios = [
        (
            "Texas City — Chlorine Tank Rupture",
            ReleaseScenario(
                "chlorine", ReleaseType.INSTANTANEOUS,
                5000, 83.3, 60, 1.0, 29.38, -94.90
            ),
            AtmosphericConditions(3.0, 270, "D", 25, 0.65, 1000)
        ),
        (
            "Dallas Refinery — Ammonia Pipeline Leak",
            ReleaseScenario(
                "ammonia", ReleaseType.CONTINUOUS,
                2000, 5.0, 400, 2.0, 32.78, -96.80
            ),
            AtmosphericConditions(5.0, 180, "C", 30, 0.55, 1500)
        ),
        (
            "Houston Ship Channel — H2S Release",
            ReleaseScenario(
                "hydrogen_sulfide", ReleaseType.CONTINUOUS,
                500, 2.0, 250, 3.0, 29.75, -95.15
            ),
            AtmosphericConditions(2.0, 315, "F", 22, 0.70, 500)
        ),
    ]

    for name, scenario, atm in scenarios:
        print(f"\n{'='*58}")
        print(f"SCENARIO: {name}")
        print(f"{'='*58}")
        result = model.calculate(scenario, atm)
        print(result.summary())