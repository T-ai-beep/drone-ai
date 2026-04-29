import math
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class HurricaneCategory(Enum):
    TROPICAL_STORM = 0
    CAT1 = 1
    CAT2 = 2
    CAT3 = 3
    CAT4 = 4
    CAT5 = 5

@dataclass
class Hurricane:
    name: str
    category: HurricaneCategory
    wind_speed_mph: float
    central_pressure_mb: float
    radius_max_wind_km: float
    forward_speed_mph: float
    heading_deg: float
    landfall_lat: float
    landfall_lon: float

@dataclass
class CoastalPoint:
    name: str
    lat: float
    lon: float
    distance_from_landfall_km: float
    coastal_bathymetry_m: float  # offshore depth
    inland_elevation_m: float
    population: int

@dataclass
class SLOSHResult:
    point: CoastalPoint
    hurricane: Hurricane
    surge_height_m: float
    inundation_depth_m: float
    inundation_distance_m: float
    arrival_time_min: float
    threat_level: str
    people_at_risk: int
    evacuation_zone: str

    def summary(self):
        return (
            f"Surge Height:      {self.surge_height_m:.1f}m\n"
            f"Inundation Depth:  {self.inundation_depth_m:.1f}m\n"
            f"Inundation Dist:   {self.inundation_distance_m:.0f}m inland\n"
            f"Arrival Time:      {self.arrival_time_min:.0f}min before landfall\n"
            f"People at Risk:    {self.people_at_risk:,}\n"
            f"Evacuation Zone:   {self.evacuation_zone}\n"
            f"Threat Level:      {self.threat_level.upper()}"
        )

# Saffir-Simpson surge ranges (meters)
CATEGORY_SURGE = {
    HurricaneCategory.TROPICAL_STORM: (0.3, 0.9),
    HurricaneCategory.CAT1: (1.2, 1.5),
    HurricaneCategory.CAT2: (1.8, 2.4),
    HurricaneCategory.CAT3: (2.7, 3.7),
    HurricaneCategory.CAT4: (4.0, 5.5),
    HurricaneCategory.CAT5: (5.5, 8.5),
}

class SLOSHModel:
    """
    NOAA SLOSH — Sea, Lake, and Overland Surges from Hurricanes.
    
    Reference: Jelesnianski, C.P., Chen, J., and Shaffer, W.A. 1992.
    SLOSH: Sea, Lake, and Overland Surges from Hurricanes.
    NOAA Technical Report NWS 48.
    
    Simplified parametric surge model based on SLOSH methodology.
    Used by NOAA and NHC for hurricane evacuation planning.
    """

    def _pressure_deficit(self, hurricane: Hurricane) -> float:
        """Pressure deficit from ambient (1013mb)."""
        return max(0, 1013 - hurricane.central_pressure_mb)

    def _surge_at_landfall(self, hurricane: Hurricane) -> float:
        """
        Maximum surge at landfall using SLOSH parametric formula.
        Based on Irish et al. 2008 simplified surge scaling.
        """
        delta_p = self._pressure_deficit(hurricane)

        # Base surge from pressure deficit
        surge_pressure = 0.0155 * delta_p

        # Wind contribution
        surge_wind = (hurricane.wind_speed_mph / 100) ** 2 * 3.0

        # Forward speed contribution
        surge_forward = hurricane.forward_speed_mph * 0.02

        # Radius of max wind effect
        rmw_factor = math.exp(-hurricane.radius_max_wind_km / 50)

        total_surge = (surge_pressure + surge_wind + surge_forward) * (1 + rmw_factor)

        # Cap by category
        cat = hurricane.category
        surge_range = CATEGORY_SURGE.get(cat, (0, 10))
        return max(surge_range[0], min(total_surge, surge_range[1] * 1.2))

    def _surge_at_distance(
        self,
        landfall_surge: float,
        distance_km: float,
        hurricane: Hurricane
    ) -> float:
        """
        Surge height at distance from landfall center.
        Surge decays exponentially with distance along coast.
        """
        # Right side of track has higher surge (Northern Hemisphere)
        rmw = hurricane.radius_max_wind_km
        
        if distance_km <= rmw:
            # Inside radius of max wind — near peak surge
            factor = 1.0 - 0.3 * (distance_km / rmw)
        else:
            # Outside — exponential decay
            factor = 0.7 * math.exp(-(distance_km - rmw) / (rmw * 2))

        return landfall_surge * factor

    def _inundation(
        self,
        surge_m: float,
        point: CoastalPoint
    ) -> tuple:
        """
        Calculate inundation depth and distance inland.
        Uses simple bathtub model with coastal slope.
        """
        # Effective surge above land elevation
        inundation_depth = max(0, surge_m - point.inland_elevation_m)

        if inundation_depth <= 0:
            return 0, 0

        # Coastal slope approximation
        # Flat coastal plains ~1:1000, steeper coasts ~1:100
        if point.inland_elevation_m < 2:
            slope = 1/1000  # very flat
        elif point.inland_elevation_m < 5:
            slope = 1/500
        else:
            slope = 1/200

        inundation_distance = inundation_depth / slope

        return inundation_depth, inundation_distance

    def _arrival_time(
        self,
        distance_km: float,
        hurricane: Hurricane
    ) -> float:
        """
        Estimate surge arrival time before landfall (minutes).
        Surge arrives before eye makes landfall.
        """
        # Surge wave speed ~ forward speed of hurricane
        wave_speed_kmh = hurricane.forward_speed_mph * 1.609
        
        # Pre-landfall surge arrives 2-6 hours early depending on size
        pre_landfall_hours = hurricane.radius_max_wind_km / wave_speed_kmh
        
        # Time for surge to travel to this coastal point
        travel_time_h = distance_km / max(wave_speed_kmh, 1)
        
        arrival_min = (pre_landfall_hours - travel_time_h) * 60
        return max(0, arrival_min)

    def _evacuation_zone(self, surge_m: float, inundation_m: float) -> str:
        """FEMA evacuation zone assignment."""
        if surge_m > 4.0 or inundation_m > 2.0:
            return "Zone A — Mandatory evacuation"
        elif surge_m > 2.5 or inundation_m > 1.0:
            return "Zone B — Evacuation recommended"
        elif surge_m > 1.5:
            return "Zone C — Voluntary evacuation"
        return "Zone D — Monitor conditions"

    def _threat_level(self, surge_m: float, inundation_m: float) -> str:
        if surge_m > 4.0 or inundation_m > 2.0:
            return "critical"
        elif surge_m > 2.0 or inundation_m > 1.0:
            return "high"
        elif surge_m > 1.0:
            return "medium"
        return "low"

    def calculate(
        self,
        hurricane: Hurricane,
        point: CoastalPoint
    ) -> SLOSHResult:
        """
        Calculate storm surge at a coastal point.
        
        Args:
            hurricane: Hurricane parameters
            point: Coastal location to assess
        """
        # Max surge at landfall
        landfall_surge = self._surge_at_landfall(hurricane)

        # Surge at this point
        surge = self._surge_at_distance(
            landfall_surge,
            point.distance_from_landfall_km,
            hurricane
        )

        # Inundation
        depth, distance = self._inundation(surge, point)

        # Arrival time
        arrival = self._arrival_time(
            point.distance_from_landfall_km,
            hurricane
        )

        # People at risk
        if distance > 0:
            risk_fraction = min(distance / 1000, 1.0)
            people = int(point.population * risk_fraction)
        else:
            people = 0

        threat = self._threat_level(surge, depth)
        evac_zone = self._evacuation_zone(surge, depth)

        return SLOSHResult(
            point=point,
            hurricane=hurricane,
            surge_height_m=surge,
            inundation_depth_m=depth,
            inundation_distance_m=distance,
            arrival_time_min=arrival,
            threat_level=threat,
            people_at_risk=people,
            evacuation_zone=evac_zone
        )

    def regional_assessment(
        self,
        hurricane: Hurricane,
        points: List[CoastalPoint]
    ) -> List[SLOSHResult]:
        """Assess surge at multiple coastal points."""
        results = []
        for point in points:
            result = self.calculate(hurricane, point)
            results.append(result)
        return sorted(results, key=lambda r: r.surge_height_m, reverse=True)

    def beacon_priority_zones(
        self,
        results: List[SLOSHResult]
    ) -> dict:
        zones = {}
        color_map = {"critical": "red", "high": "orange",
                    "medium": "yellow", "low": "yellow"}

        for result in results:
            color = color_map[result.threat_level]
            if color not in zones:
                zones[color] = {
                    "priority": {"red": 1, "orange": 2, "yellow": 3}[color],
                    "sites": [],
                    "max_surge_m": 0,
                    "deploy": "drone",
                    "label": f"Storm surge — {result.threat_level} threat"
                }
            zones[color]["sites"].append(result.point.name)
            zones[color]["max_surge_m"] = max(
                zones[color]["max_surge_m"],
                result.surge_height_m
            )

        return zones


if __name__ == "__main__":
    model = SLOSHModel()

    print("🌀 NOAA SLOSH STORM SURGE MODEL")
    print("Sea, Lake, and Overland Surges from Hurricanes\n")

    # Hurricane Harvey 2017 — Texas Gulf Coast
    harvey = Hurricane(
        name="Harvey", category=HurricaneCategory.CAT4,
        wind_speed_mph=130, central_pressure_mb=938,
        radius_max_wind_km=45, forward_speed_mph=10,
        heading_deg=330, landfall_lat=28.0, landfall_lon=-97.0
    )

    harvey_points = [
        CoastalPoint("Rockport TX", 28.02, -97.05, 5, 10, 3, 10000),
        CoastalPoint("Port Aransas TX", 27.83, -97.07, 25, 15, 2, 4000),
        CoastalPoint("Corpus Christi TX", 27.80, -97.40, 45, 20, 5, 320000),
        CoastalPoint("Houston Ship Channel", 29.75, -95.15, 120, 5, 8, 500000),
        CoastalPoint("Galveston TX", 29.30, -94.80, 150, 8, 2, 50000),
    ]

    # Hurricane Katrina 2005 — Louisiana
    katrina = Hurricane(
        name="Katrina", category=HurricaneCategory.CAT5,
        wind_speed_mph=175, central_pressure_mb=902,
        radius_max_wind_km=85, forward_speed_mph=15,
        heading_deg=0, landfall_lat=29.0, landfall_lon=-89.6
    )

    katrina_points = [
        CoastalPoint("New Orleans LA", 29.95, -90.07, 50, 5, -2, 500000),
        CoastalPoint("Biloxi MS", 30.40, -88.89, 80, 8, 3, 50000),
        CoastalPoint("Gulfport MS", 30.37, -89.09, 65, 7, 4, 75000),
        CoastalPoint("Bay St Louis MS", 30.31, -89.33, 30, 6, 2, 10000),
        CoastalPoint("Slidell LA", 30.28, -89.78, 40, 4, 3, 30000),
    ]

    for hurricane, points, name in [
        (harvey, harvey_points, "Hurricane Harvey 2017 — Texas"),
        (katrina, katrina_points, "Hurricane Katrina 2005 — Louisiana")
    ]:
        print(f"\n{'='*58}")
        print(f"SCENARIO: {name}")
        print(f"Category: {hurricane.category.name}, "
              f"{hurricane.wind_speed_mph}mph, {hurricane.central_pressure_mb}mb")
        print(f"{'='*58}")

        results = model.regional_assessment(hurricane, points)
        for r in results:
            print(f"\n📍 {r.point.name}")
            print(r.summary())