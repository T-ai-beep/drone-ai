import math
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

class SubjectCategory(Enum):
    """
    ISRID subject categories with behavioral profiles.
    Based on International Search and Rescue Incident Database.
    """
    HIKER           = "hiker"
    CHILD_1_3       = "child_1_3"
    CHILD_4_6       = "child_4_6"
    CHILD_7_9       = "child_7_9"
    CHILD_10_12     = "child_10_12"
    CHILD_13_15     = "child_13_15"
    DEMENTIA        = "dementia"
    DESPONDENT      = "despondent"
    HUNTER          = "hunter"
    CLIMBER         = "climber"
    SKIER           = "skier"
    MOUNTAIN_BIKER  = "mountain_biker"
    TRAIL_RUNNER    = "trail_runner"
    HORSEBACK       = "horseback"

@dataclass
class SubjectProfile:
    """
    Behavioral and mobility profile for a subject category.
    Based on Koester 2008 — Lost Person Behavior.
    """
    category: SubjectCategory
    travel_speed_mph: float      # typical travel speed
    max_distance_km: float       # 95th percentile max distance from LKP
    trail_tendency: float        # probability of staying on trail (0-1)
    downhill_tendency: float     # probability of moving downhill (0-1)
    water_tendency: float        # probability of moving toward water (0-1)
    shelter_tendency: float      # probability of seeking shelter (0-1)
    description: str

# ISRID behavioral profiles
SUBJECT_PROFILES = {
    SubjectCategory.HIKER: SubjectProfile(
        SubjectCategory.HIKER, 1.5, 8.0, 0.79, 0.65, 0.45, 0.30,
        "Hikers stay on trails, move downhill toward water"
    ),
    SubjectCategory.CHILD_1_3: SubjectProfile(
        SubjectCategory.CHILD_1_3, 0.5, 0.5, 0.10, 0.50, 0.70, 0.80,
        "Very young children move randomly, seek hiding spots"
    ),
    SubjectCategory.CHILD_4_6: SubjectProfile(
        SubjectCategory.CHILD_4_6, 0.8, 1.2, 0.15, 0.55, 0.65, 0.75,
        "Young children attracted to water, hide in small spaces"
    ),
    SubjectCategory.CHILD_7_9: SubjectProfile(
        SubjectCategory.CHILD_7_9, 1.0, 2.5, 0.25, 0.60, 0.60, 0.60,
        "Children begin following trails, attracted to water"
    ),
    SubjectCategory.CHILD_10_12: SubjectProfile(
        SubjectCategory.CHILD_10_12, 1.2, 4.0, 0.40, 0.62, 0.55, 0.50,
        "Pre-teen hikers, moderate trail following"
    ),
    SubjectCategory.DEMENTIA: SubjectProfile(
        SubjectCategory.DEMENTIA, 0.8, 3.5, 0.20, 0.45, 0.40, 0.25,
        "Wander randomly, avoid trails, may hide"
    ),
    SubjectCategory.DESPONDENT: SubjectProfile(
        SubjectCategory.DESPONDENT, 1.2, 5.5, 0.35, 0.50, 0.55, 0.20,
        "Move away from people, avoid trails, seek isolation"
    ),
    SubjectCategory.HUNTER: SubjectProfile(
        SubjectCategory.HUNTER, 1.5, 6.5, 0.50, 0.60, 0.65, 0.55,
        "Leave trails, move toward game habitat and water"
    ),
    SubjectCategory.CLIMBER: SubjectProfile(
        SubjectCategory.CLIMBER, 1.0, 5.0, 0.45, 0.35, 0.30, 0.65,
        "May go uphill, seek cliff faces and technical terrain"
    ),
    SubjectCategory.SKIER: SubjectProfile(
        SubjectCategory.SKIER, 2.0, 7.0, 0.60, 0.80, 0.40, 0.50,
        "Follow fall lines downhill, seek open terrain"
    ),
    SubjectCategory.TRAIL_RUNNER: SubjectProfile(
        SubjectCategory.TRAIL_RUNNER, 4.0, 15.0, 0.85, 0.55, 0.40, 0.20,
        "Stay on trails, cover large distances quickly"
    ),
    SubjectCategory.MOUNTAIN_BIKER: SubjectProfile(
        SubjectCategory.MOUNTAIN_BIKER, 5.0, 20.0, 0.90, 0.60, 0.35, 0.20,
        "Stay on trails, cover very large distances"
    ),
    SubjectCategory.HORSEBACK: SubjectProfile(
        SubjectCategory.HORSEBACK, 3.0, 12.0, 0.75, 0.55, 0.50, 0.40,
        "Follow trails and open terrain"
    ),
}

@dataclass
class SearchArea:
    """Defines a search zone relative to Last Known Position"""
    zone_name: str
    min_distance_km: float
    max_distance_km: float
    probability: float           # probability subject is in this zone
    priority: int                # 1 = highest priority
    bearing_range: Tuple[float, float]  # degrees from LKP center

@dataclass
class MattsonResult:
    """
    Output of Mattson probability model.
    Defines search zones in priority order.
    """
    subject_category: str
    lkp_lat: float
    lkp_lon: float
    hours_missing: float
    search_areas: List[SearchArea]
    total_probability: float
    recommended_deployment: str

    def summary(self):
        lines = [
            f"Subject:        {self.subject_category}",
            f"LKP:            ({self.lkp_lat}, {self.lkp_lon})",
            f"Hours Missing:  {self.hours_missing}h",
            f"Total Coverage: {self.total_probability*100:.0f}%",
            f"",
            f"SEARCH ZONES (priority order):"
        ]
        for zone in sorted(self.search_areas, key=lambda z: z.priority):
            lines.append(
                f"  Zone {zone.priority} [{zone.zone_name}]: "
                f"{zone.min_distance_km:.1f}-{zone.max_distance_km:.1f}km from LKP — "
                f"{zone.probability*100:.0f}% probability"
            )
        lines.append(f"\nDeploy: {self.recommended_deployment}")
        return "\n".join(lines)

class MattsonModel:
    """
    Mattson Lost Person Probability Model.
    
    Based on:
    - Mattson, E.L. 1980. Probability Modeling for Search and Rescue.
    - Koester, R.J. 2008. Lost Person Behavior. dbS Productions.
    - ISRID International Search and Rescue Incident Database.
    
    Calculates probability distribution of survivor location
    based on subject type, terrain, and time since last known position.
    Used by real SAR teams worldwide.
    """

    def _distance_probability(
        self,
        distance_km: float,
        profile: SubjectProfile,
        hours: float
    ) -> float:
        """
        Calculate probability of subject being at given distance from LKP.
        Uses negative exponential decay modified by subject mobility.
        """
        # Maximum expected travel distance based on time and speed
        max_travel = profile.travel_speed_mph * hours * 1.609  # convert to km
        max_travel = min(max_travel, profile.max_distance_km)

        if max_travel < 0.01:
            max_travel = 0.1

        # Exponential decay from LKP
        # Lambda controls decay rate — higher = closer to LKP
        decay_lambda = 2.0 / max_travel

        prob = decay_lambda * math.exp(-decay_lambda * distance_km)
        return prob

    def _terrain_modifier(
        self,
        profile: SubjectProfile,
        terrain_type: str = "forest"
    ) -> dict:
        """
        Adjust search zone probabilities based on terrain type.
        """
        modifiers = {
            "forest": {
                "trail": profile.trail_tendency,
                "off_trail": 1.0 - profile.trail_tendency,
                "water": profile.water_tendency,
                "high_ground": 1.0 - profile.downhill_tendency,
            },
            "mountain": {
                "trail": profile.trail_tendency * 0.8,
                "off_trail": (1.0 - profile.trail_tendency) * 1.2,
                "water": profile.water_tendency * 0.7,
                "high_ground": 1.0 - profile.downhill_tendency * 0.8,
            },
            "desert": {
                "trail": profile.trail_tendency * 0.6,
                "off_trail": (1.0 - profile.trail_tendency) * 1.4,
                "water": profile.water_tendency * 1.5,  # water highly attractive in desert
                "high_ground": 1.0 - profile.downhill_tendency,
            },
            "urban_fringe": {
                "trail": profile.trail_tendency * 1.2,
                "off_trail": (1.0 - profile.trail_tendency) * 0.8,
                "water": profile.water_tendency * 0.8,
                "high_ground": 1.0 - profile.downhill_tendency,
            }
        }
        return modifiers.get(terrain_type, modifiers["forest"])

    def calculate(
        self,
        lkp_lat: float,
        lkp_lon: float,
        subject_category: SubjectCategory,
        hours_missing: float,
        terrain_type: str = "forest"
    ) -> MattsonResult:
        """
        Calculate probability distribution of survivor location.
        
        Args:
            lkp_lat, lkp_lon: Last Known Position coordinates
            subject_category: Type of subject (hiker, child, dementia, etc.)
            hours_missing: Hours since last confirmed sighting
            terrain_type: forest/mountain/desert/urban_fringe
            
        Returns:
            MattsonResult with prioritized search zones
        """
        profile = SUBJECT_PROFILES[subject_category]
        terrain = self._terrain_modifier(profile, terrain_type)

        # Calculate max expected distance
        max_travel = min(
            profile.travel_speed_mph * hours_missing * 1.609,
            profile.max_distance_km
        )

        # Define search zones by distance rings
        zones = []

        # Zone 1: Immediate area (0-25% of max travel)
        d1_max = max(max_travel * 0.25, 0.3)
        p1 = self._distance_probability(d1_max / 2, profile, hours_missing)
        p1 *= (1.0 + profile.trail_tendency * 0.5)
        zones.append(SearchArea(
            zone_name="IMMEDIATE",
            min_distance_km=0.0,
            max_distance_km=d1_max,
            probability=min(p1 * d1_max, 0.55),
            priority=1,
            bearing_range=(0, 360)
        ))

        # Zone 2: High probability (25-60% of max travel)
        d2_min = d1_max
        d2_max = max(max_travel * 0.60, 0.8)
        p2 = self._distance_probability((d2_min + d2_max) / 2, profile, hours_missing)
        p2 *= terrain["trail"] + terrain["water"] * 0.5
        zones.append(SearchArea(
            zone_name="HIGH PROBABILITY",
            min_distance_km=d2_min,
            max_distance_km=d2_max,
            probability=min(p2 * (d2_max - d2_min), 0.35),
            priority=2,
            bearing_range=(0, 360)
        ))

        # Zone 3: Extended (60-100% of max travel)
        d3_min = d2_max
        d3_max = max_travel
        p3 = self._distance_probability(d3_max * 0.8, profile, hours_missing)
        zones.append(SearchArea(
            zone_name="EXTENDED",
            min_distance_km=d3_min,
            max_distance_km=max(d3_max, d3_min + 0.5),
            probability=min(p3 * (d3_max - d3_min + 0.1), 0.15),
            priority=3,
            bearing_range=(0, 360)
        ))

        # Zone 4: Containment boundary (beyond max travel)
        zones.append(SearchArea(
            zone_name="CONTAINMENT",
            min_distance_km=d3_max,
            max_distance_km=d3_max * 1.5,
            probability=0.05,
            priority=4,
            bearing_range=(0, 360)
        ))

        total_prob = sum(z.probability for z in zones)

        # Deployment recommendation
        if subject_category in [SubjectCategory.CHILD_1_3, SubjectCategory.CHILD_4_6]:
            deploy = "Thermal drone sweep of immediate area — children hide, look for heat signature in small spaces"
        elif subject_category == SubjectCategory.DEMENTIA:
            deploy = "Ground rover + thermal drone — dementia patients avoid trails, scan off-trail areas systematically"
        elif subject_category in [SubjectCategory.TRAIL_RUNNER, SubjectCategory.MOUNTAIN_BIKER]:
            deploy = "Extended aerial sweep along trail network — high mobility subjects cover large distances"
        elif subject_category == SubjectCategory.DESPONDENT:
            deploy = "Thermal drone — seek isolated areas, water bodies, dense vegetation"
        else:
            deploy = "Thermal drone immediate zone first, rover on trails, extended sweep if not found"

        return MattsonResult(
            subject_category=subject_category.value,
            lkp_lat=lkp_lat,
            lkp_lon=lkp_lon,
            hours_missing=hours_missing,
            search_areas=zones,
            total_probability=min(total_prob, 1.0),
            recommended_deployment=deploy
        )

    def beacon_priority_zones(self, result: MattsonResult) -> dict:
        """Convert Mattson result to Beacon priority zone format."""
        zones = {}
        color_map = {1: "red", 2: "orange", 3: "yellow", 4: "white"}

        for area in result.search_areas:
            color = color_map.get(area.priority, "white")
            zones[color] = {
                "priority": area.priority,
                "label": f"SAR {area.zone_name}",
                "min_km": area.min_distance_km,
                "max_km": area.max_distance_km,
                "probability": area.probability,
                "center": {"lat": result.lkp_lat, "lon": result.lkp_lon},
                "deploy": "drone" if area.priority <= 2 else "rover"
            }
        return zones


if __name__ == "__main__":
    model = MattsonModel()

    print("🔍 MATTSON LOST PERSON PROBABILITY MODEL")
    print("Based on ISRID + Koester 2008 Lost Person Behavior\n")

    scenarios = [
        ("Lost Hiker — Yosemite", 37.74, -119.59, SubjectCategory.HIKER, 6, "mountain"),
        ("Missing Child (age 5) — Allen TX Park", 33.10, -96.67, SubjectCategory.CHILD_4_6, 2, "urban_fringe"),
        ("Dementia Patient — Suburban Dallas", 32.90, -96.75, SubjectCategory.DEMENTIA, 4, "urban_fringe"),
        ("Trail Runner — Lost Creek Wilderness", 38.05, -106.45, SubjectCategory.TRAIL_RUNNER, 3, "forest"),
    ]

    for name, lat, lon, category, hours, terrain in scenarios:
        print(f"\n{'='*55}")
        print(f"SCENARIO: {name}")
        print(f"{'='*55}")
        result = model.calculate(lat, lon, category, hours, terrain)
        print(result.summary())