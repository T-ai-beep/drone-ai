import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class SlopeProperties:
    """Geotechnical properties of a slope"""
    name: str
    slope_angle: float        # degrees
    cohesion: float           # kPa - soil cohesion
    friction_angle: float     # degrees - internal friction angle
    unit_weight: float        # kN/m3 - soil unit weight
    saturated_weight: float   # kN/m3 - saturated soil unit weight
    hydraulic_conductivity: float  # m/s - saturated hydraulic conductivity
    diffusivity: float        # m2/s - hydraulic diffusivity
    depth: float              # m - soil depth to failure plane
    initial_water_table: float     # m - initial depth to water table

@dataclass 
class RainfallInput:
    """Rainfall intensity and duration"""
    intensity_mm_hr: float    # mm/hour rainfall intensity
    duration_hours: float     # hours of rainfall
    antecedent_moisture: float # 0-1, prior soil moisture condition

@dataclass
class TRIGRSResult:
    """Output from TRIGRS slope stability analysis"""
    slope_name: str
    factor_of_safety: float   # FS < 1.0 = failure
    failure_probability: float # 0-1
    failure_depth: float      # m - expected failure plane depth
    pore_pressure: float      # kPa - pore water pressure at failure plane
    threat_level: str
    failure_time_hours: float # estimated hours until failure
    volume_m3: float          # estimated failure volume
    runout_distance_m: float  # estimated runout distance

    def summary(self):
        stability = "UNSTABLE" if self.factor_of_safety < 1.0 else "STABLE"
        return (
            f"Factor of Safety:    {self.factor_of_safety:.3f} [{stability}]\n"
            f"Failure Probability: {self.failure_probability*100:.0f}%\n"
            f"Pore Pressure:       {self.pore_pressure:.1f}kPa\n"
            f"Failure Depth:       {self.failure_depth:.1f}m\n"
            f"Failure Volume:      {self.volume_m3:.0f}m³\n"
            f"Runout Distance:     {self.runout_distance_m:.0f}m\n"
            f"Time to Failure:     {self.failure_time_hours:.1f}h\n"
            f"Threat Level:        {self.threat_level.upper()}"
        )

# Standard slope profiles for different terrain types
SLOPE_PROFILES = {
    "coastal_cliff": SlopeProperties(
        "Coastal Cliff", 65, 5.0, 28, 18.0, 20.0, 1e-5, 1e-4, 3.0, 1.5
    ),
    "forest_hillside": SlopeProperties(
        "Forest Hillside", 30, 8.0, 32, 16.0, 18.5, 5e-6, 5e-5, 2.0, 1.0
    ),
    "mountain_slope": SlopeProperties(
        "Mountain Slope", 45, 3.0, 30, 17.0, 19.5, 2e-6, 2e-5, 4.0, 2.0
    ),
    "volcanic_ash": SlopeProperties(
        "Volcanic Ash Slope", 25, 2.0, 25, 12.0, 15.0, 1e-4, 1e-3, 1.5, 0.5
    ),
    "clay_hillside": SlopeProperties(
        "Clay Hillside", 20, 15.0, 18, 17.5, 19.0, 1e-8, 1e-7, 3.0, 1.5
    ),
    "urban_cut_slope": SlopeProperties(
        "Urban Cut Slope", 55, 10.0, 35, 18.5, 20.5, 1e-6, 1e-5, 2.5, 1.0
    ),
    "debris_fan": SlopeProperties(
        "Debris Fan", 15, 1.0, 22, 14.0, 17.0, 1e-3, 1e-2, 1.0, 0.3
    ),
}

class TRIGRSModel:
    """
    Transient Rainfall Infiltration and Grid-based Regional
    Slope-stability Model (TRIGRS).
    
    Reference: Baum, R.L., Savage, W.Z., and Godt, J.W. 2002.
    TRIGRS - A Fortran Program for Transient Rainfall Infiltration
    and Grid-Based Regional Slope-Stability Analysis.
    USGS Open-File Report 02-0424.
    
    Calculates Factor of Safety for infinite slope stability
    under transient rainfall infiltration conditions.
    Same model used by USGS for landslide hazard assessment.
    """

    def _infiltration_depth(
        self,
        slope: SlopeProperties,
        rainfall: RainfallInput,
        time_hours: float
    ) -> float:
        """
        Calculate wetting front depth using Green-Ampt infiltration.
        Returns depth of water infiltration in meters.
        """
        time_sec = time_hours * 3600
        intensity_ms = rainfall.intensity_mm_hr / (1000 * 3600)
        
        # Infiltration rate limited by hydraulic conductivity
        if intensity_ms <= slope.hydraulic_conductivity:
            # All rainfall infiltrates
            infiltration = intensity_ms * time_sec
        else:
            # Ponding occurs — use Green-Ampt
            suction = 0.5  # m capillary suction (approximate)
            initial_deficit = (1 - rainfall.antecedent_moisture) * 0.35
            
            if initial_deficit < 0.001:
                infiltration = slope.hydraulic_conductivity * time_sec
            else:
                # Green-Ampt with ponding
                F_approx = slope.hydraulic_conductivity * time_sec
                for _ in range(10):  # Newton iteration
                    F_new = (slope.hydraulic_conductivity * time_sec + 
                            suction * initial_deficit * 
                            math.log(1 + F_approx / (suction * initial_deficit)))
                    F_approx = F_new
                infiltration = F_approx

        return min(infiltration, slope.depth)

    def _pore_pressure(
        self,
        slope: SlopeProperties,
        rainfall: RainfallInput,
        time_hours: float,
        depth: float
    ) -> float:
        """
        Calculate pore water pressure at failure plane using
        Iverson 2000 linearized transient solution.
        Returns pressure head in meters.
        """
        time_sec = time_hours * 3600
        Iz = rainfall.intensity_mm_hr / (1000 * 3600)  # m/s
        D0 = slope.diffusivity
        Ks = slope.hydraulic_conductivity
        
        # Steady state pressure head
        beta = math.cos(math.radians(slope.slope_angle))**2
        
        # Initial water table depth
        d = slope.depth - slope.initial_water_table
        
        # Steady state component
        psi_steady = -(depth - d) * beta if depth < d else 0
        
        # Transient component (Iverson 2000 eq 27a)
        if D0 > 0 and time_sec > 0:
            alpha = math.sqrt(D0 * time_sec / math.pi)
            # Response function
            if alpha > 0:
                response = (Iz / Ks) * (
                    math.sqrt(4 * D0 * time_sec / math.pi) * 
                    math.exp(-depth**2 / (4 * D0 * time_sec)) -
                    depth * math.erfc(depth / (2 * math.sqrt(D0 * time_sec)))
                )
            else:
                response = 0
            psi_transient = response * beta
        else:
            psi_transient = 0

        # Total pressure head (m)
        psi_total = psi_steady + psi_transient
        
        # Convert to kPa
        gamma_w = 9.81  # kN/m3
        return max(0, psi_total) * gamma_w

    def _factor_of_safety(
        self,
        slope: SlopeProperties,
        pore_pressure_kpa: float,
        depth: float
    ) -> float:
        """
        Infinite slope stability equation.
        FS = (c' + (gamma*z*cos²a - u) * tan(phi')) / (gamma*z*sin(a)*cos(a))
        
        Standard geotechnical infinite slope model.
        """
        alpha = math.radians(slope.slope_angle)
        phi = math.radians(slope.friction_angle)
        
        gamma = slope.unit_weight
        z = max(depth, 0.1)
        
        # Normal stress on failure plane
        sigma_n = gamma * z * math.cos(alpha)**2
        
        # Pore pressure
        u = pore_pressure_kpa
        
        # Effective normal stress
        sigma_n_eff = max(sigma_n - u, 0)
        
        # Shear resistance (numerator)
        resistance = slope.cohesion + sigma_n_eff * math.tan(phi)
        
        # Driving stress (denominator)
        driving = gamma * z * math.sin(alpha) * math.cos(alpha)
        
        if driving <= 0:
            return 10.0  # flat slope, very stable
        
        return resistance / driving

    def _failure_probability(self, fs: float) -> float:
        """
        Convert Factor of Safety to failure probability.
        Uses lognormal reliability approach.
        COV of FS assumed 0.15 (standard geotechnical practice).
        """
        if fs <= 0:
            return 1.0
        
        cov = 0.15
        beta_r = math.log(fs) / math.sqrt(math.log(1 + cov**2))
        
        # Normal CDF approximation
        prob_failure = 0.5 * (1 - math.erf(beta_r / math.sqrt(2)))
        return max(0.0, min(1.0, prob_failure))

    def _failure_volume(
        self,
        slope: SlopeProperties,
        depth: float,
        width_m: float = 50
    ) -> float:
        """Estimate failure volume in m³."""
        length = depth / math.sin(math.radians(slope.slope_angle))
        return length * width_m * depth * 0.5

    def _runout_distance(
        self,
        slope: SlopeProperties,
        volume: float
    ) -> float:
        """
        Estimate runout distance using Fahrboschung angle method.
        Empirical relationship from Corominas 1996.
        """
        # Fahrboschung angle typically 11-45 degrees for landslides
        # Lower for larger volumes
        if volume < 100:
            fahrbung_angle = 35
        elif volume < 1000:
            fahrbung_angle = 25
        elif volume < 10000:
            fahrbung_angle = 18
        else:
            fahrbung_angle = 12
        
        # Vertical drop
        vertical_drop = slope.depth * math.sin(math.radians(slope.slope_angle))
        
        # Horizontal runout
        runout = vertical_drop / math.tan(math.radians(fahrbung_angle))
        return runout

    def _time_to_failure(
        self,
        slope: SlopeProperties,
        rainfall: RainfallInput,
        fs_current: float
    ) -> float:
        """Estimate hours until FS drops below 1.0."""
        if fs_current < 1.0:
            return 0.0
        
        # Binary search for failure time
        t_low, t_high = 0.0, 72.0
        
        for _ in range(20):
            t_mid = (t_low + t_high) / 2
            depth = self._infiltration_depth(slope, rainfall, t_mid)
            pore_p = self._pore_pressure(slope, rainfall, t_mid, depth)
            fs = self._factor_of_safety(slope, pore_p, depth)
            
            if fs < 1.0:
                t_high = t_mid
            else:
                t_low = t_mid
            
            if t_high - t_low < 0.1:
                break
        
        return t_high if t_high < 72.0 else float('inf')

    def _threat_from_fs(self, fs: float, prob: float) -> str:
        if fs < 1.0 or prob > 0.5:
            return "critical"
        elif fs < 1.2 or prob > 0.25:
            return "high"
        elif fs < 1.5 or prob > 0.10:
            return "medium"
        return "low"

    def calculate(
        self,
        slope: SlopeProperties,
        rainfall: RainfallInput,
        analysis_time_hours: float = 6.0
    ) -> TRIGRSResult:
        """
        Run TRIGRS slope stability analysis.
        
        Args:
            slope: Slope geotechnical properties
            rainfall: Rainfall intensity and duration
            analysis_time_hours: Time of analysis after rainfall starts
        """
        # Calculate infiltration depth
        depth = self._infiltration_depth(slope, rainfall, analysis_time_hours)
        failure_depth = min(depth + 0.5, slope.depth)

        # Calculate pore pressure at failure plane
        pore_p = self._pore_pressure(slope, rainfall, analysis_time_hours, failure_depth)

        # Calculate factor of safety
        fs = self._factor_of_safety(slope, pore_p, failure_depth)
        fs = max(0.1, fs)

        # Failure probability
        prob = self._failure_probability(fs)

        # Failure geometry
        volume = self._failure_volume(slope, failure_depth)
        runout = self._runout_distance(slope, volume)

        # Time to failure
        time_to_fail = self._time_to_failure(slope, rainfall, fs)

        # Threat level
        threat = self._threat_from_fs(fs, prob)

        return TRIGRSResult(
            slope_name=slope.name,
            factor_of_safety=fs,
            failure_probability=prob,
            failure_depth=failure_depth,
            pore_pressure=pore_p,
            threat_level=threat,
            failure_time_hours=time_to_fail,
            volume_m3=volume,
            runout_distance_m=runout
        )

    def regional_assessment(
        self,
        rainfall: RainfallInput,
        analysis_time_hours: float = 6.0
    ) -> List[TRIGRSResult]:
        """Assess all slope types for a given rainfall event."""
        results = []
        for profile in SLOPE_PROFILES.values():
            result = self.calculate(profile, rainfall, analysis_time_hours)
            results.append(result)
        return results

    def beacon_priority_zones(
        self,
        rainfall: RainfallInput,
        analysis_time_hours: float = 6.0
    ) -> dict:
        """Generate Beacon priority zones from TRIGRS output."""
        results = self.regional_assessment(rainfall, analysis_time_hours)
        
        critical = [r for r in results if r.threat_level == "critical"]
        high = [r for r in results if r.threat_level == "high"]
        medium = [r for r in results if r.threat_level == "medium"]

        zones = {}
        if critical:
            zones["red"] = {
                "priority": 1,
                "threat": "critical",
                "slopes": [r.slope_name for r in critical],
                "max_runout_m": max(r.runout_distance_m for r in critical),
                "deploy": "drone",
                "label": f"{len(critical)} slopes at critical failure risk"
            }
        if high:
            zones["orange"] = {
                "priority": 2,
                "threat": "high",
                "slopes": [r.slope_name for r in high],
                "deploy": "rover",
                "label": f"{len(high)} slopes at high failure risk"
            }
        if medium:
            zones["yellow"] = {
                "priority": 3,
                "threat": "medium",
                "slopes": [r.slope_name for r in medium],
                "deploy": "drone",
                "label": f"{len(medium)} slopes at medium failure risk"
            }
        return zones


if __name__ == "__main__":
    model = TRIGRSModel()

    print("⛰️  TRIGRS SLOPE STABILITY MODEL")
    print("USGS Transient Rainfall Infiltration and Grid-Based Analysis\n")

    scenarios = [
        ("Morocco Earthquake Aftermath — Heavy Rain on Disturbed Slopes",
         RainfallInput(40, 6, 0.8), 6),
        ("Australian Bushfire — Post-fire Debris Flow Risk",
         RainfallInput(25, 3, 0.3), 3),
        ("Pacific Northwest — Atmospheric River Event",
         RainfallInput(60, 12, 0.9), 12),
    ]

    for name, rainfall, hours in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"Rainfall: {rainfall.intensity_mm_hr}mm/hr for {rainfall.duration_hours}h")
        print(f"Antecedent moisture: {rainfall.antecedent_moisture*100:.0f}%")
        print(f"{'='*60}")

        results = model.regional_assessment(rainfall, hours)
        
        for result in sorted(results, key=lambda r: r.factor_of_safety):
            print(f"\n  [{result.threat_level.upper():8s}] {result.slope_name}")
            print(f"  FS={result.factor_of_safety:.3f} | "
                  f"Failure prob={result.failure_probability*100:.0f}% | "
                  f"Runout={result.runout_distance_m:.0f}m")