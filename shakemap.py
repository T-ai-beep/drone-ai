import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class EarthquakeEvent:
    magnitude: float        # Richter/Moment magnitude
    depth_km: float         # focal depth in km
    epicenter_lat: float    # epicenter latitude
    epicenter_lon: float    # epicenter longitude
    fault_type: str         # strike_slip, reverse, normal

@dataclass
class ShakeMapResult:
    pga: float             # Peak Ground Acceleration (g)
    pgv: float             # Peak Ground Velocity (cm/s)
    mmi: float             # Modified Mercalli Intensity (1-12)
    mmi_description: str   # human readable intensity
    site_lat: float
    site_lon: float
    distance_km: float
    threat_level: str

    def summary(self):
        return (
            f"Distance from epicenter: {self.distance_km:.1f}km\n"
            f"Peak Ground Acceleration: {self.pga:.4f}g ({self.pga*980:.1f}cm/s²)\n"
            f"Peak Ground Velocity:     {self.pgv:.1f}cm/s\n"
            f"MMI Intensity:            {self.mmi:.1f} — {self.mmi_description}\n"
            f"Threat Level:             {self.threat_level.upper()}"
        )

class ShakeMap:
    """
    USGS ShakeMap ground motion model.
    
    Uses Boore-Atkinson 2008 Ground Motion Prediction Equation (GMPE)
    — the same attenuation relationship used by USGS ShakeMap.
    
    Calculates Peak Ground Acceleration and velocity at any
    distance from an earthquake epicenter.
    """

    # MMI descriptions
    MMI_DESC = {
        1: "Not felt",
        2: "Weak",
        3: "Weak",
        4: "Light",
        5: "Moderate",
        6: "Strong",
        7: "Very Strong",
        8: "Severe",
        9: "Violent",
        10: "Extreme",
        11: "Extreme",
        12: "Catastrophic"
    }

    # Fault type coefficients for Boore-Atkinson 2008
    FAULT_COEFFS = {
        "strike_slip": 0.0,
        "reverse": 0.28,
        "normal": -0.12,
        "unknown": 0.0
    }

    def _hypocentral_distance(
        self,
        site_lat: float,
        site_lon: float,
        event: EarthquakeEvent
    ) -> float:
        """Calculate hypocentral distance (km) accounting for focal depth."""
        # Haversine formula for surface distance
        R = 6371.0  # Earth radius km
        lat1 = math.radians(event.epicenter_lat)
        lat2 = math.radians(site_lat)
        dlat = math.radians(site_lat - event.epicenter_lat)
        dlon = math.radians(site_lon - event.epicenter_lon)

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        surface_dist = R * c

        # Add focal depth for hypocentral distance
        return math.sqrt(surface_dist**2 + event.depth_km**2)

    def _boore_atkinson_pga(
        self,
        event: EarthquakeEvent,
        r_hypo: float,
        vs30: float = 760.0
    ) -> float:
        """
        Boore-Atkinson 2008 GMPE for PGA.
        vs30 = 760 m/s is reference rock condition.
        Returns PGA in g.
        """
        M = event.magnitude
        R = max(r_hypo, 1.0)

        # BA08 coefficients for PGA
        e1 = -0.53804
        e2 = -0.50350
        e3 = -0.75472
        e4 = 0.27090
        e5 = 5.00121
        e6 = -0.00430
        e7 = -0.00000
        Mh = 6.75
        c1 = -0.66050
        c2 = 0.11970
        c3 = -0.01151
        h = 2.54
        blin = -0.36
        b1 = -0.64
        b2 = -0.14

        # Fault type correction
        fault_corr = self.FAULT_COEFFS.get(event.fault_type, 0.0)

        # Distance term
        R_rup = math.sqrt(R**2 + h**2)
        f_dis = (c1 + c2*(M - Mh)) * math.log(R_rup) + c3 * R_rup

        # Magnitude scaling
        if M <= Mh:
            f_mag = e1 + e2*(M - Mh) + e3*(M - Mh)**2 + e5*(M - Mh)
        else:
            f_mag = 1 + e4*(M - Mh) + e7*(M - Mh)**2

        # Site amplification (simplified)
        Vref = 760.0
        if vs30 <= Vref:
            f_site = blin * math.log(vs30/Vref)
        else:
            f_site = 0.0

        # Total log PGA
        ln_pga = f_mag + fault_corr + f_dis + f_site

        return math.exp(ln_pga)

    def _pgv_from_pga(self, pga: float, magnitude: float) -> float:
        """Estimate PGV from PGA using empirical relationship."""
        # Campbell & Bozorgnia 2003 simplified
        return pga * 980 * 0.1 * (1 + 0.5 * (magnitude - 6.0))

    def _pga_to_mmi(self, pga: float) -> float:
        """
        Convert PGA to Modified Mercalli Intensity.
        Wald et al. 1999 relationship used by USGS ShakeMap.
        """
        if pga <= 0:
            return 1.0

        # Wald 1999: MMI = 3.66 * log10(PGA*980) - 1.66
        pga_cms2 = pga * 980.0  # convert g to cm/s2

        if pga_cms2 < 0.17:
            return 1.0
        elif pga_cms2 < 1.4:
            mmi = 3.66 * math.log10(pga_cms2) - 1.66
        else:
            mmi = 3.47 * math.log10(pga_cms2) + 1.22

        return max(1.0, min(12.0, mmi))

    def _threat_from_mmi(self, mmi: float) -> str:
        if mmi >= 8:
            return "critical"
        elif mmi >= 6:
            return "high"
        elif mmi >= 4:
            return "medium"
        return "low"

    def calculate(
        self,
        event: EarthquakeEvent,
        site_lat: float,
        site_lon: float,
        vs30: float = 760.0
    ) -> ShakeMapResult:
        """
        Calculate ground shaking at a specific site from an earthquake.
        
        Args:
            event: EarthquakeEvent parameters
            site_lat, site_lon: Location to calculate shaking at
            vs30: Average shear wave velocity top 30m (m/s)
                  760 = rock, 360 = stiff soil, 180 = soft soil
        
        Returns:
            ShakeMapResult with PGA, PGV, MMI
        """
        r_hypo = self._hypocentral_distance(site_lat, site_lon, event)
        pga = self._boore_atkinson_pga(event, r_hypo, vs30)
        pgv = self._pgv_from_pga(pga, event.magnitude)
        mmi = self._pga_to_mmi(pga)
        mmi_int = int(min(12, max(1, round(mmi))))
        threat = self._threat_from_mmi(mmi)

        return ShakeMapResult(
            pga=pga,
            pgv=pgv,
            mmi=mmi,
            mmi_description=self.MMI_DESC.get(mmi_int, "Unknown"),
            site_lat=site_lat,
            site_lon=site_lon,
            distance_km=r_hypo,
            threat_level=threat
        )

    def map_region(
        self,
        event: EarthquakeEvent,
        center_lat: float,
        center_lon: float,
        radius_km: float = 50,
        grid_points: int = 5
    ) -> List[ShakeMapResult]:
        """
        Generate ShakeMap grid across a region.
        Returns list of ShakeMapResult for each grid point.
        """
        results = []
        step = (radius_km / 111.0) / grid_points

        for i in range(-grid_points, grid_points + 1):
            for j in range(-grid_points, grid_points + 1):
                lat = center_lat + i * step
                lon = center_lon + j * step
                result = self.calculate(event, lat, lon)
                results.append(result)

        return results

    def beacon_priority_zones(
        self,
        event: EarthquakeEvent,
        center_lat: float,
        center_lon: float
    ) -> dict:
        """Generate Beacon priority zones from ShakeMap output."""
        zones = {}

        # Calculate at different distances
        distances = [5, 15, 35]
        for i, dist_km in enumerate(distances):
            # Calculate site at this distance
            site_lat = center_lat + (dist_km / 111.0)
            result = self.calculate(event, site_lat, center_lon)

            color = ["red", "orange", "yellow"][i]
            zones[color] = {
                "priority": i + 1,
                "radius_km": dist_km,
                "pga": result.pga,
                "mmi": result.mmi,
                "threat": result.threat_level,
                "deploy": "drone" if i == 0 else "rover" if i == 1 else "drone",
                "label": f"MMI {result.mmi:.1f} — {result.mmi_description}"
            }

        return zones


if __name__ == "__main__":
    shake = ShakeMap()

    print("🌍 USGS SHAKEMAP GROUND MOTION MODEL")
    print("Boore-Atkinson 2008 GMPE\n")

    scenarios = [
        ("2023 Morocco Earthquake", EarthquakeEvent(6.8, 18.5, 31.12, -8.38, "strike_slip"),
         31.5, -8.0),
        ("2021 Kermadec M8.1", EarthquakeEvent(8.1, 10.0, -29.72, -177.28, "reverse"),
         -37.0, 175.0),
        ("Hypothetical Dallas M6.5", EarthquakeEvent(6.5, 15.0, 32.78, -96.80, "strike_slip"),
         33.10, -96.67),
    ]

    for name, event, site_lat, site_lon in scenarios:
        print(f"\n{'='*55}")
        print(f"SCENARIO: {name}")
        print(f"{'='*55}")
        print(f"Magnitude:  M{event.magnitude}")
        print(f"Depth:      {event.depth_km}km")
        print(f"Fault Type: {event.fault_type}")
        result = shake.calculate(event, site_lat, site_lon)
        print(f"\n[SHAKEMAP OUTPUT]")
        print(result.summary())