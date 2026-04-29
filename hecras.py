import math
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional

# ── USGS LIVE DATA ──────────────────────────────────────────────

def get_usgs_gauge(site_number: str) -> dict:
    """
    Pull real-time river data from USGS Water Services API.
    Free, no API key required. 8,000+ gauges nationwide.
    """
    url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "sites": site_number,
        "parameterCd": "00060,00065",  # discharge (cfs) + stage (ft)
        "format": "json"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        time_series = data["value"]["timeSeries"]
        result = {"site": site_number, "live": True}

        for ts in time_series:
            param = ts["variable"]["variableCode"][0]["value"]
            values = ts["values"][0]["value"]
            if not values:
                continue
            value = float(values[0]["value"])
            timestamp = values[0]["dateTime"]

            if param == "00060":
                result["discharge_cfs"] = value
                result["discharge_m3s"] = value * 0.0283168
                result["discharge_time"] = timestamp
            elif param == "00065":
                result["stage_ft"] = value
                result["stage_m"] = value * 0.3048
                result["stage_time"] = timestamp

        return result

    except Exception as e:
        print(f"[HECRAS] USGS API error for site {site_number}: {e}")
        return {"site": site_number, "live": False}


def get_flood_thresholds(site_number: str) -> dict:
    """Pull NWS flood stage thresholds for a USGS gauge site."""
    url = f"https://api.water.noaa.gov/nwps/v1/gauges/{site_number}/stageflow"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            stages = data.get("stages", {})
            return {
                "action_stage_m":   (stages.get("action",   {}).get("stage", 0) or 0) * 0.3048,
                "flood_stage_m":    (stages.get("flood",    {}).get("stage", 0) or 0) * 0.3048,
                "moderate_stage_m": (stages.get("moderate", {}).get("stage", 0) or 0) * 0.3048,
                "major_stage_m":    (stages.get("major",    {}).get("stage", 0) or 0) * 0.3048,
            }
    except Exception as e:
        print(f"[HECRAS] NWS threshold error for {site_number}: {e}")
    return {}


def get_gauge_info(site_number: str) -> dict:
    """Get gauge metadata — name, lat, lon, drainage area."""
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "sites": site_number,
        "format": "rdb",
        "siteOutput": "expanded"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        lines = [l for l in response.text.split("\n")
                 if l and not l.startswith("#") and not l.startswith("5")]
        if len(lines) >= 2:
            headers = lines[0].split("\t")
            values = lines[2].split("\t") if len(lines) > 2 else lines[1].split("\t")
            info = dict(zip(headers, values))
            return {
                "name": info.get("station_nm", f"Site {site_number}"),
                "lat": float(info.get("dec_lat_va", 0) or 0),
                "lon": float(info.get("dec_long_va", 0) or 0),
                "drainage_area_km2": float(info.get("drain_area_va", 0) or 0) * 2.58999,
            }
    except Exception as e:
        print(f"[HECRAS] Gauge info error: {e}")
    return {"name": f"USGS {site_number}", "lat": 0, "lon": 0, "drainage_area_km2": 0}


# ── DATA CLASSES ────────────────────────────────────────────────

@dataclass
class FloodEvent:
    precipitation_mm: float
    duration_hours: float
    antecedent_moisture: float   # 0-1 soil saturation
    time_of_concentration_h: float

@dataclass
class HECRASResult:
    site_id: str
    site_name: str
    lat: float
    lon: float
    current_discharge_m3s: float
    current_stage_m: float
    predicted_peak_m3s: float
    predicted_stage_m: float
    flood_stage_m: float
    major_flood_stage_m: float
    above_flood_stage_m: float   # how far above flood stage
    flood_width_m: float
    flood_velocity_ms: float
    time_to_peak_h: float
    return_period_years: float
    threat_level: str
    live_data: bool

    def summary(self):
        status = "LIVE" if self.live_data else "MODELED"
        flood_status = (
            f"{self.above_flood_stage_m:.2f}m ABOVE flood stage"
            if self.above_flood_stage_m > 0
            else f"{abs(self.above_flood_stage_m):.2f}m below flood stage"
        )
        return (
            f"[{status}]\n"
            f"Current Stage:    {self.current_stage_m:.2f}m\n"
            f"Current Flow:     {self.current_discharge_m3s:.0f} m³/s\n"
            f"Predicted Peak:   {self.predicted_peak_m3s:.0f} m³/s\n"
            f"Predicted Stage:  {self.predicted_stage_m:.2f}m\n"
            f"Flood Stage:      {self.flood_stage_m:.2f}m\n"
            f"Status:           {flood_status}\n"
            f"Time to Peak:     {self.time_to_peak_h:.1f}h\n"
            f"Return Period:    ~{self.return_period_years:.0f} years\n"
            f"Threat Level:     {self.threat_level.upper()}"
        )


# ── KNOWN USGS GAUGES ───────────────────────────────────────────
# Format: site_id → fallback channel width (m) and Manning n
# Used when live data unavailable

KNOWN_GAUGES = {
    "08114000": {"name": "Brazos River at Richmond TX",     "width": 120, "n": 0.030, "bankfull_m3s": 1500},
    "08042500": {"name": "Trinity River at Romayor TX",     "width": 60,  "n": 0.035, "bankfull_m3s": 850},
    "07022000": {"name": "Mississippi River at Memphis TN", "width": 800, "n": 0.025, "bankfull_m3s": 25000},
    "07301500": {"name": "Red River at Gainesville TX",     "width": 80,  "n": 0.040, "bankfull_m3s": 1200},
    "02087500": {"name": "Neuse River at Clayton NC",       "width": 40,  "n": 0.038, "bankfull_m3s": 200},
    "11447650": {"name": "Sacramento River at Verona CA",   "width": 150, "n": 0.028, "bankfull_m3s": 2000},
    "01646500": {"name": "Potomac River at Little Falls MD","width": 200, "n": 0.030, "bankfull_m3s": 3000},
}


# ── HEC-RAS MODEL ───────────────────────────────────────────────

class HECRASModel:
    """
    HEC-RAS River Flood Model with live USGS gauge integration.

    Reference: US Army Corps of Engineers 2016.
    HEC-RAS River Analysis System — Hydraulic Reference Manual v5.0.

    Pulls real-time stage and discharge from USGS Water Services API.
    Uses SCS runoff + Manning's equation for flood prediction.
    Compares predicted stage against NWS flood thresholds.
    """

    def _scs_runoff(self, precip_mm: float, moisture: float) -> float:
        """SCS Curve Number direct runoff."""
        cn = 65 + (92 - 65) * moisture
        S = (25400 / cn) - 254
        Ia = 0.2 * S
        if precip_mm <= Ia:
            return 0.0
        return (precip_mm - Ia)**2 / (precip_mm - Ia + S)

    def _peak_discharge(
        self,
        runoff_mm: float,
        event: FloodEvent,
        catchment_km2: float
    ) -> float:
        """SCS unit hydrograph with basin size scaling."""
        if runoff_mm <= 0:
            return 0.0

        Tp = 0.5 * event.duration_hours + 0.6 * event.time_of_concentration_h

        # Basin size routing attenuation
        if catchment_km2 < 500:
            scale = 1.0
        elif catchment_km2 < 5000:
            scale = 0.50
        elif catchment_km2 < 50000:
            scale = 0.15
        elif catchment_km2 < 500000:
            scale = 0.05
        else:
            scale = 0.012

        return 0.208 * catchment_km2 * runoff_mm / Tp * scale

    def _manning_stage(
        self,
        discharge_m3s: float,
        width_m: float,
        n: float,
        slope: float,
        max_depth_m: float = 30.0
    ) -> float:
        """Binary search for water surface stage using Manning's equation."""
        S = max(slope, 0.00001)
        d_low, d_high = 0.01, max_depth_m

        for _ in range(60):
            d_mid = (d_low + d_high) / 2
            A = width_m * d_mid
            P = width_m + 2 * d_mid
            R = A / P
            Q_calc = (1 / n) * A * R**(2/3) * math.sqrt(S)

            if Q_calc < discharge_m3s:
                d_low = d_mid
            else:
                d_high = d_mid

            if d_high - d_low < 0.001:
                break

        return d_high

    def _flood_width(
        self,
        stage_m: float,
        channel_depth_m: float,
        channel_width_m: float
    ) -> float:
        """Estimate inundation width from stage."""
        if stage_m <= channel_depth_m:
            return channel_width_m
        overflow = stage_m - channel_depth_m
        return min(channel_width_m + overflow * 150, channel_width_m * 20)

    def _velocity(self, discharge_m3s: float, stage_m: float, width_m: float) -> float:
        area = width_m * stage_m
        return discharge_m3s / max(area, 1)

    def _return_period(self, predicted_m3s: float, bankfull_m3s: float) -> float:
        ratio = predicted_m3s / max(bankfull_m3s, 1)
        if ratio < 1.0:  return 1.0
        elif ratio < 1.5: return 2.0
        elif ratio < 2.0: return 10.0
        elif ratio < 2.5: return 25.0
        elif ratio < 3.0: return 50.0
        elif ratio < 4.0: return 100.0
        else:             return 500.0

    def _threat(self, above_flood_m: float, velocity_ms: float) -> str:
        if above_flood_m > 2.0 or (above_flood_m > 0.5 and velocity_ms > 2.0):
            return "critical"
        elif above_flood_m > 0.5:
            return "high"
        elif above_flood_m > 0.0:
            return "medium"
        return "low"

    def calculate_live(
        self,
        site_id: str,
        event: FloodEvent,
        catchment_km2: float = None
    ) -> HECRASResult:
        """
        Run HEC-RAS with live USGS gauge data + rainfall forecast.
        Pulls real current stage/discharge, predicts peak from rainfall.
        """
        print(f"[HECRAS] Fetching live data for USGS site {site_id}...")

        # Get live gauge data
        gauge = get_usgs_gauge(site_id)
        thresholds = get_flood_thresholds(site_id)
        info = get_gauge_info(site_id)

        # Gauge geometry from known database or defaults
        known = KNOWN_GAUGES.get(site_id, {})
        width = known.get("width", 80)
        n = known.get("n", 0.035)
        bankfull = known.get("bankfull_m3s", 500)
        slope = 0.0005  # default slope

        # Drainage area
        area_km2 = catchment_km2 or info.get("drainage_area_km2") or 5000

        # Current conditions from live data
        current_q = gauge.get("discharge_m3s", 0)
        current_stage = gauge.get("stage_m", 0)
        live = gauge.get("live", False)

        # Flood thresholds
        flood_stage = thresholds.get("flood_stage_m", current_stage * 1.3)
        major_stage = thresholds.get("major_flood_stage_m", current_stage * 1.6)

        if flood_stage == 0:
            # Fallback — estimate from bankfull discharge
            flood_stage = self._manning_stage(bankfull * 1.2, width, n, slope)
            major_stage = self._manning_stage(bankfull * 2.0, width, n, slope)

        # Predicted peak from rainfall
        runoff = self._scs_runoff(event.precipitation_mm, event.antecedent_moisture)
        predicted_q = self._peak_discharge(runoff, event, area_km2)

        # Add current base flow
        total_q = current_q + predicted_q

        # Predicted stage
        predicted_stage = self._manning_stage(total_q, width, n, slope)

        # Flood metrics
        above_flood = predicted_stage - flood_stage
        flood_w = self._flood_width(predicted_stage, flood_stage, width)
        velocity = self._velocity(total_q, predicted_stage, width)

        # Time to peak
        Tp = 0.5 * event.duration_hours + 0.6 * event.time_of_concentration_h

        # Return period
        rp = self._return_period(total_q, bankfull)

        # Threat
        threat = self._threat(above_flood, velocity)

        return HECRASResult(
            site_id=site_id,
            site_name=known.get("name", info.get("name", f"USGS {site_id}")),
            lat=info.get("lat", 0),
            lon=info.get("lon", 0),
            current_discharge_m3s=current_q,
            current_stage_m=current_stage,
            predicted_peak_m3s=total_q,
            predicted_stage_m=predicted_stage,
            flood_stage_m=flood_stage,
            major_flood_stage_m=major_stage,
            above_flood_stage_m=above_flood,
            flood_width_m=flood_w,
            flood_velocity_ms=velocity,
            time_to_peak_h=Tp,
            return_period_years=rp,
            threat_level=threat,
            live_data=live
        )

    def beacon_priority_zones(self, results: List[HECRASResult]) -> dict:
        zones = {}
        color_map = {
            "critical": "red", "high": "orange",
            "medium": "yellow", "low": "yellow"
        }
        for r in results:
            color = color_map[r.threat_level]
            if color not in zones:
                zones[color] = {
                    "priority": {"red": 1, "orange": 2, "yellow": 3}[color],
                    "sites": [],
                    "max_above_flood_m": 0,
                    "deploy": "sub" if r.above_flood_stage_m > 1.0 else "drone",
                    "label": f"River flood — {r.threat_level}"
                }
            zones[color]["sites"].append(r.site_name)
            zones[color]["max_above_flood_m"] = max(
                zones[color]["max_above_flood_m"],
                r.above_flood_stage_m
            )
        return zones


if __name__ == "__main__":
    model = HECRASModel()

    print("🌊 HEC-RAS RIVER FLOOD MODEL")
    print("US Army Corps of Engineers + Live USGS Gauge Data\n")

    # Real USGS gauges with live data
    scenarios = [
        (
            "Brazos River at Richmond TX — Harvey Conditions",
            "08114000",
            FloodEvent(300, 48, 0.85, 18),
            30000
        ),
        (
            "Trinity River at Romayor TX — Spring Flood",
            "08042500",
            FloodEvent(150, 24, 0.70, 8),
            8000
        ),
        (
            "Mississippi River at Memphis TN — 100-Year",
            "07022000",
            FloodEvent(200, 72, 0.80, 72),
            1800000
        ),
        (
            "Red River at Gainesville TX — Flash Flood",
            "07301500",
            FloodEvent(120, 6, 0.60, 4),
            5000
        ),
    ]

    results = []
    for name, site_id, event, catchment in scenarios:
        print(f"\n{'='*58}")
        print(f"SCENARIO: {name}")
        print(f"Rain: {event.precipitation_mm}mm over {event.duration_hours}h")
        print(f"{'='*58}")
        result = model.calculate_live(site_id, event, catchment)
        print(result.summary())
        results.append(result)

    print(f"\n{'='*58}")
    print("BEACON PRIORITY ZONES")
    print(f"{'='*58}")
    zones = model.beacon_priority_zones(results)
    for color, zone in zones.items():
        print(f"[{color.upper()}] {zone['label']}")
        print(f"  Sites: {', '.join(zone['sites'])}")
        print(f"  Max above flood: {zone['max_above_flood_m']:.2f}m")
        print(f"  Deploy: {zone['deploy']}")