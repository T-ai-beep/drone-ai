import math
from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta

@dataclass
class AftershockForecast:
    main_magnitude: float
    epicenter_lat: float
    epicenter_lon: float
    forecast_days: int
    expected_aftershocks_m3plus: float
    expected_aftershocks_m5plus: float
    expected_aftershocks_m6plus: float
    probability_m6plus: float
    probability_m7plus: float
    largest_expected_magnitude: float
    no_go_radius_km: float
    threat_level: str

    def summary(self):
        return (
            f"Main Shock:      M{self.main_magnitude}\n"
            f"Forecast Period: {self.forecast_days} days\n"
            f"\nEXPECTED AFTERSHOCKS:\n"
            f"  M3.0+:  {self.expected_aftershocks_m3plus:.0f}\n"
            f"  M5.0+:  {self.expected_aftershocks_m5plus:.1f}\n"
            f"  M6.0+:  {self.expected_aftershocks_m6plus:.2f}\n"
            f"\nPROBABILITIES:\n"
            f"  P(M6.0+): {self.probability_m6plus*100:.1f}%\n"
            f"  P(M7.0+): {self.probability_m7plus*100:.1f}%\n"
            f"\nLargest Expected: M{self.largest_expected_magnitude:.1f}\n"
            f"No-Go Radius:    {self.no_go_radius_km:.0f}km\n"
            f"Threat Level:    {self.threat_level.upper()}"
        )


class AftershockModel:
    """
    Reasenberg-Jones Aftershock Forecasting Model.
    
    Reference: Reasenberg, P.A. and Jones, L.M. 1989.
    Earthquake Hazard After a Mainshock in California.
    Science, 243, 1173-1176.
    
    Uses Omori-Utsu decay law combined with Gutenberg-Richter
    magnitude-frequency relationship to forecast aftershock sequences.
    Same model used by USGS for operational aftershock forecasting.
    """

    # Omori-Utsu parameters (generic California values)
    # dN/dt = K / (t + c)^p
    K = 0.018    # productivity
    c = 0.05     # time offset (days)
    p = 1.08     # decay rate
    
    # Gutenberg-Richter b-value
    b = 1.0
    
    # Magnitude of completeness
    Mc = 3.0

    def _omori_rate(self, t_days: float, main_mag: float) -> float:
        """
        Omori-Utsu aftershock rate at time t after mainshock.
        N(t) = K * 10^(a*(M-Mc)) / (t + c)^p
        """
        a = 1.0  # aftershock productivity parameter
        K_eff = self.K * 10**(a * (main_mag - self.Mc))
        return K_eff / (t_days + self.c)**self.p

    def _expected_count(
        self,
        main_mag: float,
        min_mag: float,
        t_start: float,
        t_end: float
    ) -> float:
        """
        Expected number of aftershocks M >= min_mag
        between t_start and t_end days after mainshock.
        Uses Reasenberg-Jones model.
        """
        # Gutenberg-Richter scaling
        gr_factor = 10**(-self.b * (min_mag - self.Mc))

        # Integrate Omori rate
        a = 1.0
        K_eff = self.K * 10**(a * (main_mag - self.Mc))

        if self.p == 1.0:
            integral = K_eff * math.log((t_end + self.c) / (t_start + self.c))
        else:
            integral = (K_eff / (1 - self.p)) * (
                (t_end + self.c)**(1 - self.p) -
                (t_start + self.c)**(1 - self.p)
            )

        return integral * gr_factor

    def _probability_exceeding(
        self,
        expected_count: float
    ) -> float:
        """
        Probability of at least one event given expected count.
        Uses Poisson distribution: P = 1 - e^(-lambda)
        """
        return 1.0 - math.exp(-expected_count)

    def _largest_expected(self, main_mag: float) -> float:
        """
        Expected magnitude of largest aftershock.
        Bath's Law: largest aftershock ~ M_main - 1.2
        """
        return main_mag - 1.2

    def _no_go_radius(self, main_mag: float, threat_level: str) -> float:
        """
        No-go radius for rescue operations.
        Based on rupture length and aftershock zone extent.
        """
        # Wells & Coppersmith rupture length scaling
        rupture_km = 10**(0.69 * main_mag - 3.22)
        
        # Aftershock zone extends ~1.5x rupture length
        zone_km = rupture_km * 1.5
        
        # Safety buffer by threat level
        if threat_level == "critical":
            return zone_km * 2.0
        elif threat_level == "high":
            return zone_km * 1.5
        else:
            return zone_km

    def _threat_level(
        self,
        prob_m6: float,
        prob_m7: float,
        expected_m5: float
    ) -> str:
        if prob_m7 > 0.10 or prob_m6 > 0.50:
            return "critical"
        elif prob_m6 > 0.20 or expected_m5 > 2.0:
            return "high"
        elif prob_m6 > 0.05 or expected_m5 > 0.5:
            return "medium"
        return "low"

    def calculate(
        self,
        main_magnitude: float,
        epicenter_lat: float,
        epicenter_lon: float,
        forecast_days: int = 7,
        t_since_mainshock_days: float = 0.0
    ) -> AftershockForecast:
        """
        Forecast aftershock sequence after a mainshock.
        
        Args:
            main_magnitude: Mainshock magnitude
            epicenter_lat, epicenter_lon: Epicenter coordinates
            forecast_days: Days to forecast ahead
            t_since_mainshock_days: Days since mainshock occurred
        """
        t_start = t_since_mainshock_days
        t_end = t_since_mainshock_days + forecast_days

        # Expected counts by magnitude threshold
        n_m3 = self._expected_count(main_magnitude, 3.0, t_start, t_end)
        n_m5 = self._expected_count(main_magnitude, 5.0, t_start, t_end)
        n_m6 = self._expected_count(main_magnitude, 6.0, t_start, t_end)
        n_m7 = self._expected_count(main_magnitude, 7.0, t_start, t_end)

        # Probabilities
        p_m6 = self._probability_exceeding(n_m6)
        p_m7 = self._probability_exceeding(n_m7)

        # Largest expected
        largest = self._largest_expected(main_magnitude)

        # Threat level
        threat = self._threat_level(p_m6, p_m7, n_m5)

        # No-go radius
        no_go = self._no_go_radius(main_magnitude, threat)

        return AftershockForecast(
            main_magnitude=main_magnitude,
            epicenter_lat=epicenter_lat,
            epicenter_lon=epicenter_lon,
            forecast_days=forecast_days,
            expected_aftershocks_m3plus=n_m3,
            expected_aftershocks_m5plus=n_m5,
            expected_aftershocks_m6plus=n_m6,
            probability_m6plus=p_m6,
            probability_m7plus=p_m7,
            largest_expected_magnitude=largest,
            no_go_radius_km=no_go,
            threat_level=threat
        )

    def sequence_forecast(
        self,
        main_magnitude: float,
        epicenter_lat: float,
        epicenter_lon: float,
        days: List[int] = [1, 3, 7, 14, 30]
    ) -> List[AftershockForecast]:
        """Forecast aftershock sequence at multiple time windows."""
        forecasts = []
        for day in days:
            forecast = self.calculate(
                main_magnitude, epicenter_lat, epicenter_lon,
                forecast_days=day
            )
            forecasts.append(forecast)
        return forecasts

    def beacon_priority_zones(self, forecast: AftershockForecast) -> dict:
        color_map = {"critical": "red", "high": "orange",
                    "medium": "yellow", "low": "yellow"}
        color = color_map[forecast.threat_level]
        return {
            color: {
                "priority": {"red": 1, "orange": 2, "yellow": 3}[color],
                "no_go_radius_km": forecast.no_go_radius_km,
                "prob_m6": forecast.probability_m6plus,
                "largest_expected": forecast.largest_expected_magnitude,
                "deploy": "drone only — no ground teams in aftershock zone",
                "label": f"Aftershock zone — {forecast.no_go_radius_km:.0f}km no-go radius"
            }
        }


if __name__ == "__main__":
    model = AftershockModel()

    print("🔄 REASENBERG-JONES AFTERSHOCK FORECASTING MODEL")
    print("Omori-Utsu decay + Gutenberg-Richter scaling\n")

    scenarios = [
        ("2023 Morocco M6.8 — Day 0", 6.8, 31.12, -8.38, 7, 0.0),
        ("2021 Kermadec M8.1 — Day 0", 8.1, -29.72, -177.28, 30, 0.0),
        ("2021 Kermadec M8.1 — Day 3", 8.1, -29.72, -177.28, 7, 3.0),
        ("Hypothetical Dallas M6.5 — Day 0", 6.5, 32.78, -96.80, 14, 0.0),
    ]

    for name, mag, lat, lon, days, t_since in scenarios:
        print(f"\n{'='*55}")
        print(f"SCENARIO: {name}")
        print(f"{'='*55}")
        result = model.calculate(mag, lat, lon, days, t_since)
        print(result.summary())