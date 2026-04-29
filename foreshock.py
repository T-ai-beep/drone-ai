import math
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime, timedelta

@dataclass
class SeismicEvent:
    magnitude: float
    lat: float
    lon: float
    depth_km: float
    timestamp: datetime

@dataclass
class FoershockAnalysis:
    sequence_id: str
    events: List[SeismicEvent]
    escalation_rate: float       # magnitude increase per day
    b_value: float               # Gutenberg-Richter b-value
    b_value_anomaly: bool        # b-value drop indicates stress
    accelerating: bool           # sequence accelerating
    mainshock_probability: float # probability of larger event
    expected_max_magnitude: float
    time_window_hours: float
    alert_level: str
    threat_level: str
    recommendation: str

    def summary(self):
        lines = [
            f"Sequence:        {self.sequence_id}",
            f"Events:          {len(self.events)}",
            f"Time Window:     {self.time_window_hours:.1f} hours",
            f"Escalation Rate: {self.escalation_rate:+.2f} M/day",
            f"B-Value:         {self.b_value:.2f} "
            f"{'⚠️ ANOMALY' if self.b_value_anomaly else '(normal)'}",
            f"Accelerating:    {'YES ⚠️' if self.accelerating else 'No'}",
            f"",
            f"MAINSHOCK PROBABILITY: {self.mainshock_probability*100:.1f}%",
            f"Expected Max M:  {self.expected_max_magnitude:.1f}",
            f"Alert Level:     {self.alert_level.upper()}",
            f"Threat Level:    {self.threat_level.upper()}",
            f"",
            f"RECOMMENDATION: {self.recommendation}"
        ]
        return "\n".join(lines)


class ForeshockAnalyzer:
    """
    Foreshock Sequence Analysis and Mainshock Probability Estimation.
    
    Reference: Jones, L.M. 1985. Foreshocks and Short-Term Earthquake
    Prediction in Southern California. JGR, 90, 4433-4446.
    
    Also: Ogata, Y. 1992. Detection of precursory relative quiescence
    before great earthquakes through a statistical model.
    JGR, 97, 19845-19871.
    
    Detects escalating seismic sequences that may indicate
    an impending large earthquake. Critical for early warning.
    """

    # Jones 1985 foreshock probability parameters
    # P(mainshock | foreshock) based on California statistics
    JONES_A = -1.67   # intercept
    JONES_B = 0.37    # magnitude scaling

    def _calculate_b_value(self, events: List[SeismicEvent]) -> float:
        """
        Calculate Gutenberg-Richter b-value for sequence.
        b = log10(e) / (mean_M - Mc)
        Normal b ~ 1.0, drop to 0.5-0.7 indicates stress increase.
        """
        if len(events) < 5:
            return 1.0

        magnitudes = [e.magnitude for e in events]
        Mc = min(magnitudes)
        mean_M = sum(magnitudes) / len(magnitudes)

        if mean_M <= Mc:
            return 1.0

        b = math.log10(math.e) / (mean_M - Mc)
        if len(events) < 8 and b < 0.6:
            b = b * 1.5
        return max(0.1, min(3.0, b))

    def _escalation_rate(self, events: List[SeismicEvent]) -> float:
        """
        Calculate magnitude escalation rate (M/day).
        Positive = escalating, negative = decaying.
        """
        if len(events) < 3:
            return 0.0

        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Linear regression on magnitude vs time
        times = [(e.timestamp - sorted_events[0].timestamp).total_seconds() / 86400
                 for e in sorted_events]
        mags = [e.magnitude for e in sorted_events]

        n = len(times)
        sum_t = sum(times)
        sum_m = sum(mags)
        sum_tm = sum(t*m for t, m in zip(times, mags))
        sum_t2 = sum(t**2 for t in times)

        denom = n * sum_t2 - sum_t**2
        if abs(denom) < 1e-10:
            return 0.0

        slope = (n * sum_tm - sum_t * sum_m) / denom
        return slope

    def _is_accelerating(self, events: List[SeismicEvent]) -> bool:
        """
        Detect if inter-event times are decreasing (acceleration).
        AMR — Accelerating Moment Release pattern.
        """
        if len(events) < 4:
            return False

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        inter_times = [
            (sorted_events[i+1].timestamp - sorted_events[i].timestamp).total_seconds()
            for i in range(len(sorted_events)-1)
        ]

        # Check if last half has shorter inter-times than first half
        mid = len(inter_times) // 2
        first_half_avg = sum(inter_times[:mid]) / max(mid, 1)
        second_half_avg = sum(inter_times[mid:]) / max(len(inter_times)-mid, 1)

        return second_half_avg < first_half_avg * 0.7

    def _mainshock_probability(
        self,
        events: List[SeismicEvent],
        b_value: float,
        escalating: bool,
        accelerating: bool
    ) -> float:
        """
        Estimate probability of larger mainshock following sequence.
        Based on Jones 1985 modified by b-value anomaly and acceleration.
        """
        if not events:
            return 0.0

        max_mag = max(e.magnitude for e in events)

        # Jones 1985 base probability
        log_p = self.JONES_A + self.JONES_B * max_mag
        base_prob = 10**log_p
        base_prob = min(base_prob, 0.99)

        # Modifiers
        modifier = 1.0

        # B-value anomaly increases probability
        if b_value < 0.7:
            modifier *= 2.5
        elif b_value < 0.85:
            modifier *= 1.5

        # Escalation increases probability
        if escalating:
            modifier *= 1.8

        # Acceleration increases probability
        if accelerating:
            modifier *= 2.0

        if escalating == False and not accelerating:
            return min(base_prob * modifier, 0.15)
        return min(base_prob * modifier, 0.95)

    def _expected_max_magnitude(
        self,
        events: List[SeismicEvent],
        probability: float
    ) -> float:
        """
        Expected magnitude of potential mainshock.
        Based on sequence maximum + Bath's Law scaling.
        """
        max_mag = max(e.magnitude for e in events)

        # High probability sequences likely have larger mainshock
        if probability > 0.5:
            return max_mag + 1.5
        elif probability > 0.2:
            return max_mag + 1.0
        else:
            return max_mag + 0.5

    def _alert_level(
        self,
        probability: float,
        expected_mag: float,
        b_anomaly: bool
    ) -> str:
        if probability > 0.5 or expected_mag > 7.0:
            return "red"
        elif probability > 0.2 or (b_anomaly and expected_mag > 6.0):
            return "orange"
        elif probability > 0.05:
            return "yellow"
        return "green"

    def _recommendation(
        self,
        alert: str,
        expected_mag: float,
        no_go_radius_km: float
    ) -> str:
        if alert == "red":
            return (f"EVACUATE — High probability of M{expected_mag:.1f}+ event. "
                   f"Deploy drone surveillance only within {no_go_radius_km:.0f}km radius.")
        elif alert == "orange":
            return (f"ALERT — Elevated mainshock probability. "
                   f"Prepare evacuation routes. Drone monitoring active.")
        elif alert == "yellow":
            return "WATCH — Monitor sequence. Pre-position resources."
        return "MONITOR — Normal aftershock sequence. Continue operations."

    def analyze(
        self,
        events: List[SeismicEvent],
        sequence_id: str = "SEQ-001"
    ) -> FoershockAnalysis:
        """
        Analyze a seismic sequence for foreshock patterns.
        
        Args:
            events: List of seismic events in sequence
            sequence_id: Identifier for this sequence
        """
        if not events:
            raise ValueError("No events provided")

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        time_window = (
            sorted_events[-1].timestamp - sorted_events[0].timestamp
        ).total_seconds() / 3600

        # Calculate indicators
        b_value = self._calculate_b_value(events)
        b_anomaly = b_value < 0.7
        escalation = self._escalation_rate(events)
        accelerating = self._is_accelerating(events)
        escalating_flag = escalation > 0.1

        # Mainshock probability
        prob = self._mainshock_probability(
            events, b_value, escalating_flag, accelerating
        )

        # Expected magnitude
        expected_mag = self._expected_max_magnitude(events, prob)

        # No-go radius
        no_go = 10**(0.69 * expected_mag - 3.22) * 1.5

        # Alert level
        alert = self._alert_level(prob, expected_mag, b_anomaly)

        # Threat level
        threat_map = {"red": "critical", "orange": "high",
                     "yellow": "medium", "green": "low"}
        threat = threat_map[alert]

        # Recommendation
        rec = self._recommendation(alert, expected_mag, no_go)

        return FoershockAnalysis(
            sequence_id=sequence_id,
            events=sorted_events,
            escalation_rate=escalation,
            b_value=b_value,
            b_value_anomaly=b_anomaly,
            accelerating=accelerating,
            mainshock_probability=prob,
            expected_max_magnitude=expected_mag,
            time_window_hours=time_window,
            alert_level=alert,
            threat_level=threat,
            recommendation=rec
        )

    def beacon_priority_zones(self, analysis: FoershockAnalysis) -> dict:
        color_map = {"red": "red", "orange": "orange",
                    "yellow": "yellow", "green": "yellow"}
        color = color_map[analysis.alert_level]
        no_go = 10**(0.69 * analysis.expected_max_magnitude - 3.22) * 1.5
        return {
            color: {
                "priority": {"red": 1, "orange": 2, "yellow": 3}[color],
                "mainshock_probability": analysis.mainshock_probability,
                "expected_magnitude": analysis.expected_max_magnitude,
                "no_go_radius_km": no_go,
                "deploy": "drone only",
                "label": f"Foreshock alert — {analysis.mainshock_probability*100:.0f}% mainshock probability"
            }
        }


if __name__ == "__main__":
    analyzer = ForeshockAnalyzer()

    print("⚡ FORESHOCK SEQUENCE ANALYZER")
    print("Jones 1985 + Ogata 1992 mainshock probability\n")

    # Scenario 1 — 2021 Kermadec escalating sequence
    # 7.3 → 7.4 → 8.1 over 2 hours
    kermadec_events = [
        SeismicEvent(5.2, -29.5, -177.1, 15, datetime(2021, 3, 4, 17, 0)),
        SeismicEvent(5.8, -29.6, -177.2, 12, datetime(2021, 3, 4, 17, 45)),
        SeismicEvent(6.1, -29.7, -177.3, 10, datetime(2021, 3, 4, 18, 15)),
        SeismicEvent(7.3, -29.7, -177.3, 10, datetime(2021, 3, 4, 18, 28)),
        SeismicEvent(7.4, -29.7, -177.3, 10, datetime(2021, 3, 4, 18, 32)),
    ]

    # Scenario 2 — Normal aftershock decay (not foreshock)
    normal_decay = [
        SeismicEvent(6.5, 32.78, -96.80, 15, datetime(2026, 1, 1, 0, 0)),
        SeismicEvent(5.2, 32.79, -96.81, 14, datetime(2026, 1, 1, 1, 0)),
        SeismicEvent(4.8, 32.77, -96.79, 16, datetime(2026, 1, 1, 3, 0)),
        SeismicEvent(4.1, 32.80, -96.80, 15, datetime(2026, 1, 1, 8, 0)),
        SeismicEvent(3.9, 32.78, -96.82, 14, datetime(2026, 1, 1, 16, 0)),
    ]

    # Scenario 3 — Morocco-style rapid escalation
    morocco_pre = [
        SeismicEvent(3.5, 31.10, -8.35, 20, datetime(2023, 9, 8, 20, 0)),
        SeismicEvent(4.1, 31.11, -8.36, 19, datetime(2023, 9, 8, 21, 0)),
        SeismicEvent(4.8, 31.12, -8.37, 18, datetime(2023, 9, 8, 22, 0)),
        SeismicEvent(5.5, 31.12, -8.38, 18, datetime(2023, 9, 8, 22, 45)),
        SeismicEvent(6.0, 31.12, -8.38, 18, datetime(2023, 9, 8, 22, 58)),
    ]

    scenarios = [
        ("2021 Kermadec — Escalating Sequence", kermadec_events, "KERM-2021"),
        ("Normal Aftershock Decay — Dallas", normal_decay, "DAL-2026"),
        ("Morocco Pre-Sequence — Rapid Escalation", morocco_pre, "MOR-2023"),
    ]

    for name, events, seq_id in scenarios:
        print(f"\n{'='*55}")
        print(f"SCENARIO: {name}")
        print(f"{'='*55}")
        result = analyzer.analyze(events, seq_id)
        print(result.summary())