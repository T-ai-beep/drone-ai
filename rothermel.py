import math
from dataclasses import dataclass

@dataclass
class FuelModel:
    name: str
    fuel_load_1hr: float      # lb/ft2 - fine dead fuel
    fuel_load_10hr: float     # lb/ft2 - medium dead fuel
    fuel_load_100hr: float    # lb/ft2 - coarse dead fuel
    fuel_load_live: float     # lb/ft2 - live fuel
    fuel_depth: float         # feet - fuel bed depth
    extinction_moisture: float # fraction (not percent)
    sav_1hr: float            # ft2/ft3 - surface area to volume ratio

# Anderson 1982 fuel models converted to lb/ft2
# Original tons/acre * 2000lb/ton / 43560ft2/acre = lb/ft2
CONV = 2000 / 43560  # tons/acre to lb/ft2

FUEL_MODELS = {
    1:  FuelModel("Short Grass",     0.74*CONV, 0.00,      0.00,      0.00,      1.0, 0.12, 3500),
    2:  FuelModel("Timber Grass",    2.00*CONV, 1.00*CONV, 0.50*CONV, 0.50*CONV, 1.0, 0.15, 3000),
    3:  FuelModel("Tall Grass",      3.01*CONV, 0.00,      0.00,      0.00,      2.5, 0.25, 1500),
    4:  FuelModel("Chaparral",       5.01*CONV, 4.01*CONV, 2.00*CONV, 5.01*CONV, 6.0, 0.20, 2000),
    5:  FuelModel("Brush",           1.00*CONV, 0.50*CONV, 0.00,      2.00*CONV, 2.0, 0.20, 2000),
    6:  FuelModel("Dormant Brush",   1.50*CONV, 2.50*CONV, 2.00*CONV, 0.00,      2.5, 0.25, 1750),
    7:  FuelModel("Southern Rough",  1.13*CONV, 1.87*CONV, 1.50*CONV, 0.37*CONV, 2.5, 0.40, 1750),
    8:  FuelModel("Compact Timber",  1.50*CONV, 1.00*CONV, 2.50*CONV, 0.00,      0.2, 0.30, 2000),
    9:  FuelModel("Hardwood Litter", 2.92*CONV, 0.41*CONV, 0.15*CONV, 0.00,      0.2, 0.25, 2500),
    10: FuelModel("Timber Litter",   3.01*CONV, 2.00*CONV, 5.01*CONV, 2.00*CONV, 1.0, 0.25, 2000),
    11: FuelModel("Light Slash",     1.50*CONV, 3.50*CONV, 5.30*CONV, 0.00,      1.0, 0.15, 1500),
    12: FuelModel("Medium Slash",    4.01*CONV, 14.0*CONV, 16.5*CONV, 0.00,      2.3, 0.20, 1500),
    13: FuelModel("Heavy Slash",     7.01*CONV, 23.0*CONV, 28.0*CONV, 0.00,      3.0, 0.25, 1500),
}

@dataclass
class FireEnvironment:
    fuel_model: int
    wind_speed: float         # mph at 20ft
    wind_direction: float     # degrees FROM
    slope: float              # percent
    aspect: float             # degrees
    moisture_1hr: float       # fraction (e.g. 0.06 = 6%)
    moisture_10hr: float      # fraction
    moisture_100hr: float     # fraction
    moisture_live: float      # fraction

@dataclass
class FireBehavior:
    spread_rate_fpm: float    # ft/min
    spread_rate_mph: float    # mph
    spread_rate_mpm: float    # m/min
    intensity: float          # BTU/ft/s fireline intensity
    flame_length: float       # feet
    direction: float          # degrees

    def summary(self):
        chains_hr = self.spread_rate_fpm * 60 / 66
        return (
            f"Rate of Spread: {chains_hr:.1f} ch/hr ({self.spread_rate_mph:.2f} mph)\n"
            f"Flame Length:   {self.flame_length:.1f} ft\n"
            f"Intensity:      {self.intensity:.0f} BTU/ft/s\n"
            f"Direction:      {self.direction:.0f}°"
        )

class RothermelModel:
    """
    Rothermel 1972 Surface Fire Spread Model
    Reference: Rothermel, R.C. 1972. A mathematical model for predicting
    fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.
    """

    def calculate(self, env: FireEnvironment) -> FireBehavior:
        fuel = FUEL_MODELS.get(env.fuel_model)
        if not fuel:
            raise ValueError(f"Unknown fuel model: {env.fuel_model}")

        w_d = fuel.fuel_load_1hr + fuel.fuel_load_10hr + fuel.fuel_load_100hr
        w_n = w_d * (1.0 - 0.0555)  # net fuel load (subtract mineral content)

        # Bulk density (lb/ft3)
        rho_b = w_d / fuel.fuel_depth if fuel.fuel_depth > 0 else 0.001

        # Particle density (lb/ft3) - standard for wood
        rho_p = 32.0

        # Packing ratio
        beta = rho_b / rho_p
        beta = max(beta, 1e-6)

        # Optimum packing ratio (Rothermel eq 37)
        sigma = fuel.sav_1hr
        beta_op = 3.348 * sigma**(-0.8189)

        # Maximum reaction velocity (1/min) (eq 36)
        sigma15 = sigma**1.5
        gamma_max = sigma15 / (495.0 + 0.0594 * sigma15)

        # Optimum reaction velocity (1/min) (eq 38)
        A = 133.0 * sigma**(-0.7913)
        beta_ratio = beta / beta_op
        gamma_op = gamma_max * (beta_ratio**A) * math.exp(A * (1.0 - beta_ratio))

        # Moisture damping (eq 29)
        M_f = env.moisture_1hr  # using 1hr as representative dead fuel moisture
        M_x = fuel.extinction_moisture
        r_M = min(M_f / M_x, 1.0)
        eta_M = 1.0 - 2.59*r_M + 5.11*r_M**2 - 3.52*r_M**3
        eta_M = max(eta_M, 0.0)

        # Mineral damping (eq 30) - fixed value for standard fuels
        eta_s = 0.174 * (0.01)**(-0.19)  # ~0.417

        # Reaction intensity (BTU/ft2/min) (eq 27)
        h = 8000.0  # BTU/lb heat content
        I_R = gamma_op * w_n * h * eta_M * eta_s

        # Propagating flux ratio (eq 42)
        xi = math.exp((0.792 + 0.681 * sigma**0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)

        # Wind factor (eq 47)
        # Convert 20ft wind to midflame (multiply by 0.4 for open, then to ft/min)
        U = env.wind_speed * 0.4 * 88.0  # ft/min at midflame
        U = min(U, 300.0)  # hard cap at ~4mph midflame — Rothermel empirical limit
        B = 0.02526 * sigma**0.54
        C = 7.47 * math.exp(-0.133 * sigma**0.55)
        E = 0.715 * math.exp(-3.59e-4 * sigma)
        phi_w = C * (U**B) * (beta**(-E))
        # Wind speed limit — Rothermel caps wind effect
        # Max effective wind = 0.9 * I_R (empirical limit)
        U_max = 0.9 * I_R
        if U > U_max:
            U = U_max
            phi_w = C * (U**B) * (beta**(-E))

        # Slope factor (eq 51)
        phi_s = 5.275 * (beta**(-0.3)) * (env.slope / 100.0)**2

        # Heat sink (BTU/ft3) (eq 52)
        Q_ig = 250.0 + 1116.0 * env.moisture_1hr
        heat_sink = rho_b * Q_ig

        # Rate of spread (ft/min) (eq 1)
        R = (I_R * xi * (1.0 + phi_w + phi_s)) / max(heat_sink, 0.001)

        # Fireline intensity (BTU/ft/s) - Byram 1959
        # I = H * w * R (BTU/ft/s)
        I_B = (I_R * fuel.fuel_depth * R) / 60.0 / 10.0

        # Flame length (ft) - Byram 1959
        L = 0.45 * I_B**0.46 if I_B > 0 else 0.0

        # Direction of spread (downwind)
        direction = (env.wind_direction + 180.0) % 360.0

        return FireBehavior(
            spread_rate_fpm=R,
            spread_rate_mph=R * 60.0 / 5280.0,
            spread_rate_mpm=R * 0.3048,
            intensity=I_B,
            flame_length=L,
            direction=direction
        )

    def threat_level(self, b: FireBehavior) -> str:
        if b.spread_rate_mph > 3.0 or b.flame_length > 15.0:
            return "critical"
        elif b.spread_rate_mph > 1.0 or b.flame_length > 8.0:
            return "high"
        elif b.spread_rate_mph > 0.2 or b.flame_length > 2.0:
            return "medium"
        return "low"


def run_scenario(name, env):
    model = RothermelModel()
    b = model.calculate(env)
    threat = model.threat_level(b)
    fuel = FUEL_MODELS[env.fuel_model]
    print(f"\n{'='*52}")
    print(f"SCENARIO: {name}")
    print(f"{'='*52}")
    print(f"Fuel Model:   {env.fuel_model} — {fuel.name}")
    print(f"Wind:         {env.wind_speed}mph @ {env.wind_direction}°")
    print(f"Slope:        {env.slope}%")
    print(f"Moisture 1hr: {env.moisture_1hr*100:.0f}%")
    print(f"\n[ROTHERMEL OUTPUT]")
    print(b.summary())
    print(f"Threat Level: {threat.upper()}")


if __name__ == "__main__":
    print("🔥 ROTHERMEL SURFACE FIRE SPREAD MODEL")
    print("Rothermel 1972 — USDA Forest Service INT-115\n")

    # California Sierra Nevada — Dixie Fire conditions
    run_scenario("California Sierra Nevada — Extreme", FireEnvironment(
        fuel_model=9,
        wind_speed=25, wind_direction=225,
        slope=30, aspect=180,
        moisture_1hr=0.02, moisture_10hr=0.03,
        moisture_100hr=0.05, moisture_live=0.60
    ))

    # Texas grassland — high wind
    run_scenario("Texas Grassland — High Wind", FireEnvironment(
        fuel_model=1,
        wind_speed=35, wind_direction=270,
        slope=5, aspect=90,
        moisture_1hr=0.06, moisture_10hr=0.08,
        moisture_100hr=0.10, moisture_live=0.80
    ))

    # Australian chaparral — extreme
    run_scenario("Australian Bush — Critical", FireEnvironment(
        fuel_model=4,
        wind_speed=40, wind_direction=315,
        slope=20, aspect=270,
        moisture_1hr=0.03, moisture_10hr=0.04,
        moisture_100hr=0.06, moisture_live=0.50
    ))