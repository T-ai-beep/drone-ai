import subprocess
import os
import json
import tempfile
import math
from dataclasses import dataclass
from typing import List, Dict, Optional

WINDNINJA_CLI = os.path.expanduser("~/windninja/build/src/cli/WindNinja_cli")
WINDNINJA_DATA = os.path.expanduser("~/windninja/data")

@dataclass
class WindPoint:
    lat: float
    lon: float
    speed_mph: float
    direction_deg: float
    x_component: float   # u - east/west
    y_component: float   # v - north/south

@dataclass
class WindNinjaResult:
    center_lat: float
    center_lon: float
    input_speed_mph: float
    input_direction_deg: float
    wind_points: List[WindPoint]
    max_speed_mph: float
    min_speed_mph: float
    avg_speed_mph: float
    terrain_amplification: float  # max/input ratio
    dangerous_zones: List[Dict]
    beacon_deploy: str

    def summary(self):
        lines = [
            f"Input Wind:    {self.input_speed_mph:.1f}mph @ {self.input_direction_deg:.0f}°",
            f"Terrain Max:   {self.max_speed_mph:.1f}mph",
            f"Terrain Min:   {self.min_speed_mph:.1f}mph",
            f"Amplification: {self.terrain_amplification:.2f}x",
            f"Grid Points:   {len(self.wind_points)}",
            f"",
            f"DANGEROUS ZONES:",
        ]
        for zone in self.dangerous_zones:
            lines.append(
                f"  [{zone['threat'].upper():8s}] {zone['label']} — "
                f"{zone['speed_mph']:.1f}mph"
            )
        lines.append(f"\nDeploy: {self.beacon_deploy}")
        return "\n".join(lines)


class WindNinjaWrapper:
    """
    WindNinja US Forest Service Hyper-Local Wind Model.
    
    WindNinja uses computational fluid dynamics to model
    how terrain affects wind at meter-level resolution.
    Same tool used by US Forest Service for fire weather.
    
    Uses fetch_elevation to download DEM automatically
    then runs mass-conserving wind simulation.
    """

    def __init__(self):
        self.cli = os.path.expanduser(WINDNINJA_CLI)
        if not os.path.exists(self.cli):
            raise FileNotFoundError(f"WindNinja not found at {self.cli}")

    def _download_elevation(
        self,
        lat: float,
        lon: float,
        buffer_miles: float,
        output_path: str
    ) -> str:
        """Download elevation data for a region."""
        dem_path = os.path.join(output_path, "terrain.tif")

        cmd = [
            self.cli,
            "--fetch_elevation", dem_path,
            "--x_center", str(lon),
            "--y_center", str(lat),
            "--x_buffer", str(buffer_miles),
            "--y_buffer", str(buffer_miles),
            "--buffer_units", "miles",
            "--elevation_source", "srtm",
        ]

        print(f"[WINDNINJA] Downloading elevation data for ({lat}, {lon})...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if os.path.exists(dem_path):
            print(f"[WINDNINJA] Elevation downloaded: {dem_path}")
            return dem_path
        else:
            print(f"[WINDNINJA] Download failed, using synthetic DEM")
            return self._create_synthetic_dem(lat, lon, output_path)

    def _create_synthetic_dem(
    self,
    lat: float,
    lon: float,
    output_path: str
) -> str:
        """Create synthetic DEM in UTM projection using GDAL."""
        import subprocess
        
        dem_path = os.path.join(output_path, "terrain.tif")
        
        # Determine UTM zone
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        
        # Create synthetic elevation CSV
        ncols, nrows = 50, 50
        cellsize = 100  # meters
        
        # Center in UTM (approximate)
        utm_x = (lon + 180) / 6 * 1000000
        utm_y = lat * 111320
        
        xll = utm_x - (ncols * cellsize / 2)
        yll = utm_y - (nrows * cellsize / 2)
        
        asc_path = os.path.join(output_path, "terrain_wgs.asc")
        
        with open(asc_path, 'w') as f:
            f.write(f"ncols         {ncols}\n")
            f.write(f"nrows         {nrows}\n")
            f.write(f"xllcorner     {xll:.2f}\n")
            f.write(f"yllcorner     {yll:.2f}\n")
            f.write(f"cellsize      {cellsize}\n")
            f.write(f"NODATA_value  -9999\n")
            for i in range(nrows):
                row = []
                for j in range(ncols):
                    base = 500
                    ridge = 200 * math.exp(-((j-ncols*0.4)**2)/(2*(ncols*0.1)**2))
                    valley = -100 * math.exp(-((i-nrows*0.6)**2)/(2*(nrows*0.15)**2))
                    noise = 10 * math.sin(i*0.5) * math.cos(j*0.3)
                    row.append(base + ridge + valley + noise)
                f.write(" ".join(f"{v:.1f}" for v in row) + "\n")
        
        # Assign UTM projection using gdal_translate
        result = subprocess.run([
            "gdal_translate",
            "-a_srs", f"EPSG:{epsg}",
            asc_path, dem_path
        ], capture_output=True, text=True)
        
        if os.path.exists(dem_path):
            print(f"[WINDNINJA] Synthetic UTM DEM created: {dem_path}")
            return dem_path
        else:
            print(f"[WINDNINJA] GDAL failed: {result.stderr}")
            return asc_path

    def _run_simulation(
        self,
        dem_path: str,
        wind_speed_mph: float,
        wind_direction: float,
        output_path: str,
        vegetation: str = "trees",
        mesh_choice: str = "coarse"
    ) -> bool:
        """Run WindNinja mass-conserving wind simulation."""

        points_file = os.path.join(output_path, "output_points.csv")

        cmd = [
            self.cli,
            "--elevation_file", dem_path,
            "--initialization_method", "domainAverageInitialization",
            "--input_speed", str(wind_speed_mph),
            "--input_speed_units", "mph",
            "--input_direction", str(wind_direction),
            "--input_wind_height", "20",
            "--units_input_wind_height", "ft",
            "--output_wind_height", "20",
            "--units_output_wind_height", "ft",
            "--vegetation", vegetation,
            "--mesh_choice", mesh_choice,
            "--write_ascii_output", "true",
            "--ascii_out_geog", "true",
            "--output_path", output_path,
            "--num_threads", "4",
            "--diurnal_winds", "false",
        ]

        print(f"[WINDNINJA] Running simulation: {wind_speed_mph}mph @ {wind_direction}°")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )

        if result.returncode == 0:
            print("[WINDNINJA] Simulation complete")
            return True
        else:
            print(f"[WINDNINJA] Simulation error: {result.stderr[-500:]}")
            return False

    def _parse_ascii_output(
        self,
        output_path: str,
        center_lat: float,
        center_lon: float
    ) -> List[WindPoint]:
        """Parse WindNinja ASCII grid output files."""
        points = []

        # Find output files
        speed_file = None
        dir_file = None

        for f in os.listdir(output_path):
            if f.endswith("_vel.asc") or f.endswith("_spd.asc"):
                speed_file = os.path.join(output_path, f)
            elif f.endswith("_ang.asc") or f.endswith("_dir.asc"):
                dir_file = os.path.join(output_path, f)

        if not speed_file:
            print("[WINDNINJA] No output files found — using synthetic output")
            return self._synthetic_wind_field(center_lat, center_lon, 10.0, 270.0)

        try:
            # Parse speed grid
            speeds = self._parse_asc(speed_file)
            directions = self._parse_asc(dir_file) if dir_file else None

            header = speeds["header"]
            ncols = header["ncols"]
            nrows = header["nrows"]
            xll = header["xllcorner"]
            yll = header["yllcorner"]
            cell = header["cellsize"]

            for i, row in enumerate(speeds["data"]):
                for j, speed in enumerate(row):
                    if speed == header.get("nodata", -9999):
                        continue

                    lat = yll + (nrows - i) * cell
                    lon = xll + j * cell
                    direction = directions["data"][i][j] if directions else 270.0

                    u = -speed * math.sin(math.radians(direction))
                    v = -speed * math.cos(math.radians(direction))

                    points.append(WindPoint(
                        lat=lat, lon=lon,
                        speed_mph=speed,
                        direction_deg=direction,
                        x_component=u,
                        y_component=v
                    ))

        except Exception as e:
            print(f"[WINDNINJA] Parse error: {e} — using synthetic output")
            return self._synthetic_wind_field(center_lat, center_lon, 10.0, 270.0)

        return points

    def _parse_asc(self, filepath: str) -> dict:
        """Parse ESRI ASCII raster file."""
        header = {}
        data = []

        with open(filepath) as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line[0].isdigit() or line[0] == '-':
                break
            parts = line.split()
            if len(parts) == 2:
                try:
                    header[parts[0].lower()] = float(parts[1])
                except:
                    pass
            i += 1

        for line in lines[i:]:
            row = []
            for val in line.strip().split():
                try:
                    row.append(float(val))
                except:
                    pass
            if row:
                data.append(row)

        return {"header": header, "data": data}

    def _synthetic_wind_field(
        self,
        center_lat: float,
        center_lon: float,
        base_speed: float,
        base_direction: float
    ) -> List[WindPoint]:
        """
        Generate synthetic wind field with terrain effects.
        Used as fallback when WindNinja output parsing fails.
        Simulates channeling, acceleration over ridges, deceleration in valleys.
        """
        points = []
        grid_size = 10
        step = 0.01  # degrees (~1km)

        for i in range(-grid_size, grid_size + 1):
            for j in range(-grid_size, grid_size + 1):
                lat = center_lat + i * step
                lon = center_lon + j * step

                # Simulate terrain effects
                # Ridge effect — wind accelerates over high ground
                dist_from_center = math.sqrt(i**2 + j**2)
                ridge_factor = 1.0 + 0.3 * math.exp(-dist_from_center / 5)

                # Valley channeling — wind direction shifts in valleys
                channel_shift = 15 * math.sin(i * 0.3) * math.exp(-dist_from_center / 8)

                speed = base_speed * ridge_factor
                direction = (base_direction + channel_shift) % 360

                u = -speed * math.sin(math.radians(direction))
                v = -speed * math.cos(math.radians(direction))

                points.append(WindPoint(
                    lat=lat, lon=lon,
                    speed_mph=speed,
                    direction_deg=direction,
                    x_component=u,
                    y_component=v
                ))

        return points

    def _identify_dangerous_zones(
        self,
        points: List[WindPoint],
        input_speed: float
    ) -> List[Dict]:
        """Identify zones with dangerous wind conditions."""
        zones = []

        if not points:
            return zones

        speeds = [p.speed_mph for p in points]
        max_speed = max(speeds)
        p90 = sorted(speeds)[int(len(speeds) * 0.9)]

        # Find acceleration zones
        high_points = [p for p in points if p.speed_mph >= p90]

        if high_points:
            center_lat = sum(p.lat for p in high_points) / len(high_points)
            center_lon = sum(p.lon for p in high_points) / len(high_points)
            avg_high = sum(p.speed_mph for p in high_points) / len(high_points)

            if avg_high > input_speed * 1.5:
                threat = "critical"
            elif avg_high > input_speed * 1.25:
                threat = "high"
            else:
                threat = "medium"

            zones.append({
                "label": "Wind acceleration zone — ridge/canyon effect",
                "lat": center_lat,
                "lon": center_lon,
                "speed_mph": avg_high,
                "threat": threat,
                "deploy": "avoid — dangerous for drone operation"
            })

        # Identify channeling zones (large direction deviation)
        directions = [p.direction_deg for p in points]
        avg_dir = sum(directions) / len(directions)
        deviations = [abs(p.direction_deg - avg_dir) for p in points]
        high_deviation = [p for p, d in zip(points, deviations) if d > 30]

        if high_deviation:
            zones.append({
                "label": "Wind channeling zone — direction unstable",
                "lat": high_deviation[0].lat,
                "lon": high_deviation[0].lon,
                "speed_mph": sum(p.speed_mph for p in high_deviation) / len(high_deviation),
                "threat": "medium",
                "deploy": "caution — unpredictable wind direction"
            })

        return zones

    def calculate(
        self,
        lat: float,
        lon: float,
        wind_speed_mph: float,
        wind_direction: float,
        vegetation: str = "trees",
        buffer_miles: float = 1.0,
        download_dem: bool = True
    ) -> WindNinjaResult:
        """
        Run WindNinja terrain wind simulation.

        Args:
            lat, lon: Center of simulation area
            wind_speed_mph: Input wind speed
            wind_direction: Input wind direction (degrees FROM)
            vegetation: grass, brush, or trees
            buffer_miles: Radius of simulation area
            download_dem: Download real elevation data (requires internet)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Get elevation data
            if download_dem:
                dem_path = self._download_elevation(lat, lon, buffer_miles, tmpdir)
            else:
                dem_path = self._create_synthetic_dem(lat, lon, tmpdir)

            # Run simulation
            success = self._run_simulation(
                dem_path, wind_speed_mph, wind_direction, tmpdir, vegetation
            )

            # Parse output
            if success:
                points = self._parse_ascii_output(tmpdir, lat, lon)
            else:
                print("[WINDNINJA] Using synthetic wind field")
                points = self._synthetic_wind_field(lat, lon, wind_speed_mph, wind_direction)

            if not points:
                points = self._synthetic_wind_field(lat, lon, wind_speed_mph, wind_direction)

            # Calculate statistics
            speeds = [p.speed_mph for p in points]
            max_speed = max(speeds)
            min_speed = min(speeds)
            avg_speed = sum(speeds) / len(speeds)
            amplification = max_speed / max(wind_speed_mph, 0.1)

            # Identify dangerous zones
            dangerous = self._identify_dangerous_zones(points, wind_speed_mph)

            # Deployment recommendation
            if max_speed > 25:
                deploy = "GROUND ONLY — wind too strong for safe drone operation"
            elif max_speed > 15:
                deploy = "DRONE with caution — avoid acceleration zones"
            else:
                deploy = "DRONE safe — terrain wind within operational limits"

            return WindNinjaResult(
                center_lat=lat,
                center_lon=lon,
                input_speed_mph=wind_speed_mph,
                input_direction_deg=wind_direction,
                wind_points=points,
                max_speed_mph=max_speed,
                min_speed_mph=min_speed,
                avg_speed_mph=avg_speed,
                terrain_amplification=amplification,
                dangerous_zones=dangerous,
                beacon_deploy=deploy
            )

    def beacon_priority_zones(self, result: WindNinjaResult) -> dict:
        """Convert WindNinja result to Beacon priority zones."""
        zones = {}

        for zone in result.dangerous_zones:
            color = {"critical": "red", "high": "orange", "medium": "yellow"}.get(
                zone["threat"], "yellow"
            )
            zones[color] = {
                "priority": {"red": 1, "orange": 2, "yellow": 3}[color],
                "lat": zone["lat"],
                "lon": zone["lon"],
                "speed_mph": zone["speed_mph"],
                "label": zone["label"],
                "deploy": zone["deploy"]
            }

        return zones


if __name__ == "__main__":
    model = WindNinjaWrapper()

    print("🌬️  WINDNINJA HYPER-LOCAL TERRAIN WIND MODEL")
    print("US Forest Service Computational Fluid Dynamics\n")

    scenarios = [
        ("Plumas County CA — Wildfire Conditions", 40.1, -121.4, 25, 225, "trees"),
        ("Allen TX — Tornado Alley", 33.1, -96.6, 35, 200, "grass"),
        ("Yosemite Valley — Canyon Wind", 37.74, -119.59, 20, 270, "trees"),
    ]

    for name, lat, lon, speed, direction, veg in scenarios:
        print(f"\n{'='*55}")
        print(f"SCENARIO: {name}")
        print(f"{'='*55}")
        result = model.calculate(
            lat, lon, speed, direction, veg,
            buffer_miles=0.5,
            download_dem=True
        )
        print(result.summary())