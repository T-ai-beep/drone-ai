import requests
import json
from datetime import datetime
from dataclasses import dataclass

@dataclass
class WeatherConditions:
    temperature_f: float
    humidity: float          # percent
    wind_speed_mph: float
    wind_direction: float    # degrees
    precipitation: float     # inches last hour
    dewpoint_f: float
    station: str
    timestamp: str

    def to_rothermel_moisture(self) -> float:
        """
        Convert weather conditions to 1hr fuel moisture fraction.
        Uses NFDRS dead fuel moisture model simplified.
        """
        # Higher humidity = higher moisture
        # Higher temp = lower moisture
        base_moisture = self.humidity / 100.0 * 0.30
        temp_adjustment = (self.temperature_f - 70) * 0.001
        moisture = base_moisture - temp_adjustment
        return max(0.01, min(0.40, moisture))

    def summary(self):
        return (
            f"Station:     {self.station}\n"
            f"Time:        {self.timestamp}\n"
            f"Temperature: {self.temperature_f:.1f}°F\n"
            f"Humidity:    {self.humidity:.0f}%\n"
            f"Wind:        {self.wind_speed_mph:.1f}mph @ {self.wind_direction:.0f}°\n"
            f"Dewpoint:    {self.dewpoint_f:.1f}°F\n"
            f"Est. Fuel Moisture: {self.to_rothermel_moisture()*100:.1f}%"
        )

class NOAAFeed:
    """
    NOAA Weather API — free, no API key required.
    Pulls real-time weather observations for any US location.
    """

    BASE_URL = "https://api.weather.gov"

    def get_nearest_station(self, lat: float, lon: float) -> str:
        """Find nearest weather observation station to coordinates."""
        url = f"{self.BASE_URL}/points/{lat},{lon}"
        headers = {"User-Agent": "Beacon-Disaster-Response/1.0"}

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Get observation stations
        stations_url = data["properties"]["observationStations"]
        stations_response = requests.get(stations_url, headers=headers, timeout=10)
        stations_response.raise_for_status()
        stations_data = stations_response.json()

        # Return nearest station ID
        station = stations_data["features"][0]
        station_id = station["properties"]["stationIdentifier"]
        station_name = station["properties"]["name"]
        print(f"[NOAA] Nearest station: {station_id} — {station_name}")
        return station_id

    def get_current_conditions(self, lat: float, lon: float) -> WeatherConditions:
        """
        Fetch current weather conditions for any US coordinates.
        Returns WeatherConditions ready for Rothermel model input.
        """
        headers = {"User-Agent": "Beacon-Disaster-Response/1.0"}

        print(f"[NOAA] Fetching conditions for ({lat}, {lon})...")
        station_id = self.get_nearest_station(lat, lon)

        # Get latest observation
        obs_url = f"{self.BASE_URL}/stations/{station_id}/observations/latest"
        obs_response = requests.get(obs_url, headers=headers, timeout=10)
        obs_response.raise_for_status()
        obs = obs_response.json()["properties"]

        # Extract values with fallbacks
        temp_c = obs.get("temperature", {}).get("value") or 20
        temp_f = temp_c * 9/5 + 32

        humidity = obs.get("relativeHumidity", {}).get("value") or 50

        wind_ms = obs.get("windSpeed", {}).get("value") or 0
        wind_mph = wind_ms * 2.237

        wind_dir = obs.get("windDirection", {}).get("value") or 0

        dewpoint_c = obs.get("dewpoint", {}).get("value") or 10
        dewpoint_f = dewpoint_c * 9/5 + 32

        precip = obs.get("precipitationLastHour", {}).get("value") or 0
        precip_in = (precip or 0) * 0.0394

        timestamp = obs.get("timestamp", datetime.now().isoformat())

        return WeatherConditions(
            temperature_f=temp_f,
            humidity=humidity,
            wind_speed_mph=wind_mph,
            wind_direction=wind_dir,
            precipitation=precip_in,
            dewpoint_f=dewpoint_f,
            station=station_id,
            timestamp=timestamp
        )

    def get_fire_weather(self, lat: float, lon: float) -> dict:
        """
        Get weather conditions formatted for Rothermel fire modeling.
        Returns dict ready to pass directly into FireEnvironment.
        """
        conditions = self.get_current_conditions(lat, lon)
        moisture = conditions.to_rothermel_moisture()

        return {
            "wind_speed": conditions.wind_speed_mph,
            "wind_direction": conditions.wind_direction,
            "moisture_1hr": moisture,
            "moisture_10hr": min(moisture * 1.5, 0.40),
            "moisture_100hr": min(moisture * 2.0, 0.40),
            "moisture_live": min(moisture * 4.0, 1.50),
            "conditions": conditions
        }

if __name__ == "__main__":
    feed = NOAAFeed()

    # Test locations
    locations = [
        ("Dixie Fire Area — Plumas County CA", 40.1, -121.0),
        ("Allen TX — Home", 33.1, -96.6),
        ("Los Angeles CA", 34.05, -118.25),
    ]

    for name, lat, lon in locations:
        print(f"\n{'='*52}")
        print(f"LOCATION: {name}")
        print(f"{'='*52}")
        try:
            weather = feed.get_fire_weather(lat, lon)
            print(weather["conditions"].summary())
            print(f"\n[ROTHERMEL INPUTS]")
            print(f"Wind Speed:    {weather['wind_speed']:.1f}mph")
            print(f"Wind Dir:      {weather['wind_direction']:.0f}°")
            print(f"Moisture 1hr:  {weather['moisture_1hr']*100:.1f}%")
            print(f"Moisture 10hr: {weather['moisture_10hr']*100:.1f}%")
        except Exception as e:
            print(f"Error: {e}")