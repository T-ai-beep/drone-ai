import random
import time
import datetime
import math
import json
import os

SCAN_INTERVAL = 2
WIFI_FREQ = "2.4 GHz (WiFi)"
BT_FREQ = "2.4 GHz (Bluetooth)"
SIGNAL_THRESHOLD = -90
RESULTS_DIR = "sniffer_results"

# Allen, TX coords
DRONE_LAT = 33.1032
DRONE_LON = -96.6706


def generate_fake_mac() -> str:
    octets = [format(random.randint(0, 255), '02X') for _ in range(6)]
    octets[0] = format(random.randint(0, 255) & 0xFE, '02X')  # ensure unicast
    return ':'.join(octets)


def dbm_to_distance(dbm: float) -> float:
    """
    Free-space path loss inverse at 2.4 GHz.
    FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4pi/c)
    At 2.4 GHz, constant ~= 100 dB at 1m.
    d = 10 ^ ((TX_power - RSSI - path_loss_at_1m) / 20)
    Using TX_power = -50 dBm reference at 1m.
    """
    reference_dbm = -50.0   # signal at ~1m
    path_loss_exp = 20.0    # free space
    distance = 10 ** ((reference_dbm - dbm) / path_loss_exp)
    return round(max(0.5, distance), 2)


def offset_coords(lat: float, lon: float, distance_m: float) -> tuple[float, float]:
    """Return a lat/lon offset from drone position by distance_m in a random direction."""
    bearing = random.uniform(0, 2 * math.pi)
    earth_radius_m = 6_371_000
    delta = distance_m / earth_radius_m
    new_lat = math.degrees(
        math.asin(
            math.sin(math.radians(lat)) * math.cos(delta) +
            math.cos(math.radians(lat)) * math.sin(delta) * math.cos(bearing)
        )
    )
    new_lon = lon + math.degrees(
        math.atan2(
            math.sin(bearing) * math.sin(delta) * math.cos(math.radians(lat)),
            math.cos(delta) - math.sin(math.radians(lat)) * math.sin(math.radians(new_lat))
        )
    )
    return round(new_lat, 6), round(new_lon, 6)


def signal_to_confidence(dbm: float) -> float:
    """Stronger signal = higher confidence in position estimate."""
    normalized = (dbm - SIGNAL_THRESHOLD) / (-50 - SIGNAL_THRESHOLD)
    return round(min(1.0, max(0.0, normalized)), 2)


def create_device_pool(drone_lat: float, drone_lon: float, size: int = 8) -> list[dict]:
    """
    Create a persistent set of devices that exist in the area for the full scan session.
    Each has a fixed true position and base signal; detections will be noisy samples of these.
    """
    pool = []
    for _ in range(size):
        distance = random.uniform(2, 80)
        true_lat, true_lon = offset_coords(drone_lat, drone_lon, distance)
        pool.append({
            "mac": generate_fake_mac(),
            "true_lat": true_lat,
            "true_lon": true_lon,
            "base_signal_dbm": random.uniform(-85, -55),
            "frequency": random.choice([WIFI_FREQ, BT_FREQ]),
        })
    return pool


def simulate_scan(drone_lat: float, drone_lon: float, device_pool: list[dict]) -> list[dict]:
    detections = []

    for device in device_pool:
        if random.random() > 0.30:  # 30% detection chance per device per scan
            continue

        noisy_signal = device["base_signal_dbm"] + random.uniform(-3, 3)
        noisy_signal = round(max(-90.0, min(-50.0, noisy_signal)), 1)

        distance = dbm_to_distance(noisy_signal)

        # Small positional noise: offset true position by 0-2m in a random direction
        noise_dist = random.uniform(0, 2)
        noisy_lat, noisy_lon = offset_coords(device["true_lat"], device["true_lon"], noise_dist)

        detections.append({
            "mac_address": device["mac"],
            "signal_strength_dbm": noisy_signal,
            "frequency": device["frequency"],
            "timestamp": datetime.datetime.now().isoformat(),
            "estimated_distance_m": distance,
            "lat": noisy_lat,
            "lon": noisy_lon,
            "confidence": signal_to_confidence(noisy_signal),
        })

    # 10% chance of a completely spurious noise device per scan
    if random.random() < 0.10:
        dbm = random.uniform(-90, -50)
        distance = dbm_to_distance(dbm)
        est_lat, est_lon = offset_coords(drone_lat, drone_lon, distance)
        detections.append({
            "mac_address": generate_fake_mac(),
            "signal_strength_dbm": round(dbm, 1),
            "frequency": random.choice([WIFI_FREQ, BT_FREQ]),
            "timestamp": datetime.datetime.now().isoformat(),
            "estimated_distance_m": distance,
            "lat": est_lat,
            "lon": est_lon,
            "confidence": signal_to_confidence(dbm),
        })

    return detections


def run_sniffer(drone_lat: float, drone_lon: float, duration_seconds: int) -> tuple[list, dict]:
    all_detections = []
    unique_devices: dict[str, list[dict]] = {}

    device_pool = create_device_pool(drone_lat, drone_lon)
    scans = duration_seconds // SCAN_INTERVAL
    print(f"[BEACON] Device pool: {len(device_pool)} persistent devices seeded in area")
    print(f"[BEACON] Running {scans} scans over {duration_seconds}s at ({drone_lat}, {drone_lon})")

    for i in range(scans):
        detections = simulate_scan(drone_lat, drone_lon, device_pool)
        for d in detections:
            all_detections.append(d)
            mac = d["mac_address"]
            if mac not in unique_devices:
                unique_devices[mac] = []
            unique_devices[mac].append(d)

        seen = len(unique_devices)
        print(f"  Scan {i+1:02d}/{scans} — {len(detections)} device(s) detected | {seen} unique so far")
        time.sleep(SCAN_INTERVAL)

    return all_detections, unique_devices


def triangulate(device_detections: list[dict]) -> tuple[float, float, float]:
    if len(device_detections) == 1:
        d = device_detections[0]
        return d["lat"], d["lon"], d["confidence"] * 0.5  # single reading = low confidence

    lats = [d["lat"] for d in device_detections]
    lons = [d["lon"] for d in device_detections]
    confidences = [d["confidence"] for d in device_detections]

    avg_lat = sum(lats) / len(lats)
    avg_lon = sum(lons) / len(lons)
    # More readings = higher triangulation confidence (caps at 0.95)
    base_conf = sum(confidences) / len(confidences)
    reading_bonus = min(0.2, (len(device_detections) - 1) * 0.05)
    final_conf = round(min(0.95, base_conf + reading_bonus), 2)

    return round(avg_lat, 6), round(avg_lon, 6), final_conf


def is_likely_mobile(detections: list[dict]) -> bool:
    """
    Heuristic: if position estimates vary > 3m across detections, device is likely moving.
    Parked cars and static APs won't drift. Phones do.
    """
    if len(detections) < 2:
        return True  # can't tell, assume mobile
    lats = [d["lat"] for d in detections]
    lons = [d["lon"] for d in detections]
    lat_spread_m = (max(lats) - min(lats)) * 111_000
    lon_spread_m = (max(lons) - min(lons)) * 85_000  # approx at 33° lat
    return (lat_spread_m + lon_spread_m) > 3.0


def generate_report(unique_devices: dict) -> list[dict]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    survivor_candidates = []

    for mac, detections in unique_devices.items():
        lat, lon, conf = triangulate(detections)
        mobile = is_likely_mobile(detections)
        avg_signal = sum(d["signal_strength_dbm"] for d in detections) / len(detections)
        avg_dist = sum(d["estimated_distance_m"] for d in detections) / len(detections)

        candidate = {
            "mac_address": mac,
            "estimated_lat": lat,
            "estimated_lon": lon,
            "triangulation_confidence": conf,
            "likely_mobile_device": mobile,
            "detection_count": len(detections),
            "avg_signal_dbm": round(avg_signal, 1),
            "avg_distance_m": round(avg_dist, 1),
            "frequencies_seen": list({d["frequency"] for d in detections}),
            "first_seen": detections[0]["timestamp"],
            "last_seen": detections[-1]["timestamp"],
        }

        if mobile:
            survivor_candidates.append(candidate)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESULTS_DIR, f"beacon_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(),
            "drone_position": {"lat": DRONE_LAT, "lon": DRONE_LON},
            "total_unique_devices": len(unique_devices),
            "survivor_candidates": survivor_candidates,
        }, f, indent=2)

    print(f"\n[BEACON] Report saved → {report_path}")
    return survivor_candidates


def main():
    print("=" * 52)
    print("  BEACON RF Sniffer — Search & Rescue Simulator")
    print("=" * 52)
    print(f"  Drone position : {DRONE_LAT}°N, {DRONE_LON}°W  (Allen, TX)")
    print(f"  Scan interval  : {SCAN_INTERVAL}s")
    print(f"  Signal floor   : {SIGNAL_THRESHOLD} dBm")
    print("=" * 52 + "\n")

    all_detections, unique_devices = run_sniffer(DRONE_LAT, DRONE_LON, duration_seconds=60)

    print(f"\n[BEACON] Scan complete — {len(all_detections)} total detections, {len(unique_devices)} unique MACs")

    candidates = generate_report(unique_devices)

    print(f"\n{'='*52}")
    print(f"  SURVIVOR CANDIDATES  ({len(candidates)} found)")
    print(f"{'='*52}")

    if not candidates:
        print("  No mobile devices detected. Area may be clear.")
    else:
        for i, c in enumerate(candidates, 1):
            print(f"\n  [{i}] MAC: {c['mac_address']}")
            print(f"      Position : {c['estimated_lat']}°N, {c['estimated_lon']}°W")
            print(f"      Distance : ~{c['avg_distance_m']}m from drone")
            print(f"      Signal   : {c['avg_signal_dbm']} dBm avg")
            print(f"      Readings : {c['detection_count']} detection(s)")
            print(f"      Confidence: {int(c['triangulation_confidence'] * 100)}%")
            print(f"      Freq     : {', '.join(c['frequencies_seen'])}")

    print(f"\n{'='*52}\n")


if __name__ == "__main__":
    main()