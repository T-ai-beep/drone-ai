from pymavlink import mavutil
import time
import json

def connect():
    print("🚁 Connecting to SITL...")
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    master.wait_heartbeat()
    print(f"✅ Connected! (system {master.target_system})")
    return master

def set_mode(master, mode):
    mode_id = master.mode_mapping()[mode]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    time.sleep(1)

def arm(master):
    print("Arming...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    time.sleep(3)
    print("✅ Armed!")

def takeoff(master, altitude=10):
    print(f"🚁 Taking off to {altitude}m...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude
    )
    time.sleep(8)
    print("✅ Airborne!")

def land(master):
    print("🛬 Landing...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0
    )

def move(master, direction, speed=2, duration=2):
    print(f"➡️ Moving {direction}...")
    vx, vy, vz = 0, 0, 0
    if direction == "forward": vx = speed
    elif direction == "backward": vx = -speed
    elif direction == "left": vy = -speed
    elif direction == "right": vy = speed
    elif direction == "up": vz = -speed
    elif direction == "down": vz = speed

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    time.sleep(duration)

def scan(master, direction="area"):
    print(f"🔍 Scanning {direction}...")
    for yaw_deg in [30, -60, 30]:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            abs(yaw_deg),
            20,
            1 if yaw_deg > 0 else -1,
            1,
            0, 0, 0
        )
        time.sleep(2)
    print("✅ Scan complete")

def orbit(master):
    print("🔄 Orbiting...")
    import math
    speed = 2
    radius = 5
    duration = 15
    start = time.time()
    
    while time.time() - start < duration:
        t = time.time() - start
        angle = (t / duration) * 2 * math.pi
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        
        master.mav.set_position_target_local_ned_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, 0,
            0, 0, 0,
            0, 0
        )
        time.sleep(0.2)
    
    print("✅ Orbit complete")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_ORBIT,
        0,
        10,
        2,
        0,
        0,
        0, 0, 0
    )
    time.sleep(10)
    print("✅ Orbit complete")

def execute_command(master, cmd_json):
    cmd = cmd_json.get("command", "hover")
    direction = cmd_json.get("direction", "forward")

    if cmd == "takeoff":
        set_mode(master, "GUIDED")
        arm(master)
        takeoff(master)
    elif cmd == "land":
        land(master)
    elif cmd == "move":
        move(master, direction)
    elif cmd == "scan":
        scan(master, direction)
    elif cmd == "orbit":
        orbit(master)
    elif cmd == "hover":
        print("🚁 Hovering...")
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    master = connect()

    print("\n🎮 Drone simulator ready!")
    print("Commands: takeoff, land, scan, orbit, move")
    print('Examples:')
    print('  {"command": "takeoff"}')
    print('  {"command": "move", "direction": "forward"}')
    print('  {"command": "scan", "direction": "left"}')
    print('  {"command": "orbit"}')
    print('  {"command": "land"}\n')

    # Auto takeoff
    set_mode(master, "GUIDED")
    arm(master)
    takeoff(master)

    while True:
        try:
            line = input("Command: ")
            if not line.strip():
                continue
            cmd = json.loads(line)
            execute_command(master, cmd)
        except KeyboardInterrupt:
            print("\nShutting down...")
            land(master)
            break
        except json.JSONDecodeError:
            print("Invalid JSON. Example: {\"command\": \"scan\"}")
        except Exception as e:
            print(f"Error: {e}")