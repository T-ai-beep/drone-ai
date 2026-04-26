import numpy as np
import json
import time
import subprocess
from heapq import heappush, heappop

class Navigator:
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))  # 0 = free, 1 = obstacle
        self.current_position = np.array([50, 50])  # Start center
        self.target_position = None
        self.path = []
        self.step_index = 0
        self.directions_given = []

    def update_position(self, x, y):
        gx = int(np.clip(x * 5 + 50, 0, self.grid_size - 1))
        gy = int(np.clip(y * 5 + 50, 0, self.grid_size - 1))
        self.current_position = np.array([gx, gy])

    def mark_obstacle(self, x, y, radius=2):
        gx = int(np.clip(x * 5 + 50, 0, self.grid_size - 1))
        gy = int(np.clip(y * 5 + 50, 0, self.grid_size - 1))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.grid[nx, ny] = 1

    def set_target(self, x, y):
        gx = int(np.clip(x * 5 + 50, 0, self.grid_size - 1))
        gy = int(np.clip(y * 5 + 50, 0, self.grid_size - 1))
        self.target_position = np.array([gx, gy])

    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def astar(self, start, goal):
        open_set = []
        heappush(open_set, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.heuristic(start, goal)}

        while open_set:
            _, current = heappop(open_set)

            if current == tuple(goal):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(tuple(start))
                return path[::-1]

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.grid_size and
                        0 <= neighbor[1] < self.grid_size):
                    continue
                if self.grid[neighbor[0], neighbor[1]] == 1:
                    continue

                tentative_g = g_score[current] + self.heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def plan_path(self):
        if self.target_position is None:
            return []
        path = self.astar(self.current_position, self.target_position)
        self.path = path
        self.step_index = 0
        return path

    def grid_to_world(self, gx, gy):
        wx = (gx - 50) / 5.0
        wy = (gy - 50) / 5.0
        return wx, wy

    def get_direction(self, from_pos, to_pos):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        angle = np.degrees(np.arctan2(dy, dx))

        if -45 <= angle < 45:
            return "east"
        elif 45 <= angle < 135:
            return "south"
        elif angle >= 135 or angle < -135:
            return "west"
        else:
            return "north"

    def get_distance_feet(self, from_pos, to_pos):
        grid_dist = self.heuristic(from_pos, to_pos)
        # Each grid cell = 0.5 meters = ~1.64 feet
        feet = grid_dist * 1.64
        return round(feet)

    def generate_directions(self):
        if not self.path or len(self.path) < 2:
            return ["No path available. Stay in place and signal for help."]

        directions = []
        i = 0

        while i < len(self.path) - 1:
            current = self.path[i]
            direction = self.get_direction(current, self.path[i + 1])

            j = i + 1
            while j < len(self.path) - 1:
                next_dir = self.get_direction(self.path[j], self.path[j + 1])
                if next_dir != direction:
                    break
                j += 1

            distance = self.get_distance_feet(current, self.path[j])

            if direction == "north":
                directions.append(f"Head north for {distance} feet")
            elif direction == "south":
                directions.append(f"Head south for {distance} feet")
            elif direction == "east":
                directions.append(f"Turn right and walk {distance} feet east")
            elif direction == "west":
                directions.append(f"Turn left and walk {distance} feet west")

            i = j

        directions.append("You have reached the extraction point. Help is on the way.")
        return directions

    def speak_direction(self, text):
        print(f"[NAVIGATE] 🗣️  {text}")
        subprocess.run(['say', '-v', 'Samantha', text], capture_output=True)

    def run_navigation(self, person_x, person_y, safe_x, safe_y):
        print("🧭 NAVIGATION SYSTEM ACTIVE")
        print(f"   Person location: ({person_x}, {person_y})")
        print(f"   Safe zone: ({safe_x}, {safe_y})\n")

        self.update_position(person_x, person_y)
        self.set_target(safe_x, safe_y)

        # Simulate some terrain obstacles
        self.mark_obstacle(1, 0, radius=1)
        self.mark_obstacle(-1, 2, radius=1)
        self.mark_obstacle(2, 3, radius=2)

        path = self.plan_path()

        if not path:
            self.speak_direction("Warning. No clear path found. Stay in place and signal for help.")
            return

        print(f"[NAVIGATE] Path found: {len(path)} waypoints\n")

        directions = self.generate_directions()

        print("[NAVIGATE] Turn-by-turn directions:")
        for i, direction in enumerate(directions):
            print(f"  Step {i+1}: {direction}")

        print()
        self.speak_direction("Navigation active. Follow these directions to reach safety.")
        time.sleep(1)

        for direction in directions:
            self.speak_direction(direction)
            time.sleep(2)

if __name__ == "__main__":
    nav = Navigator()
    nav.run_navigation(
        person_x=0, person_y=0,
        safe_x=5, safe_y=5
    )