import cv2
import numpy as np
import subprocess
import time
import json
from collections import deque

FRAME_PATH = "/Users/tanayshah/drone/snapshot.jpg"

class SLAMSystem:
    def __init__(self):
        # Camera intrinsics - calibrate for your actual camera later
        self.fx = 718.8560
        self.fy = 718.8560
        self.cx = 607.1928
        self.cy = 185.2157
        
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # ORB feature detector
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
        )

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.pose = np.eye(4)  # Current camera pose
        self.trajectory = []   # List of positions
        self.map_points = []   # 3D map points
        self.frame_count = 0
        self.initialized = False

        print("🗺️ SLAM System initialized")
        print(f"   ORB features: 2000")
        print(f"   Camera matrix loaded")

    def capture_frame(self):
        # Use existing snapshot - swap this for live capture when camera is free
        frame = cv2.imread(FRAME_PATH, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            # Add slight noise to simulate camera movement between frames
            noise = np.random.randint(-3, 3, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return frame
        return None

    def extract_features(self, frame):
        kp, des = self.orb.detectAndCompute(frame, None)
        return kp, des

    def match_features(self, des1, des2):
        if des1 is None or des2 is None:
            return []
    
        if len(des1) < 2 or len(des2) < 2:
            return []
    
        matches = self.matcher.knnMatch(des1, des2, k=2)
    
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            elif len(match) == 1:
                good.append(match[0])
        
        return good

    def estimate_pose(self, kp1, kp2, matches):
        if len(matches) < 8:
            return None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None, None

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t

    def update_pose(self, R, t):
        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        # Update global pose
        self.pose = self.pose @ np.linalg.inv(T)
        
        # Extract position
        position = self.pose[:3, 3]
        self.trajectory.append(position.copy())
        
        return position

    def triangulate_points(self, kp1, kp2, matches, R, t):
        if len(matches) < 8:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T

        # Projection matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = self.K @ np.hstack([R, t])

        # Triangulate
        points4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points3D = points4D[:3] / points4D[3]

        return points3D.T

    def process_frame(self, frame):
        self.frame_count += 1
        kp, des = self.extract_features(frame)

        if not self.initialized:
            self.prev_frame = frame
            self.prev_kp = kp
            self.prev_des = des
            self.initialized = True
            print(f"[SLAM] Initialized with {len(kp)} features")
            return {
                "status": "initializing",
                "features": len(kp),
                "position": [0, 0, 0],
                "map_points": 0,
                "frame": self.frame_count
            }

        # Match features
        matches = self.match_features(self.prev_des, des)

        if len(matches) < 8:
            print(f"[SLAM] Not enough matches: {len(matches)}")
            self.prev_kp = kp
            self.prev_des = des
            return {
                "status": "tracking_lost",
                "features": len(kp),
                "matches": len(matches),
                "position": self.trajectory[-1].tolist() if self.trajectory else [0,0,0],
                "map_points": len(self.map_points),
                "frame": self.frame_count
            }

        # Estimate pose
        R, t = self.estimate_pose(self.prev_kp, kp, matches)

        if R is None:
            self.prev_kp = kp
            self.prev_des = des
            return {
                "status": "pose_failed",
                "features": len(kp),
                "matches": len(matches),
                "position": self.trajectory[-1].tolist() if self.trajectory else [0,0,0],
                "map_points": len(self.map_points),
                "frame": self.frame_count
            }

        # Update pose
        position = self.update_pose(R, t)

        # Triangulate new map points
        new_points = self.triangulate_points(self.prev_kp, kp, matches, R, t)
        if new_points is not None:
            self.map_points.extend(new_points.tolist())
            # Keep map manageable
            if len(self.map_points) > 10000:
                self.map_points = self.map_points[-10000:]

        result = {
            "status": "tracking",
            "features": len(kp),
            "matches": len(matches),
            "position": position.tolist(),
            "map_points": len(self.map_points),
            "frame": self.frame_count,
            "trajectory_length": len(self.trajectory)
        }

        print(f"[SLAM] Frame {self.frame_count} | "
              f"Features: {len(kp)} | "
              f"Matches: {len(matches)} | "
              f"Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) | "
              f"Map points: {len(self.map_points)}")

        # Update previous frame
        self.prev_frame = frame
        self.prev_kp = kp
        self.prev_des = des

        return result

    def get_map_summary(self):
        return {
            "total_frames": self.frame_count,
            "map_points": len(self.map_points),
            "trajectory_length": len(self.trajectory),
            "current_position": self.trajectory[-1].tolist() if self.trajectory else [0,0,0]
        }

def run_slam(num_frames=20):
    slam = SLAMSystem()
    results = []

    print(f"\n🚁 Starting SLAM session — {num_frames} frames\n")

    for i in range(num_frames):
        print(f"Capturing frame {i+1}/{num_frames}...")
        frame = slam.capture_frame()

        if frame is None:
            print("  ✗ Capture failed")
            time.sleep(0.5)
            continue

        result = slam.process_frame(frame)
        results.append(result)
        time.sleep(0.5)

    print("\n📊 SLAM Session Summary:")
    summary = slam.get_map_summary()
    print(f"  Total frames processed: {summary['total_frames']}")
    print(f"  Map points generated:   {summary['map_points']}")
    print(f"  Trajectory length:      {summary['trajectory_length']} poses")
    print(f"  Final position:         {[round(x,2) for x in summary['current_position']]}")

    return slam, results

if __name__ == "__main__":
    slam, results = run_slam(num_frames=20)