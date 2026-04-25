import open3d as o3d
import numpy as np
import cv2
import subprocess
import time
import json
import os

FRAME_PATH = "/tmp/mapping_frame.jpg"
POINTCLOUD_PATH = "/tmp/terrain.pcd"

def capture_frame():
    result = subprocess.run([
        'ffmpeg', '-f', 'avfoundation',
        '-pixel_format', 'uyvy422',
        '-framerate', '30',
        '-video_size', '1280x720',
        '-i', '0',
        '-vframes', '1',
        '-update', '1',
        FRAME_PATH, '-y'
    ], capture_output=True)
    return result.returncode == 0

def frame_to_pointcloud(frame_path, altitude=10.0, yaw=0.0):
    img = cv2.imread(frame_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    points = []
    colors = []
    
    # Sample every 10th pixel for performance
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            depth = (255 - gray[y, x]) / 255.0 * altitude
            
            # Convert pixel to world coordinates
            world_x = (x - w/2) * altitude / w
            world_y = (y - h/2) * altitude / h
            world_z = depth
            
            # Apply yaw rotation
            yaw_rad = np.radians(yaw)
            rx = world_x * np.cos(yaw_rad) - world_y * np.sin(yaw_rad)
            ry = world_x * np.sin(yaw_rad) + world_y * np.cos(yaw_rad)
            
            points.append([rx, ry, world_z])
            
            b, g, r = img[y, x]
            colors.append([r/255, g/255, b/255])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    return pcd

def merge_pointclouds(pcds):
    merged = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged += pcd
    
    # Downsample to keep it manageable
    merged = merged.voxel_down_sample(voxel_size=0.1)
    return merged

def save_pointcloud(pcd, path=POINTCLOUD_PATH):
    o3d.io.write_point_cloud(path, pcd)
    print(f"✅ Point cloud saved: {path} ({len(pcd.points)} points)")

def run_mapping_session(num_frames=10, altitude=10.0):
    print("🗺️ Starting terrain mapping session...")
    print(f"Capturing {num_frames} frames at {altitude}m altitude\n")
    
    pointclouds = []
    
    for i in range(num_frames):
        yaw = (i / num_frames) * 360
        print(f"Frame {i+1}/{num_frames} — yaw: {yaw:.0f}°")
        
        if capture_frame():
            pcd = frame_to_pointcloud(FRAME_PATH, altitude=altitude, yaw=yaw)
            if pcd:
                pointclouds.append(pcd)
                print(f"  ✓ {len(pcd.points)} points captured")
        else:
            print(f"  ✗ Frame capture failed")
        
        time.sleep(0.5)
    
    if not pointclouds:
        print("❌ No point clouds generated")
        return None
    
    print(f"\nMerging {len(pointclouds)} point clouds...")
    merged = merge_pointclouds(pointclouds)
    save_pointcloud(merged)
    
    print(f"\n✅ Mapping complete!")
    print(f"Total points: {len(merged.points)}")
    print(f"Saved to: {POINTCLOUD_PATH}")
    
    return merged

if __name__ == "__main__":
    pcd = run_mapping_session(num_frames=10, altitude=10.0)