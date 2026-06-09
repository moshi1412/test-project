#!/usr/bin/env python3
"""
S3Gaussian 3D Scene Visualization Tool

This script loads a trained Gaussian model and visualizes the scene in 3D.
It supports:
- Interactive 3D point cloud visualization
- Camera trajectory visualization
- Dynamic/static decomposition visualization
- Multiple viewing modes

Usage:
    python visualize_scene.py --model_path /path/to/trained/model --source_path /path/to/data
"""

import os
import sys
import argparse
import torch
import numpy as np
import open3d as o3d
from argparse import Namespace

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from utils.general_utils import safe_state


def load_model_and_scene(args):
    """Load trained model and scene data."""
    print(f"Loading model from: {args.model_path}")
    print(f"Loading scene from: {args.source_path}")
    
    # Initialize Gaussian model
    gaussians = GaussianModel(args.sh_degree, args)
    
    # Load scene (this will load the trained point cloud)
    scene = Scene(
        args,
        gaussians,
        load_iteration=-1,  # Load latest iteration
        load_coarse=False,
        bg_gaussians=None,
        build_octree=False,
        build_grid=False,
        build_featgrid=False
    )
    
    return scene, gaussians


def extract_gaussian_points(gaussians, filter_opacity=True, opacity_threshold=0.1):
    """Extract Gaussian points with their colors and sizes."""
    print("Extracting Gaussian points...")
    
    # Get positions
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    # Get opacities and filter
    opacity = gaussians.get_opacity.detach().cpu().numpy()
    if filter_opacity:
        mask = opacity.squeeze() > opacity_threshold
        xyz = xyz[mask]
        print(f"Filtered {np.sum(~mask)} points with low opacity")
    
    # Get colors from SH features
    features_dc = gaussians._features_dc.detach().cpu().numpy()
    if filter_opacity:
        features_dc = features_dc[mask]
    
    # DC component gives the base color
    colors = features_dc[:, :, 0]  # [N, 3]
    colors = (colors + 1) / 2  # Normalize from [-1, 1] to [0, 1]
    colors = np.clip(colors, 0, 1)
    
    # Get scaling (size)
    scaling = gaussians._scaling.detach().cpu().numpy()
    if filter_opacity:
        scaling = scaling[mask]
    
    # Compute average scale for visualization
    scales = np.exp(scaling).mean(axis=1)  # Convert from log scale
    
    print(f"Total points: {xyz.shape[0]}")
    print(f"Point cloud bounds: min={xyz.min(axis=0)}, max={xyz.max(axis=0)}")
    
    return xyz, colors, scales


def create_point_cloud(xyz, colors, scales=None, point_size=3.0):
    """Create Open3D point cloud object."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Convert colors to Open3D format (RGB)
    colors = (colors * 255).astype(np.uint8)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # Estimate normals for better visualization
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    return pcd


def visualize_camera_trajectory(scene):
    """Extract camera trajectory from scene."""
    print("Extracting camera trajectory...")
    
    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    positions = []
    
    for cam in all_cameras:
        # Camera center is in world coordinates
        cam_center = cam.camera_center.cpu().numpy()
        positions.append(cam_center)
    
    positions = np.array(positions)
    
    # Create line set for trajectory
    lines = []
    for i in range(len(positions) - 1):
        lines.append([i, i + 1])
    
    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red trajectory
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(positions),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Create camera frustum markers
    camera_markers = []
    for i, cam in enumerate(all_cameras):
        if i % 10 == 0:  # Sample every 10 cameras
            center = cam.camera_center.cpu().numpy()
            # Create small sphere for camera position
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(center)
            sphere.paint_uniform_color([0, 1, 0])  # Green
            camera_markers.append(sphere)
    
    return line_set, camera_markers


def visualize_scene_3d(scene, gaussians, args):
    """Main visualization function."""
    print("Creating 3D visualization...")
    
    # Extract Gaussian points
    xyz, colors, scales = extract_gaussian_points(gaussians, 
                                                  filter_opacity=args.filter_opacity,
                                                  opacity_threshold=args.opacity_threshold)
    
    # Create point cloud
    pcd = create_point_cloud(xyz, colors, scales)
    
    # Create visualization elements
    vis_elements = [pcd]
    
    # Add camera trajectory if requested
    if args.show_camera_trajectory:
        trajectory, camera_markers = visualize_camera_trajectory(scene)
        vis_elements.append(trajectory)
        vis_elements.extend(camera_markers)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis_elements.append(coord_frame)
    
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="S3Gaussian 3D Scene Visualization", width=1280, height=720)
    
    # Add all elements
    for element in vis_elements:
        vis.add_geometry(element)
    
    # Set view parameters
    ctr = vis.get_view_control()
    # Look at the scene from a reasonable distance
    scene_center = xyz.mean(axis=0)
    ctr.set_lookat(scene_center)
    ctr.set_up([0, -1, 0])  # Y up (assuming Waymo coordinate system)
    ctr.set_front([-1, 0, 0])
    ctr.set_zoom(0.5)
    
    # Add help text
    print("\n=== 3D Visualization Controls ===")
    print("Mouse:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll: Zoom in/out")
    print("\nKeyboard:")
    print("  - ESC: Close window")
    print("  - H: Print help")
    print("  - R: Reset view")
    print("  - S: Save screenshot")
    print("=================================\n")
    
    # Register key callbacks
    def save_screenshot(vis):
        screenshot_path = os.path.join(args.model_path, "scene_visualization.png")
        vis.capture_screen_image(screenshot_path)
        print(f"Screenshot saved to: {screenshot_path}")
        return False
    
    def reset_view(vis):
        ctr = vis.get_view_control()
        ctr.set_lookat(scene_center)
        ctr.set_up([0, -1, 0])
        ctr.set_front([-1, 0, 0])
        ctr.set_zoom(0.5)
        return False
    
    vis.register_key_callback(ord("S"), save_screenshot)
    vis.register_key_callback(ord("R"), reset_view)
    
    # Run visualization
    print("Starting visualization... Press ESC to exit.")
    vis.run()
    vis.destroy_window()


def main():
    # Set up command line argument parser (same as training script)
    parser = argparse.ArgumentParser(description="S3Gaussian 3D Scene Visualization")
    
    # Use the same parameter groups as training script
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    # Visualization options
    parser.add_argument("--filter_opacity", action="store_true", default=True, 
                        help="Filter out low opacity points")
    parser.add_argument("--opacity_threshold", type=float, default=0.1,
                        help="Opacity threshold for filtering")
    parser.add_argument("--show_camera_trajectory", action="store_true", default=True,
                        help="Show camera trajectory")
    parser.add_argument("--quiet", action="store_true", default=False)
    
    args = parser.parse_args(sys.argv[1:])
    
    # Extract parameters using the same method as training script
    model_args = lp.extract(args)
    pipeline_args = pp.extract(args)
    hidden_args = hp.extract(args)
    
    # Merge all arguments
    args = Namespace(**{**vars(model_args), **vars(pipeline_args), **vars(hidden_args), **vars(args)})
    
    # Set random state
    safe_state(args.quiet)
    
    # Load model and scene
    scene, gaussians = load_model_and_scene(args)
    
    # Visualize
    visualize_scene_3d(scene, gaussians, args)


if __name__ == "__main__":
    main()