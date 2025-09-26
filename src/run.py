# python
import os
import sys
import logging
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import imageio.v2 as imageio
from pathlib import Path

from parameters import Parameters
import overviews

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v

params = Parameters()
logger.info(f"Parameters: {params}")

# Load dataset
points, bbox_polygon = overviews.load_dataset(params.dataset_path)
logger.info(f"Loaded {points.shape[0]} points with {points.shape[1]} dimensions from {params.dataset_path}.")
logger.info(f"Bounding box in original coordinates: {bbox_polygon}")

# Compute bounding box center
min_vals = points.min(axis=0)
max_vals = points.max(axis=0)
center = (min_vals + max_vals) / 2

# Downsample
if len(points) > params.max_points:
    mask_random = np.random.choice(len(points), int(params.max_points), replace=False)
    logger.debug(f"Downsampled {len(points)} points to {params.max_points} points.")
else:
    mask_random = np.arange(len(points))
    logger.debug(f"Using all points for rendering as condition {len(points)} < {len(mask_random)} was not met.")

sampled_points = points[mask_random]

# Prepare colored point cloud
logger.debug("Converting numpy array to Open3D point cloud. If this is the last message, a segfault occurred in Vector3dVector due to Open3D issues.")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(sampled_points)
logger.debug("Successfully converted numpy array to Open3D point cloud. No segmentation fault was thrown.")

heights = sampled_points[:, 2]
colors = plt.get_cmap(params.cmap)((heights - heights.min()) / (heights.max() - heights.min()))[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

# GPU visualizer (headless window)
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=params.image_width, height=params.image_height)
vis.add_geometry(pcd)

ctr = vis.get_view_control()
opt = vis.get_render_option()
opt.background_color = np.array([1.0, 1.0, 1.0])  # white background
opt.point_size = 1.0

# Initialize camera roughly centered
ctr.set_lookat(center.tolist())
ctr.set_up([0, 0, 1])
ctr.set_front([1, 0, 0])
ctr.set_zoom(0.7)

def _capture_image():
    vis.poll_events()
    vis.update_renderer()
    img_float = vis.capture_screen_float_buffer(do_render=True)
    img_uint8 = (np.asarray(img_float) * 255).astype(np.uint8)
    return o3d.geometry.Image(img_uint8)

# Start rendering
start_time = time()

# Top views: orbit camera around Z with slight downward tilt
for i, angle in tqdm(enumerate(range(0, 360, params.top_views_deg)),
                     total=int(360 / params.top_views_deg),
                     desc="Rendering top views",
                     file=sys.stdout):
    rad = np.deg2rad(angle)
    # View from above with a gentle tilt for depth perception
    front = _normalize([np.cos(rad), np.sin(rad), 0.5])
    ctr.set_front(front.tolist())
    ctr.set_up([0, 0, 1])
    ctr.set_lookat(center.tolist())

    img = _capture_image()
    output_path = params.output_dir / f'top_view_{i:02d}.png'
    o3d.io.write_image(str(output_path), img)

# Section views with center‑based slicing and median fallback
for direction, axis_index, front in tqdm(zip(['ns', 'ew'], [0, 1], [[1, 0, 0], [0, 1, 0]]),
                                         total=2, file=sys.stdout):
    center_coord = center[axis_index]

    # Slice around center
    mask = (sampled_points[:, axis_index] > center_coord - params.section_width / 2) & \
           (sampled_points[:, axis_index] < center_coord + params.section_width / 2)

    # Fallback to median if empty
    if not np.any(mask):
        median_coord = np.median(sampled_points[:, axis_index])
        mask = (sampled_points[:, axis_index] > median_coord - params.section_width / 2) & \
               (sampled_points[:, axis_index] < median_coord + params.section_width / 2)
        if not np.any(mask):
            logger.warning(f"Skipping section view {direction} — no points in center or median slice.")
            continue
        center_coord = median_coord

    section_points = sampled_points[mask]
    section_pcd = o3d.geometry.PointCloud()
    section_pcd.points = o3d.utility.Vector3dVector(section_points)

    # Color along the slicing axis (consistent with old GPU method)
    color_values = section_points[:, axis_index]
    c = (color_values - color_values.min()) / max(1e-9, (color_values.max() - color_values.min()))
    section_colors = plt.get_cmap(params.cmap)(c)[:, :3]
    section_pcd.colors = o3d.utility.Vector3dVector(section_colors)

    # Swap geometry in the scene to render just the section
    vis.clear_geometries()
    vis.add_geometry(section_pcd)
    opt.point_size = 1.0

    # Camera aligned with requested axis
    ctr.set_front(front)
    # Keep Z as up
    ctr.set_up([0, 0, 1])
    # Look at the global center but align to the section center on that axis
    lookat = center.copy()
    lookat[axis_index] = center_coord
    ctr.set_lookat(lookat.tolist())
    # Slight zoom‑out to cover the slice
    ctr.set_zoom(0.8)

    img = _capture_image()
    output_path = params.output_dir / f'section_{direction}.png'
    o3d.io.write_image(str(output_path), img)

logger.info(f"Rendered {len(range(0, 360, params.top_views_deg))} top views and 2 section views in {time() - start_time:.2f} seconds.")

# Create overview GIF animation (unchanged)
images = []
for img_path in sorted(params.output_dir.glob("top_view_*.png")):
    images.append(imageio.imread(img_path))
imageio.mimsave(str(params.output_dir / "overview_round.gif"), images, duration=0.15, loop=0)
logger.debug(f"Created overview GIF animation at {params.output_dir / 'overview_round.gif'}")

# Debug: list outputs (unchanged)
logger.debug("=== DEBUG: Output directory contents ===")
logger.debug(f"Output directory: {params.output_dir}")
logger.debug(f"Output directory absolute: {params.output_dir.absolute()}")
logger.debug(f"Current working directory: {Path.cwd()}")
logger.debug("Files in output directory:")
for file_path in params.output_dir.iterdir():
    logger.debug(f"  {file_path.name}")

# Cleanup
vis.destroy_window()
os.chdir(params.output_dir)