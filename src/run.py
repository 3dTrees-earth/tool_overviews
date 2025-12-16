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
import math
from open3d.visualization import rendering as r

from parameters import Parameters
import overviews

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v

params = Parameters()
logger.info(f"Parameters: {params}")

# Find all LAZ/LAS files in input directory
laz_files = sorted(
    list(params.input_dir.glob("*.laz")) + 
    list(params.input_dir.glob("*.las")) +
    list(params.input_dir.glob("*.LAZ")) + 
    list(params.input_dir.glob("*.LAS"))
)

if len(laz_files) == 0:
    raise ValueError(f"No LAZ/LAS files found in {params.input_dir}")

logger.info(f"Found {len(laz_files)} file(s) in collection: {[f.name for f in laz_files]}")

if len(laz_files) == 1:
    # Single file - load directly
    points = overviews.load_single_dataset(str(laz_files[0]))
    logger.info(f"Loaded {len(points):,} points from single file")
else:
    # Multiple files - aggregate with proportional sampling
    points = overviews.load_and_aggregate_datasets(
        [str(f) for f in laz_files],
        params.max_total_points
    )
    logger.info(f"Aggregated {len(laz_files)} files into {len(points):,} points")

# Compute bounding box center
min_vals = points.min(axis=0)
max_vals = points.max(axis=0)
center = (min_vals + max_vals) / 2

start_time = time()

# Prepare colored point cloud
logger.debug("Converting numpy array to Open3D point cloud. If this is the last message, a segfault occurred in Vector3dVector due to Open3D issues.")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
logger.debug("Successfully converted numpy array to Open3D point cloud. No segmentation fault was thrown.")

# Outlier-aware normalization using percentile clipping
heights = points[:, 2]
low, high = np.percentile(heights, [2, 98])
if not np.isfinite(low) or not np.isfinite(high) or np.isclose(high, low):
    low, high = heights.min(), heights.max()
norm = (np.clip(heights, low, high) - low) / max(1e-9, (high - low))
colors = plt.get_cmap(params.cmap)(norm)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

renderer = r.OffscreenRenderer(params.image_width, params.image_height)
scene = renderer.scene

# White background
scene.set_background([1.0, 1.0, 1.0, 1.0])

# Unlit point material (uses per-vertex colors)
mat = r.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = 1.0

# Add initial geometry
scene.add_geometry("cloud", pcd, mat)

# Camera helpers
bbox = pcd.get_axis_aligned_bounding_box()
center3 = np.asarray(bbox.get_center())
extent = np.linalg.norm(bbox.get_extent())
radius = max(1e-6, 0.5 * extent)
fov = 60.0
aspect = params.image_width / params.image_height
scene.camera.set_projection(fov, aspect, 0.1, 10000.0, r.Camera.FovType.Vertical)

def _look_at(front, up=np.array([0.0, 0.0, 1.0]), zoom=0.7, target=center3, rad=radius):
    f = _normalize(front)
    dist = (rad / math.tan(math.radians(fov / 2.0))) / max(1e-6, zoom)
    eye = np.asarray(target) - f * dist
    scene.camera.look_at(np.asarray(target).tolist(), eye.tolist(), np.asarray(up).tolist())

def _capture_image():
    return renderer.render_to_image()

# --- Top views (orbit) ---
for i, angle in enumerate(range(0, 360, params.top_views_deg)):
    rad = np.deg2rad(angle)
    front = _normalize([np.cos(rad), np.sin(rad), -0.5])
    _look_at(front, zoom=1, target=center3, rad=radius)
    img = _capture_image()
    o3d.io.write_image(str(params.output_dir / f"top_view_{i:02d}.png"), img)

# --- Section views (replace geometry, then frame and render) ---
for direction, axis_index, front in zip(["ns", "ew"], [0, 1], [[1, 0, 0], [0, 1, 0]]):
    center_coord = center[axis_index]
    mask = (points[:, axis_index] > center_coord - params.section_width / 2) & \
           (points[:, axis_index] < center_coord + params.section_width / 2)
    if not np.any(mask):
        median_coord = np.median(points[:, axis_index])
        mask = (points[:, axis_index] > median_coord - params.section_width / 2) & \
               (points[:, axis_index] < median_coord + params.section_width / 2)
        if not np.any(mask):
            logger.warning(f"Skipping section view {direction} — no points in center or median slice.")
            continue
        center_coord = median_coord

    section_points = points[mask]
    section_pcd = o3d.geometry.PointCloud()
    section_pcd.points = o3d.utility.Vector3dVector(section_points)

    color_values = section_points[:, axis_index]
    c = (color_values - color_values.min()) / max(1e-9, (color_values.max() - color_values.min()))
    section_colors = plt.get_cmap(params.cmap)(c)[:, :3]
    section_pcd.colors = o3d.utility.Vector3dVector(section_colors)

    scene.clear_geometry()
    scene.add_geometry("section", section_pcd, mat)

    section_bbox = section_pcd.get_axis_aligned_bounding_box()
    section_center = np.asarray(section_bbox.get_center())
    section_extent = np.linalg.norm(section_bbox.get_extent())
    section_radius = max(1e-6, 0.5 * section_extent)

    _look_at(front, up=[0, 0, 1], zoom=0.8, target=section_center, rad=section_radius)
    img = _capture_image()
    o3d.io.write_image(str(params.output_dir / f"section_{direction}.png"), img)

logger.info(f"Rendered {len(range(0, 360, params.top_views_deg))} top views and 2 section views in {time() - start_time:.2f} seconds.")

# Create overview GIF animation
images = []
for img_path in sorted(params.output_dir.glob("top_view_*.png")):
    images.append(imageio.imread(img_path))
imageio.mimsave(str(params.output_dir / "overview_round.gif"), images, duration=0.15, loop=0)
logger.debug(f"Created overview GIF animation at {params.output_dir / 'overview_round.gif'}")

# Debug: list outputs
logger.debug("=== DEBUG: Output directory contents ===")
logger.debug(f"Output directory: {params.output_dir}")
logger.debug(f"Output directory absolute: {params.output_dir.absolute()}")
logger.debug(f"Current working directory: {Path.cwd()}")
logger.debug("Files in output directory:")
for file_path in params.output_dir.iterdir():
    logger.debug(f"  {file_path.name}")

# Cleanup
os.chdir(params.output_dir)
