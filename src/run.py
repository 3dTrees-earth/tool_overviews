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


params = Parameters()
logger.info(f"Parameters: {params}")


# Tool
points, bbox_polygon =overviews.load_dataset(params.dataset_path)
logger.info(f"Loaded {points.shape[0]} points with {points.shape[1]} dimensions from {params.dataset_path}.")
logger.info(f"Bounding box in original coordinates: {bbox_polygon}")

# Compute bounding box center
min_vals = points.min(axis=0)
max_vals = points.max(axis=0)
center = (min_vals + max_vals) / 2

# Generate mask for random downsampling
if len(points) > params.max_points:
    mask_random = np.random.choice(len(points), int(params.max_points), replace=False)
    logger.debug(f"Downsampled {len(points)} points to {params.max_points} points.")
else:
    mask_random = np.arange(len(points))
    logger.debug(f"Using all points for rendering as condition {len(points)} < {len(mask_random)} was not met.")

sampled_points = points[mask_random]

# Convert to Open3D point cloud
logger.debug(f"Converting numpy array to Open3D point cloud. If this this the last message, you got a segmentation fault when using Vector3dVector. This is a known issue in Open3D and usually the open3d version is not correct.")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(sampled_points)
logger.debug(f"Successfully converted numpy array to Open3D point cloud. No segmentation fault was thrown.")


heights = sampled_points[:, 2]
colors = plt.get_cmap(params.cmap)((heights - heights.min()) / (heights.max() - heights.min()))[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Build a headless renderer to capture the scene
# I might well have screwed somthing up in here
renderer = o3d.visualization.rendering.OffscreenRenderer(params.image_width, params.image_height)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = 1.0
logger.info(f"Setting up renderer with image width {params.image_width} and image height {params.image_height}: {renderer}")
renderer.scene.add_geometry("pcd", pcd, mat)
renderer.scene.set_background([1, 1, 1, 1])  # white background


# Now we get started
start_time = time()

for i, angle in tqdm(enumerate(range(0, 360, params.top_views_deg)), total=360 / params.top_views_deg, desc="Rendering top views", file=sys.stdout):
    img = overviews.render_top_down_views(renderer, angle, center, params.camera_distance)
    output_path = params.output_dir / f'top_view_{i:02d}.png'
    o3d.io.write_image(str(output_path), img)


for direction, axis_index, front in tqdm(zip(['ns', 'ew'], [0, 1], [[1, 0, 0], [0, 1, 0]]), total=2, file=sys.stdout):
    img = overviews.render_section_views(renderer,points, axis_index, direction, front, mat, center, params.section_width)
    
    output_path = params.output_dir / f'section_{direction}.png'
    o3d.io.write_image(str(output_path), img)

logger.info(f"Rendered {len(range(0, 360, params.top_views_deg))} top views and 2 section views in {time() - start_time:.2f} seconds.")


# Create overview GIF animation
images = []
for img_path in sorted(params.output_dir.glob("top_view_*.png")):
    images.append(imageio.imread(img_path))

imageio.mimsave(str(params.output_dir / "overview_round.gif"), images, duration=0.15, loop=0)
logger.debug(f"Created overview GIF animation at {params.output_dir / 'overview_round.gif'}")

# Debug: List all files in output directory
logger.debug(f"=== DEBUG: Output directory contents ===")
logger.debug(f"Output directory: {params.output_dir}")
logger.debug(f"Output directory absolute: {params.output_dir.absolute()}")
logger.debug(f"Current working directory: {Path.cwd()}")
logger.debug(f"Files in output directory:")
for file_path in params.output_dir.iterdir():
    logger.debug(f"  {file_path.name}")

# cleanup
os.chdir(params.output_dir)
del renderer
