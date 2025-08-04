import numpy as np
import laspy
import open3d as o3d
from shapely.geometry import Polygon
from pyproj import Transformer


def load_dataset(dataset_path: str):
    las = laspy.read(dataset_path)
    points = np.vstack((las.x, las.y, las.z)).T
    center = points.mean(axis=0)

    # Compute bounding box in original coordinates
    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)

    # Define bounding box polygon
    bbox_polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

    points -= center  # Shift points to be centered at the origin

    # Get CRS from LAS file
    input_crs = None
    if las.header.parse_crs():
        input_crs = las.header.parse_crs()
    if input_crs is None:
        raise ValueError("No spatial reference found in LAS file.")
        
    # Transform to EPSG:3857 (Web Mercator)
    transformer = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
    transformed_coords = [transformer.transform(x, y) for x, y in bbox_polygon.exterior.coords]
    bbox_polygon_4326 = Polygon(transformed_coords)

    return points, bbox_polygon_4326


def render_top_down_views(renderer, angle: float, center: np.ndarray, camera_distance: float):
    direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0.5])
    up = np.array([0, 0, 1])
    eye = center + direction * camera_distance
    renderer.setup_camera(60.0, center, eye, up)
    img = renderer.render_to_image()
    return img


def render_section_views(renderer, points: np.ndarray, axis_index: int, direction: str, front: np.ndarray, mat: o3d.visualization.rendering.MaterialRecord, center: np.ndarray, section_width: float):
    center_coord = center[axis_index]

    # Initial mask using center-based slicing
    mask = (points[:, axis_index] > center_coord - section_width / 2) & \
            (points[:, axis_index] < center_coord + section_width / 2)

    # If mask is empty, fall back to median coordinate
    if not np.any(mask):
        median_coord = np.median(points[:, axis_index])
        mask = (points[:, axis_index] > median_coord - section_width / 2) & \
                (points[:, axis_index] < median_coord + section_width / 2)
        if not np.any(mask):
            print(f"Skipping section view {direction} — no points in center or median slice.")
            return None           
        center_coord = median_coord
        
    section_points = points[mask]
    section_pcd = o3d.geometry.PointCloud()
    section_pcd.points = o3d.utility.Vector3dVector(section_points)
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("section_pcd", section_pcd, mat)
    
    # Set camera for section view
    section_center = section_pcd.get_center()
    eye = section_center + np.array(front) * 10
    up = np.array([0, 0, 1])
    renderer.setup_camera(60.0, section_center, eye, up)
    
    img = renderer.render_to_image()
    return img


