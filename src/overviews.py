import numpy as np
import laspy
from typing import List
import logging

logger = logging.getLogger(__name__)


def load_single_dataset(dataset_path: str) -> np.ndarray:
    """Load a single point cloud and return centered points."""
    las = laspy.read(dataset_path)
    points = np.vstack((las.x, las.y, las.z)).T
    points -= points.mean(axis=0)  # Center at origin
    return points


def load_and_aggregate_datasets(paths: List[str], max_total_points: int) -> np.ndarray:
    """
    Load multiple point clouds and aggregate with proportional sampling.
    
    Memory efficient: samples from each file before merging.
    Proportional: larger files contribute more points.
    
    Args:
        paths: List of paths to LAZ/LAS files
        max_total_points: Maximum total points in the aggregated result
        
    Returns:
        Centered numpy array of shape (N, 3) containing merged points
    """
    # 1. First pass: count points per file
    file_infos = []
    total_points = 0
    for path in paths:
        las = laspy.read(path)
        count = len(las.points)
        file_infos.append({'path': path, 'count': count})
        total_points += count
        logger.debug(f"File {path}: {count:,} points")
    
    logger.info(f"Total points across {len(paths)} files: {total_points:,}")
    
    # 2. Calculate sample ratio (proportional sampling)
    sample_ratio = min(1.0, max_total_points / total_points)
    if sample_ratio < 1.0:
        logger.info(f"Sampling ratio: {sample_ratio:.4f} (target: {max_total_points:,} points)")
    
    # 3. Load and sample each file (memory efficient)
    all_points = []
    for info in file_infos:
        las = laspy.read(info['path'])
        points = np.vstack((las.x, las.y, las.z)).T
        
        # Sample this file's points immediately
        n_sample = int(info['count'] * sample_ratio)
        if n_sample < len(points):
            indices = np.random.choice(len(points), n_sample, replace=False)
            points = points[indices]
            logger.debug(f"Sampled {n_sample:,} points from {info['path']}")
        
        all_points.append(points)
    
    # 4. Merge sampled points and center
    merged = np.vstack(all_points)
    merged -= merged.mean(axis=0)
    
    logger.info(f"Merged result: {len(merged):,} points")
    
    return merged
