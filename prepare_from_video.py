#!/usr/bin/env python3
"""
Video preparation script for Gaussian Splatting.

This script:
1. Extracts frames from video using ffmpeg
2. Runs COLMAP for camera pose estimation
3. Prepares directory structure for gaussian-splatting training

Output structure (compatible with gaussian-splatting):
    scene_dir/
    ├── input/            # Source images
    ├── images/           # Processed images (may be same as input)
    ├── sparse/0/         # COLMAP sparse reconstruction
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    └── database.db       # COLMAP database
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cmd(cmd: list, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command with logging."""
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        capture_output=True,
        text=True
    )
    if result.stdout:
        logger.debug(result.stdout)
    if result.stderr:
        logger.debug(result.stderr)
    return result


def extract_frames(video_path: Path, output_dir: Path, fps: int = 2) -> int:
    """
    Extract frames from video using ffmpeg.
    
    Args:
        video_path: Path to input video
        output_dir: Directory for extracted frames
        fps: Frames per second to extract
        
    Returns:
        Number of extracted frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames with ffmpeg
    output_pattern = str(output_dir / "frame_%05d.jpg")
    
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality JPEG
        output_pattern
    ])
    
    # Count extracted frames
    frames = list(output_dir.glob("*.jpg"))
    logger.info(f"Extracted {len(frames)} frames")
    
    return len(frames)


def run_colmap_feature_extraction(database_path: Path, image_dir: Path):
    """Run COLMAP feature extraction."""
    logger.info("Running COLMAP feature extraction...")
    
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1"
    ])


def run_colmap_matcher(database_path: Path):
    """Run COLMAP feature matching."""
    logger.info("Running COLMAP exhaustive matching...")
    
    run_cmd([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "1"
    ])


def run_colmap_mapper(database_path: Path, image_dir: Path, sparse_dir: Path):
    """Run COLMAP sparse reconstruction."""
    logger.info("Running COLMAP mapper...")
    
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    run_cmd([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_extra_params", "0"
    ])


def run_colmap_image_undistorter(image_dir: Path, sparse_dir: Path, output_dir: Path):
    """Run COLMAP image undistortion (optional, for clean images)."""
    logger.info("Running COLMAP image undistorter...")
    
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(output_dir),
        "--output_type", "COLMAP"
    ])


def run_colmap_pipeline(scene_dir: Path, image_dir: Path) -> Path:
    """
    Run complete COLMAP pipeline.
    
    Args:
        scene_dir: Root scene directory
        image_dir: Directory with input images
        
    Returns:
        Path to sparse reconstruction directory
    """
    database_path = scene_dir / "database.db"
    sparse_dir = scene_dir / "sparse"
    
    # Remove old database if exists
    if database_path.exists():
        database_path.unlink()
    
    # Run COLMAP pipeline
    run_colmap_feature_extraction(database_path, image_dir)
    run_colmap_matcher(database_path)
    run_colmap_mapper(database_path, image_dir, sparse_dir)
    
    # Verify reconstruction exists
    reconstruction_dir = sparse_dir / "0"
    if not reconstruction_dir.exists():
        # Try to find any reconstruction
        reconstructions = list(sparse_dir.glob("*"))
        if reconstructions:
            reconstruction_dir = reconstructions[0]
        else:
            raise RuntimeError("COLMAP failed to produce sparse reconstruction")
    
    logger.info(f"Sparse reconstruction created at: {reconstruction_dir}")
    return sparse_dir


def prepare_scene(video_path: Path, output_dir: Path, fps: int = 2):
    """
    Prepare complete scene for gaussian-splatting training.
    
    Args:
        video_path: Path to input video
        output_dir: Output scene directory
        fps: Frames per second to extract
    """
    logger.info(f"Preparing scene from {video_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "input"
    images_dir = output_dir / "images"
    
    # Step 1: Extract frames
    logger.info("Step 1: Extracting frames from video...")
    num_frames = extract_frames(video_path, input_dir, fps)
    
    if num_frames < 3:
        raise RuntimeError(f"Not enough frames extracted ({num_frames}). Need at least 3.")
    
    # Step 2: Copy to images directory (gaussian-splatting expects this)
    logger.info("Step 2: Preparing images directory...")
    if images_dir.exists():
        shutil.rmtree(images_dir)
    shutil.copytree(input_dir, images_dir)
    
    # Step 3: Run COLMAP
    logger.info("Step 3: Running COLMAP pipeline...")
    sparse_dir = run_colmap_pipeline(output_dir, images_dir)
    
    # Verify final structure
    required_files = [
        images_dir,
        sparse_dir / "0" / "cameras.bin",
        sparse_dir / "0" / "images.bin",
        sparse_dir / "0" / "points3D.bin"
    ]
    
    for f in required_files:
        if not f.exists():
            logger.warning(f"Missing expected file/dir: {f}")
    
    logger.info("Scene preparation complete!")
    logger.info(f"Ready for training with: python train.py -s {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare video for Gaussian Splatting training"
    )
    parser.add_argument(
        "--video", "-v",
        type=Path,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        required=True,
        help="Output scene directory"
    )
    parser.add_argument(
        "--fps", "-f",
        type=int,
        default=2,
        help="Frames per second to extract (default: 2)"
    )
    
    args = parser.parse_args()
    
    if not args.video.exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    try:
        prepare_scene(args.video, args.out, args.fps)
    except Exception as e:
        logger.error(f"Failed to prepare scene: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

