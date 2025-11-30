#!/usr/bin/env python3
"""
RunPod Serverless Handler for Gaussian Splatting training.

This handler processes video input and creates 3D Gaussian Splatting models.
It handles the full pipeline:
1. Download video from URL
2. Extract frames using ffmpeg
3. Run COLMAP for camera poses
4. Train 3DGS model
5. Upload results to S3/MinIO
"""
import os
import json
import subprocess
import uuid
import shutil
from pathlib import Path
from typing import Any, Dict
import logging

import runpod
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Settings from environment ===
OUTPUT_BUCKET_URL = os.getenv("OUTPUT_BUCKET_URL", "")
OUTPUT_BUCKET_KEY = os.getenv("OUTPUT_BUCKET_KEY", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "gsplatt-results")

WORKDIR = Path("/workspace")


def run_cmd(cmd: list, cwd: Path = None) -> subprocess.CompletedProcess:
    """Run shell command with logging and error checking."""
    logger.info(f"RUN: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True
    )
    if result.stdout:
        logger.debug(result.stdout)
    return result


def download_video(url: str, dst: Path) -> None:
    """Download video from URL."""
    logger.info(f"Downloading video from {url}")
    
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        logger.info(f"Downloaded {downloaded} bytes to {dst}")


def upload_results_s3(result_dir: Path, scene_id: str) -> str:
    """
    Upload results to S3/MinIO.
    Returns public URL to the uploaded file.
    """
    try:
        import boto3
        from botocore.config import Config
        
        # Create zip archive
        zip_name = f"{scene_id}.zip"
        zip_path = result_dir.parent / zip_name
        
        logger.info(f"Creating archive: {zip_path}")
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', result_dir)
        
        # Upload to S3
        s3_config = Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        )
        
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT_URL if S3_ENDPOINT_URL else None,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=s3_config
        )
        
        s3_key = f"results/{zip_name}"
        logger.info(f"Uploading to S3: {S3_BUCKET_NAME}/{s3_key}")
        
        s3_client.upload_file(
            str(zip_path),
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ACL': 'public-read'}
        )
        
        # Generate public URL
        if S3_ENDPOINT_URL:
            public_url = f"{S3_ENDPOINT_URL}/{S3_BUCKET_NAME}/{s3_key}"
        else:
            public_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        
        logger.info(f"Uploaded to: {public_url}")
        return public_url
        
    except ImportError:
        logger.warning("boto3 not available, falling back to simple upload")
        return upload_results_simple(result_dir, scene_id)
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        raise


def upload_results_simple(result_dir: Path, scene_id: str) -> str:
    """
    Simple HTTP upload fallback.
    """
    if not OUTPUT_BUCKET_URL:
        raise ValueError("OUTPUT_BUCKET_URL not configured")
    
    zip_name = f"{scene_id}.zip"
    zip_path = result_dir.parent / zip_name
    
    logger.info(f"Creating archive: {zip_path}")
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', result_dir)
    
    logger.info(f"Uploading to {OUTPUT_BUCKET_URL}/{zip_name}")
    
    with open(zip_path, "rb") as f:
        headers = {}
        if OUTPUT_BUCKET_KEY:
            headers["Authorization"] = f"Bearer {OUTPUT_BUCKET_KEY}"
        
        resp = requests.put(
            f"{OUTPUT_BUCKET_URL}/{zip_name}",
            data=f,
            headers=headers,
            timeout=600
        )
        resp.raise_for_status()
    
    return f"{OUTPUT_BUCKET_URL}/{zip_name}"


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main job handler.
    
    Expected input format:
    {
        "video_url": "https://.../video.mp4",
        "scene_id": "optional-id",
        "params": {
            "iterations": 30000,
            "fps": 2
        }
    }
    """
    job_id = job.get("id", "unknown")
    
    try:
        job_input = job["input"]
        video_url: str = job_input["video_url"]
        params: Dict[str, Any] = job_input.get("params", {})
        iterations = int(params.get("iterations", 30000))
        fps = int(params.get("fps", 2))
        
        scene_id = job_input.get("scene_id") or str(uuid.uuid4())
        scene_root = WORKDIR / "scenes" / scene_id
        scene_root.mkdir(parents=True, exist_ok=True)
        
        # === Stage 0: Download video (0-10%) ===
        runpod.serverless.progress_update(job, {"progress": 0, "stage": "downloading_video"})
        
        video_path = scene_root / "input.mp4"
        download_video(video_url, video_path)
        
        runpod.serverless.progress_update(job, {"progress": 10, "stage": "video_downloaded"})
        
        # === Stage 1: Prepare dataset (10-30%) ===
        runpod.serverless.progress_update(job, {"progress": 10, "stage": "preparing_dataset"})
        
        run_cmd([
            "python3", str(WORKDIR / "prepare_from_video.py"),
            "--video", str(video_path),
            "--out", str(scene_root),
            "--fps", str(fps)
        ], cwd=WORKDIR)
        
        runpod.serverless.progress_update(job, {"progress": 30, "stage": "dataset_ready"})
        
        # === Stage 2: Train 3DGS (30-90%) ===
        runpod.serverless.progress_update(job, {"progress": 30, "stage": "training"})
        
        output_dir = scene_root / "output"
        output_dir.mkdir(exist_ok=True)
        
        run_cmd([
            "python3", "train.py",
            "-s", str(scene_root),
            "-m", str(output_dir),
            f"--iterations={iterations}"
        ], cwd=WORKDIR)
        
        runpod.serverless.progress_update(job, {"progress": 90, "stage": "training_complete"})
        
        # === Stage 3: Upload results (90-100%) ===
        runpod.serverless.progress_update(job, {"progress": 90, "stage": "uploading"})
        
        plt_url = upload_results_s3(output_dir, scene_id)
        
        runpod.serverless.progress_update(job, {"progress": 100, "stage": "done"})
        
        # Cleanup
        try:
            shutil.rmtree(scene_root)
        except Exception as e:
            logger.warning(f"Failed to cleanup: {e}")
        
        return {
            "status": "success",
            "scene_id": scene_id,
            "progress": 100,
            "plt_url": plt_url
        }
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        return {
            "status": "fail",
            "error": str(e),
            "progress": 0
        }


if __name__ == "__main__":
    logger.info("Starting RunPod Gaussian Splatting worker...")
    runpod.serverless.start({"handler": handler})

