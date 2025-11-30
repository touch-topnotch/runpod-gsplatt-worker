#!/usr/bin/env python3
"""
RunPod Serverless Handler for Gaussian Splatting training.

This handler processes video input and creates 3D Gaussian Splatting models.
It handles the full pipeline:
1. Download video from URL
2. Extract frames using ffmpeg
3. Run COLMAP for camera poses
4. Train 3DGS model
5. Upload results to master server
"""
import os
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
# URL вашего master-server для загрузки результатов
MASTER_SERVER_URL = os.getenv("MASTER_SERVER_URL", "")
# API ключ для авторизации загрузки (опционально)
UPLOAD_API_KEY = os.getenv("UPLOAD_API_KEY", "")

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


def upload_results(result_dir: Path, scene_id: str) -> str:
    """
    Upload results to master server.
    Returns public URL to the uploaded file.
    """
    if not MASTER_SERVER_URL:
        raise ValueError("MASTER_SERVER_URL not configured")
    
    # Create zip archive
    zip_name = f"{scene_id}.zip"
    zip_path = result_dir.parent / zip_name
    
    logger.info(f"Creating archive: {zip_path}")
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', result_dir)
    
    # Upload to master server
    upload_url = f"{MASTER_SERVER_URL.rstrip('/')}/api/upload/gsplatt"
    logger.info(f"Uploading to {upload_url}")
    
    headers = {}
    if UPLOAD_API_KEY:
        headers["Authorization"] = f"Bearer {UPLOAD_API_KEY}"
    
    with open(zip_path, "rb") as f:
        files = {"file": (zip_name, f, "application/zip")}
        data = {"scene_id": scene_id}
        
        resp = requests.post(
            upload_url,
            files=files,
            data=data,
            headers=headers,
            timeout=600
        )
        resp.raise_for_status()
    
    # Parse response to get public URL
    try:
        result = resp.json()
        public_url = result.get("url") or result.get("download_url")
        if public_url:
            logger.info(f"Uploaded successfully: {public_url}")
            return public_url
    except Exception:
        pass
    
    # Fallback: construct URL
    public_url = f"{MASTER_SERVER_URL.rstrip('/')}/files/gsplatt/{zip_name}"
    logger.info(f"Uploaded to: {public_url}")
    return public_url


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
        
        plt_url = upload_results(output_dir, scene_id)
        
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
