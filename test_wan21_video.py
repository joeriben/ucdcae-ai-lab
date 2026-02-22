#!/usr/bin/env python3
"""PoC: Wan 2.1 video generation via GPU Service (Port 17803).

Usage:
    venv/bin/python test_wan21_video.py            # 1.3B (480p, ~8GB VRAM)
    venv/bin/python test_wan21_video.py --14b       # 14B (720p, ~40GB VRAM)
    venv/bin/python test_wan21_video.py --prompt "A bird flying over the ocean"
"""

import argparse
import base64
import json
import sys
import time

import requests

GPU_SERVICE_URL = "http://localhost:17803"

MODELS = {
    "1.3b": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "width": 848,
        "height": 480,
    },
    "14b": {
        "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "width": 1280,
        "height": 720,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Wan 2.1 Video PoC")
    parser.add_argument("--14b", dest="use_14b", action="store_true",
                        help="Use 14B model (720p, ~40GB VRAM)")
    parser.add_argument("--prompt", default="A cat walking slowly through a sunlit garden",
                        help="Text prompt for video generation")
    parser.add_argument("--frames", type=int, default=33,
                        help="Number of frames (default: 33 for quick test)")
    parser.add_argument("--steps", type=int, default=15,
                        help="Inference steps (default: 15 for quick test)")
    parser.add_argument("--output", default="test_output_video.mp4",
                        help="Output file path")
    args = parser.parse_args()

    model_key = "14b" if args.use_14b else "1.3b"
    model = MODELS[model_key]

    # Check GPU service is reachable
    try:
        r = requests.get(f"{GPU_SERVICE_URL}/api/diffusers/status", timeout=5)
        print(f"GPU Service status: {r.status_code}")
    except requests.ConnectionError:
        print(f"ERROR: GPU Service not reachable at {GPU_SERVICE_URL}")
        sys.exit(1)

    payload = {
        "model_id": model["model_id"],
        "prompt": args.prompt,
        "width": model["width"],
        "height": model["height"],
        "num_frames": args.frames,
        "steps": args.steps,
        "cfg_scale": 5.0,
        "fps": 16,
        "seed": 42,
    }

    print(f"\nModel: {model_key} ({model['model_id']})")
    print(f"Resolution: {model['width']}x{model['height']}")
    print(f"Frames: {args.frames}, Steps: {args.steps}")
    print(f"Prompt: {args.prompt}")
    print(f"\nSending request...")

    start = time.time()
    try:
        r = requests.post(
            f"{GPU_SERVICE_URL}/api/diffusers/generate/video",
            json=payload,
            timeout=600,  # 10 min timeout for video generation
        )
    except requests.Timeout:
        print("ERROR: Request timed out (10 min)")
        sys.exit(1)

    elapsed = time.time() - start

    if r.status_code != 200:
        print(f"ERROR: HTTP {r.status_code}")
        print(r.text[:500])
        sys.exit(1)

    data = r.json()
    if not data.get("success"):
        print(f"ERROR: {data.get('error', 'unknown')}")
        sys.exit(1)

    video_bytes = base64.b64decode(data["video_base64"])
    with open(args.output, "wb") as f:
        f.write(video_bytes)

    size_mb = len(video_bytes) / (1024 * 1024)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
