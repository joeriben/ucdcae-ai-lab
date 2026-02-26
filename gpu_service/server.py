#!/usr/bin/env python3
"""
AI4ArtsEd GPU Service — Shared Diffusers + HeartMuLa Inference

Standalone server that handles all GPU-intensive model inference.
Both dev (17802) and prod (17801) backends call this via HTTP REST.

Port: 17803 (configurable via GPU_SERVICE_PORT env var)
"""

import logging
import sys
import os

# Configure logging before importing app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gpu_service")

# Ensure gpu_service directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waitress import serve
from app import create_app
from config import HOST, PORT, THREADS


def main():
    app = create_app()

    print(f"=== AI4ArtsEd GPU Service ===")
    print(f"Starting on http://{HOST}:{PORT}")
    print(f"Using {THREADS} threads")
    print(f"Endpoints:")
    print(f"  POST /api/diffusers/generate")
    print(f"  POST /api/diffusers/generate/fusion")
    print(f"  POST /api/diffusers/generate/attention")
    print(f"  POST /api/diffusers/generate/probing")
    print(f"  POST /api/diffusers/generate/algebra")
    print(f"  POST /api/diffusers/generate/archaeology")
    print(f"  POST /api/heartmula/generate")
    print(f"  GET  /api/health")
    print(f"Press Ctrl+C to stop")
    print()

    serve(
        app,
        host=HOST,
        port=PORT,
        threads=THREADS,
        url_scheme='http',
        channel_timeout=600,  # 10 min for long generations
    )


if __name__ == "__main__":
    main()
