#!/bin/bash
# Start ComfyUI directly (no SwarmUI middleware)
#
# This replaces 2_start_swarmui.sh for COMFYUI_DIRECT mode.
# ComfyUI runs on port 7821 and is accessed directly by the DevServer
# via WebSocket (progress tracking) and HTTP (media download).

# Keep window open on error
trap 'echo ""; echo "❌ Script failed! Press any key to close..."; read -n 1 -s -r' ERR

COMFYUI_PORT=7821

# Get directory where this script lives (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ComfyUI location (inside SwarmUI's dlbackend)
COMFYUI_DIR="$SCRIPT_DIR/../SwarmUI/dlbackend/ComfyUI"

# Check if port is in use and terminate any process using it
echo "Checking port ${COMFYUI_PORT}..."
if lsof -ti:${COMFYUI_PORT} > /dev/null 2>&1; then
    echo "Port ${COMFYUI_PORT} is in use. Terminating existing process..."
    lsof -ti:${COMFYUI_PORT} | xargs -r kill -9
    sleep 2
    echo "✅ Process terminated."
fi

echo "=== Starting ComfyUI (Direct Mode) ==="
echo ""

# Check if ComfyUI directory exists
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "❌ ERROR: ComfyUI not found at: $COMFYUI_DIR"
    echo ""
    echo "Expected location: SwarmUI/dlbackend/ComfyUI/"
    exit 1
fi

cd "$COMFYUI_DIR"
echo "Working directory: $COMFYUI_DIR"

# Activate venv (SwarmUI's venv has all ComfyUI dependencies)
VENV_DIR="$SCRIPT_DIR/../SwarmUI/venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
else
    echo "⚠️ No venv found at $VENV_DIR, using system Python"
fi

echo "Starting ComfyUI on port $COMFYUI_PORT..."
echo "Press Ctrl+C to stop"
echo ""

# Remove error trap - allow normal server exit without "Script failed" message
trap - ERR

# Start ComfyUI directly
# --listen 127.0.0.1: only local access (DevServer connects locally)
# --port 7821: same port as SwarmUI's integrated ComfyUI
# --extra-model-paths-config: use SwarmUI's model directories
python main.py \
    --listen 127.0.0.1 \
    --port "$COMFYUI_PORT" \
    --extra-model-paths-config extra_model_paths.yaml \
    --preview-method auto
