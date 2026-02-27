# AI4ArtsEd DevServer - Installation Guide

Complete step-by-step guide for installing AI4ArtsEd DevServer on a production Linux server.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [System Dependencies](#system-dependencies)
3. [Ollama Installation](#ollama-installation)
4. [SwarmUI Installation](#swarmui-installation)
5. [ComfyUI Custom Nodes](#comfyui-custom-nodes)
6. [AI Model Downloads](#ai-model-downloads)
7. [Application Setup](#application-setup)
8. [Configuration](#configuration)
9. [Service Setup (Optional)](#service-setup-optional)
10. [Starting Services](#starting-services)
11. [Verification](#verification)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **OS:** Ubuntu 22.04+, Fedora 38+, or Arch Linux
- **Disk Space:** 350GB free (400GB recommended)
- **RAM:** 16GB minimum (32GB recommended)
- **GPU:** NVIDIA with 16GB+ VRAM (RTX 4090/5090)
- **CUDA:** Version 12.0+ drivers installed
- **Network:** Stable internet connection for downloads (~100GB total)

### Verify System Compatibility

```bash
# Check disk space
df -h /
# Should show at least 350GB free

# Check RAM
free -h
# Should show at least 16GB total

# Check GPU
nvidia-smi
# Should show your NVIDIA GPU with VRAM info

# Check CUDA version
nvidia-smi | grep "CUDA Version"
# Should show CUDA 12.0 or higher

# Check available ports
sudo lsof -i :7801,7821,11434,17801
# Should show no output (ports available)
```

### Expected Output Examples

**GPU Check:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.154.05   Driver Version: 535.154.05   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 5090     Off  | 00000000:01:00.0 Off |                  Off |
| 30%   45C    P0    65W / 450W |      0MiB / 24564MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**If any checks fail, stop here and resolve issues before continuing.**

---

## System Dependencies

Install required system packages for your Linux distribution.

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install all dependencies
sudo apt install -y \
  git curl wget \
  python3.13 python3-pip python3-venv \
  build-essential \
  libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev libgirepository1.0-dev \
  nodejs npm

# Verify installations
python3 --version  # Should show Python 3.13.x
node --version     # Should show v20.x or v22.x
npm --version      # Should show 9.x or 10.x
```

### Fedora

```bash
# Install all dependencies
sudo dnf install -y \
  git curl wget \
  python3.13 python3-pip \
  gcc make \
  cairo-devel pango-devel libjpeg-devel giflib-devel gobject-introspection-devel \
  nodejs npm

# Verify installations
python3 --version
node --version
npm --version
```

### Arch Linux

```bash
# Install all dependencies
sudo pacman -S --needed \
  git curl wget \
  python python-pip \
  base-devel \
  cairo pango libjpeg-turbo giflib gobject-introspection \
  nodejs npm

# Verify installations
python --version
node --version
npm --version
```

### NVIDIA Drivers

**If `nvidia-smi` didn't work in Prerequisites:**

NVIDIA driver installation is system-specific. Please follow NVIDIA's official guide:
- Ubuntu: https://ubuntu.com/server/docs/nvidia-drivers-installation
- Fedora: https://rpmfusion.org/Howto/NVIDIA
- Arch: https://wiki.archlinux.org/title/NVIDIA

**After installation, verify:**
```bash
nvidia-smi
# Should display GPU information
```

---

## Ollama Installation

Ollama provides local LLM inference for translation, safety checks, and image analysis.

### Install Ollama

```bash
# Download and run official installer
curl -fsSL https://ollama.com/install.sh | sh

# Enable and start service
sudo systemctl enable ollama
sudo systemctl start ollama

# Verify service is running
systemctl status ollama
# Should show "active (running)"
```

### Download Required Models

```bash
# Download gpt-OSS:20b (~21GB) - Safety checks, translation
echo "Downloading gpt-OSS:20b (~21GB, takes 10-15 minutes)..."
ollama pull gpt-OSS:20b

# Download llama3.2-vision (~8GB) - Image analysis
echo "Downloading llama3.2-vision (~8GB, takes 5-10 minutes)..."
ollama pull llama3.2-vision:latest

# Verify models installed
ollama list
```

**Expected Output:**
```
NAME                       ID              SIZE      MODIFIED
gpt-OSS:20b               abc123def456    21 GB     2 minutes ago
llama3.2-vision:latest    def789ghi012    8.0 GB    1 minute ago
```

**Total Download:** ~29GB
**Time:** 15-25 minutes on 100Mbps connection

---

## SwarmUI Installation

SwarmUI provides the image and video generation backend, with integrated ComfyUI.

### Create Installation Directory

```bash
# Choose your installation path (adjust as needed)
cd /opt
sudo mkdir -p ai4artsed
sudo chown $USER:$USER ai4artsed
cd ai4artsed
```

### Clone and Install SwarmUI

```bash
# Clone SwarmUI repository
git clone https://github.com/mcmonkeyprojects/SwarmUI.git SwarmUI
cd SwarmUI

# Run SwarmUI's automated installer
# This will:
# - Create Python venv
# - Install SwarmUI dependencies
# - Clone ComfyUI to dlbackend/ComfyUI/
# - Set up initial configuration
./install-linux.sh
```

**Installer will download ~2GB of dependencies and take 5-10 minutes.**

### Verify Installation

```bash
# Check directory structure
ls -la dlbackend/ComfyUI/
# Should show ComfyUI directory with models/, custom_nodes/, etc.

ls -la venv/
# Should show Python virtual environment

# Test start SwarmUI (will initialize, then exit with Ctrl+C)
./launch-linux.sh
# Wait for "Server started" message, then press Ctrl+C
```

---

## ComfyUI Custom Nodes

AI4ArtsEd requires three custom ComfyUI nodes for pedagogical workflows.

**Recommended:** Use the automated helper script below.

### Automated Installation (Recommended)

```bash
cd /opt/ai4artsed/ai4artsed_webserver
./install_comfyui_nodes.sh
```

**Or with custom SwarmUI path:**

```bash
SWARMUI_PATH=/custom/path/SwarmUI /opt/ai4artsed/ai4artsed_webserver/install_comfyui_nodes.sh
```

This script automatically:
- Detects SwarmUI path
- Activates virtual environment
- Clones/updates all 3 nodes
- Installs dependencies

**Full documentation:** [COMFYUI_NODES_HELPER.md](COMFYUI_NODES_HELPER.md)

### Manual Installation (If Needed)

For more control, or if the helper script doesn't work:

```bash
cd /opt/ai4artsed/SwarmUI/dlbackend/ComfyUI/custom_nodes

# Activate SwarmUI's venv for pip installs
source ../../venv/bin/activate
```

**1. ComfyUI-LTXVideo (Video Generation)**

```bash
git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git
cd ComfyUI-LTXVideo
pip install -r requirements.txt
cd ..
```

**2. ai4artsed_comfyui (Custom Pedagogical Nodes)**

```bash
git clone https://github.com/joeriben/ai4artsed_comfyui_nodes.git ai4artsed_comfyui
cd ai4artsed_comfyui
pip install -r requirements.txt
cd ..
```

**3. comfyui-sound-lab (Audio Generation)**

```bash
git clone https://github.com/MixLabPro/comfyui-sound-lab.git
cd comfyui-sound-lab
pip install -r requirements.txt || echo "⚠ Some dependencies failed (optional)"
cd ..
```

**Verify and finish:**

```bash
# List installed custom nodes
ls -la
# Should show: ComfyUI-LTXVideo, ai4artsed_comfyui, comfyui-sound-lab

# Deactivate venv
deactivate
```

---

## AI Model Downloads

Download ~48GB of AI models for image and video generation.

**Total Size:** ~48GB
**Time:** 30-60 minutes on 100Mbps connection

### 1. Stable Diffusion 3.5 Large + Encoders (~22GB)

```bash
# Create directory for main model
mkdir -p /opt/ai4artsed/SwarmUI/Models/Stable-Diffusion/OfficialStableDiffusion
cd /opt/ai4artsed/SwarmUI/Models/Stable-Diffusion/OfficialStableDiffusion

# Download SD3.5 Large (16GB)
echo "Downloading sd3.5_large.safetensors (16GB, ~15-30 min)..."
wget https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors

# Download CLIP encoders
cd /opt/ai4artsed/SwarmUI/dlbackend/ComfyUI/models/clip

echo "Downloading clip_g.safetensors (1.3GB)..."
wget https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/clip_g.safetensors

echo "Downloading t5xxl_enconly.safetensors (4.6GB)..."
wget https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/t5xxl_enconly.safetensors
```

### 2. LTX-Video Model (~15GB)

```bash
cd /opt/ai4artsed/SwarmUI/dlbackend/ComfyUI/models/checkpoints

echo "Downloading ltxv-13b-0.9.7-distilled-fp8.safetensors (15GB, ~15-30 min)..."
wget https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-distilled-fp8.safetensors
```

### 3. T5 Encoder for LTX-Video (~11GB)

```bash
cd /opt/ai4artsed/SwarmUI/dlbackend/ComfyUI/models/clip

echo "Downloading t5xxl_fp16.safetensors (11GB, ~10-20 min)..."
wget https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/resolve/main/t5xxl_fp16.safetensors
```

### Faster Downloads (Optional)

If you have `aria2c` installed, use it for faster multi-connection downloads:

```bash
# Install aria2c (if not already installed)
sudo apt install aria2  # Ubuntu/Debian
# OR
sudo dnf install aria2  # Fedora
# OR
sudo pacman -S aria2    # Arch

# Download with 16 parallel connections
aria2c -x 16 -s 16 [URL]
```

### Create Required Symlink

```bash
# Create symlink for SwarmUI model access
cd /opt/ai4artsed/SwarmUI/dlbackend/ComfyUI/models/checkpoints
ln -s ../../../Models/Stable-Diffusion/OfficialStableDiffusion OfficialStableDiffusion
```

### Verify Model Installation

```bash
# Check all models exist
ls -lh /opt/ai4artsed/SwarmUI/Models/Stable-Diffusion/OfficialStableDiffusion/sd3.5_large.safetensors
# Should show ~16GB file

ls -lh /opt/ai4artsed/SwarmUI/dlbackend/ComfyUI/models/clip/
# Should show clip_g.safetensors (1.3GB), t5xxl_enconly.safetensors (4.6GB), t5xxl_fp16.safetensors (11GB)

ls -lh /opt/ai4artsed/SwarmUI/dlbackend/ComfyUI/models/checkpoints/
# Should show ltxv-13b-0.9.7-distilled-fp8.safetensors (15GB) and OfficialStableDiffusion symlink
```

**For checksums and detailed model information, see:** [MODELS_REQUIRED.md](MODELS_REQUIRED.md)

---

## Application Setup

Install the AI4ArtsEd DevServer application.

### Clone Repository

```bash
cd /opt/ai4artsed
git clone https://github.com/joerissenbenjamin/ai4artsed_webserver.git
cd ai4artsed_webserver
```

### Python Backend Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

**Expected packages installed:**
- Flask==3.0.0, Flask-CORS==4.0.0 (Web framework)
- waitress==2.1.2 (Production WSGI server)
- requests==2.31.0, aiohttp==3.13.2 (HTTP clients)
- python-dotenv==1.0.0 (Environment variables)
- weasyprint==60.2, python-docx==1.1.0, lxml==5.1.0 (Export functionality)

### Verify Python Installation

```bash
# Verify packages installed
pip list | grep Flask
pip list | grep waitress
```

### HeartMuLa Music Generation (Optional)

HeartMuLa is a direct Diffusers backend for music generation. Skip this section if you don't need music generation or will use ComfyUI-based ACE-Step instead.

**Prerequisites:**
- heartlib repository cloned to `~/ai/heartlib`
- Model checkpoint in `~/ai/heartlib/ckpt/`

**Installation:**

```bash
# Ensure venv is still active
# Install heartlib in editable mode WITHOUT its dependencies
# (dependencies are already in requirements.txt to avoid version conflicts)
pip install --no-deps -e ~/ai/heartlib

# Verify installation
pip list | grep heartlib
# Should show: heartlib 0.x.x /home/USER/ai/heartlib
```

**Required Post-Install Fix — Codebook Index Clamping:**

HeartMuLa's audio vocabulary (8197 tokens) exceeds HeartCodec's codebook size (8192). Without clamping, certain generated tokens cause CUDA crashes. Apply this fix in heartlib:

File: `~/ai/heartlib/src/heartlib/heartcodec/models/flow_matching.py`, in `inference_codes()`, before the `get_output_from_indices` call (~line 75):

```python
# Add this line:
codes_bestrq_emb = torch.clamp(codes_bestrq_emb, 0, self.vq_embed.codebook_size - 1)
```

**PyTorch Version Sensitivity:**

heartlib's codec is sensitive to PyTorch nightly versions, especially on newer GPUs (Blackwell/sm_120). If you encounter CUDA errors during codec detokenization, ensure all machines use the **exact same PyTorch nightly build**. Even a one-day difference between nightlies can introduce regressions.

**Configuration:**

Edit `devserver/config/backends.yaml` to enable/disable:
```yaml
heartmula:
  enabled: true
  config:
    device: cuda
    version: "3B"
    lazy_load: true
```

**Note:** If heartlib is not installed, the system automatically falls back to ComfyUI for music generation.

```bash
# Deactivate venv for now
deactivate
```

### Frontend Build

```bash
cd /opt/ai4artsed/ai4artsed_webserver/public/ai4artsed-frontend

# Install Node.js dependencies
npm install

# Build production frontend
npm run build
```

**Build will take 2-5 minutes.**

### Verify Frontend Build

```bash
# Check dist directory exists
ls -la dist/
# Should show index.html, assets/, logos/, etc.

ls dist/index.html
# Should exist
```

### Create Required Directories

```bash
cd /opt/ai4artsed/ai4artsed_webserver

# Create exports directories
mkdir -p exports/{json,html,pdf,docx,xml,pipeline_runs}

# Verify
ls -la exports/
```

---

## Configuration

Configure AI4ArtsEd for your system.

### 1. Edit Main Configuration

```bash
cd /opt/ai4artsed/ai4artsed_webserver/devserver
nano config.py  # Or use your preferred editor (vim, vi, etc.)
```

**Key settings to update:**

```python
# Lines 298-299: Update paths to match your installation
SWARMUI_BASE_PATH = os.environ.get("SWARMUI_PATH", "/opt/ai4artsed/SwarmUI")
COMFYUI_BASE_PATH = os.environ.get("COMFYUI_PATH", "/opt/ai4artsed/SwarmUI/dlbackend/ComfyUI")

# Line 67: Production port
PORT = 17801  # Keep as 17801 for production

# Line 66: Listen on all interfaces
HOST = "0.0.0.0"  # Keep as is

# Lines 35, 61: UI mode and safety level
UI_MODE = "youth"  # Options: "kids", "youth", "expert"
DEFAULT_SAFETY_LEVEL = "youth"  # Options: "kids", "youth", "adult", "off"

# Lines 87-102: Model configuration (already configured, but can be adjusted)
# These control which LLMs are used for each pipeline stage
```

**For detailed configuration options, see:** [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)

### 2. Create API Keys File

```bash
cd /opt/ai4artsed/ai4artsed_webserver/devserver
nano api_keys.json
```

**Content:**
```json
{
  "openrouter": "sk-or-v1-YOUR_OPENROUTER_KEY_HERE",
  "openai": "sk-proj-YOUR_OPENAI_KEY_HERE",
  "openai_org_id": "org-YOUR_OPENAI_ORG_HERE"
}
```

**API Key Information:**
- **OpenRouter** (Required): Get key from https://openrouter.ai/keys
  - Used for cloud LLM access (Claude, Gemini, etc.)
- **OpenAI** (Optional): Get key from https://platform.openai.com/api-keys
  - Only needed if using GPT-Image-1 or DALL-E output configs

**Save and exit** (Ctrl+X, Y, Enter in nano)

### 3. Set File Permissions

```bash
# Secure API keys file
chmod 600 /opt/ai4artsed/ai4artsed_webserver/devserver/api_keys.json
```

---

## Service Setup (Optional)

Set up systemd services for production deployment with auto-start.

**Skip this section if you want to run services manually.**

### Create System User

```bash
# Create dedicated user for running services
sudo useradd --system --no-create-home --shell /bin/false ai4artsed

# Grant ownership of installation
sudo chown -R ai4artsed:ai4artsed /opt/ai4artsed
```

### Create Systemd Service Files

**1. SwarmUI Service:**

```bash
sudo nano /etc/systemd/system/ai4artsed-swarmui.service
```

**Content:**
```ini
[Unit]
Description=AI4ArtsEd SwarmUI (Image/Video Generation Backend)
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=ai4artsed
WorkingDirectory=/opt/ai4artsed/SwarmUI
Environment="PATH=/opt/ai4artsed/SwarmUI/venv/bin:/usr/bin:/bin"
ExecStart=/opt/ai4artsed/SwarmUI/launch-linux.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**2. Backend Service:**

```bash
sudo nano /etc/systemd/system/ai4artsed-backend.service
```

**Content:**
```ini
[Unit]
Description=AI4ArtsEd Backend (Flask Application)
After=network.target ai4artsed-swarmui.service
Requires=ai4artsed-swarmui.service

[Service]
Type=simple
User=ai4artsed
WorkingDirectory=/opt/ai4artsed/ai4artsed_webserver/devserver
Environment="PATH=/opt/ai4artsed/ai4artsed_webserver/venv/bin:/usr/bin:/bin"
Environment="SWARMUI_PATH=/opt/ai4artsed/SwarmUI"
Environment="COMFYUI_PATH=/opt/ai4artsed/SwarmUI/dlbackend/ComfyUI"
ExecStart=/opt/ai4artsed/ai4artsed_webserver/venv/bin/python3 server.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Enable Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services (auto-start on boot)
sudo systemctl enable ai4artsed-swarmui
sudo systemctl enable ai4artsed-backend

# Verify services are enabled
systemctl list-unit-files | grep ai4artsed
```

### Ollama Self-Healing (Recommended)

The safety system depends on Ollama for LLM verification. If Ollama becomes unresponsive during a workshop, the system can automatically restart it — but only if passwordless sudo is configured:

```bash
# Run the setup script (one-time, as root)
sudo ./0_setup_ollama_watchdog.sh
```

This creates `/etc/sudoers.d/ai4artsed-ollama` granting the service user passwordless access to `systemctl restart/start/stop ollama` only. No other sudo rights are granted.

**What it does:** When the safety circuit breaker detects 3 consecutive Ollama failures, it automatically restarts Ollama and waits for recovery (max 30s). If the restart succeeds, the user never sees an error.

**Without this setup:** The system falls back to a human-readable error message asking the admin to restart Ollama manually.

---

## Starting Services

### Option A: With Systemd Services (Recommended)

```bash
# Start services in order
sudo systemctl start ai4artsed-swarmui
sudo systemctl start ai4artsed-backend

# Check status
sudo systemctl status ai4artsed-swarmui
sudo systemctl status ai4artsed-backend

# View logs
sudo journalctl -u ai4artsed-swarmui -f  # Press Ctrl+C to exit
sudo journalctl -u ai4artsed-backend -f
```

### Option B: Manual Startup

**Terminal 1 - Start SwarmUI:**
```bash
cd /opt/ai4artsed/SwarmUI
./launch-linux.sh
```

**Terminal 2 - Start Backend (wait for SwarmUI to be ready):**
```bash
cd /opt/ai4artsed/ai4artsed_webserver/devserver
source ../venv/bin/activate
python3 server.py
```

**Keep both terminals open while services run.**

---

## Verification

### Check All Services Running

```bash
# Check Ollama
systemctl status ollama | grep Active
# Should show "active (running)"

# Check SwarmUI
curl -s http://localhost:7801/API/GetNewSession > /dev/null && echo "✓ SwarmUI ready" || echo "✗ SwarmUI not responding"

# Check Backend
curl -s http://localhost:17801/ > /dev/null && echo "✓ Backend ready" || echo "✗ Backend not responding"
```

### Access Web Interface

**Local Access:**
```
http://localhost:17801
```

**Network Access:**
```
http://YOUR_SERVER_IP:17801
```

**Expected:** AI4ArtsEd web interface loads with language selection (Deutsch/English)

### Test Image Generation

1. Open web interface
2. Select language
3. Enter a simple prompt: "ein roter Apfel" (a red apple)
4. Click "Transformieren" (Transform)
5. Select "Bild" (Image)
6. Wait for generation (~30-60 seconds)

**If image generates successfully, installation is complete!**

---

## Troubleshooting

### Services Won't Start

**Check logs:**
```bash
# SwarmUI logs
sudo journalctl -u ai4artsed-swarmui --no-pager -n 50

# Backend logs
sudo journalctl -u ai4artsed-backend --no-pager -n 50
```

**Common issues:**
- **Port already in use:** Another service using 7801 or 17801
  - Check: `sudo lsof -i :7801` or `sudo lsof -i :17801`
  - Solution: Stop conflicting service or change port in config.py

- **Models not found:** Model files missing or symlink broken
  - Check: Verify all models exist (see "Verify Model Installation" above)
  - Solution: Re-download missing models

- **Permission denied:** User doesn't have access to files
  - Solution: `sudo chown -R ai4artsed:ai4artsed /opt/ai4artsed`

### Image Generation Fails

**Check SwarmUI is running:**
```bash
curl http://localhost:7801/API/GetNewSession
# Should return JSON with session_id
```

**Check model paths in config.py:**
```bash
cd /opt/ai4artsed/ai4artsed_webserver/devserver
grep SWARMUI_BASE_PATH config.py
# Should match your actual SwarmUI path
```

**Check SwarmUI logs:**
```bash
sudo journalctl -u ai4artsed-swarmui -f
# Watch for errors during generation
```

### Out of Memory Errors

**Check VRAM usage:**
```bash
nvidia-smi
```

**Solutions:**
- Ensure no other GPU-intensive processes running
- Restart SwarmUI to clear VRAM: `sudo systemctl restart ai4artsed-swarmui`
- Check model quantization (FP8 models use less VRAM)

### Frontend Not Loading

**Check backend is serving frontend:**
```bash
ls /opt/ai4artsed/ai4artsed_webserver/public/ai4artsed-frontend/dist/index.html
# Should exist
```

**Rebuild frontend:**
```bash
cd /opt/ai4artsed/ai4artsed_webserver/public/ai4artsed-frontend
npm run build
sudo systemctl restart ai4artsed-backend
```

---

## Next Steps

### Configure Cloudflare Tunnel (Optional)

For public access via custom domain, see: [CLOUDFLARE_TUNNEL_SETUP.md](CLOUDFLARE_TUNNEL_SETUP.md)

### Update System

Use the provided update script:
```bash
cd /opt/ai4artsed/ai4artsed_webserver
./update.sh
```

### Monitor System

```bash
# View backend logs
sudo journalctl -u ai4artsed-backend -f

# View SwarmUI logs
sudo journalctl -u ai4artsed-swarmui -f

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## Getting Help

- **Documentation:** Check [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md), [MODELS_REQUIRED.md](MODELS_REQUIRED.md), [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)
- **Issues:** https://github.com/joerissenbenjamin/ai4artsed_webserver/issues
- **Project Website:** https://ai4artsed.org

---

**Installation Complete!** 🎉

Your AI4ArtsEd DevServer is now ready for educational use.
