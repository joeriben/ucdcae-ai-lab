#!/bin/bash
# Setup passwordless sudo for Ollama service management.
# This allows the DevServer to automatically restart Ollama
# when the safety system detects it has become unresponsive.
#
# Run once during system setup: sudo ./0_setup_ollama_watchdog.sh
#
# Security: Only grants access to start/stop/restart ollama — no other sudo rights.
# Safe for dedicated workshop PCs without competing inference systems.

set -e

# Detect the user who will run the DevServer
if [ -n "$SUDO_USER" ]; then
    TARGET_USER="$SUDO_USER"
else
    TARGET_USER="$(whoami)"
fi

SUDOERS_FILE="/etc/sudoers.d/ai4artsed-ollama"

# Must run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo ./0_setup_ollama_watchdog.sh)"
    exit 1
fi

# Find systemctl path
SYSTEMCTL_PATH="$(which systemctl 2>/dev/null || echo "/usr/bin/systemctl")"

echo "Setting up passwordless Ollama restart for user: $TARGET_USER"
echo "systemctl path: $SYSTEMCTL_PATH"

# Write sudoers rule
cat > "$SUDOERS_FILE" << EOF
# AI4ArtsEd Ollama Watchdog — auto-restart on safety system failure
$TARGET_USER ALL=(ALL) NOPASSWD: $SYSTEMCTL_PATH restart ollama, $SYSTEMCTL_PATH start ollama, $SYSTEMCTL_PATH stop ollama
EOF

chmod 440 "$SUDOERS_FILE"

# Validate sudoers syntax
if visudo -c -f "$SUDOERS_FILE" 2>/dev/null; then
    echo "OK: Sudoers rule installed at $SUDOERS_FILE"
    echo "    $TARGET_USER can now restart Ollama without password."
    echo ""
    echo "Test: sudo systemctl restart ollama  (should work without password prompt)"
else
    echo "ERROR: Sudoers syntax check failed — removing file"
    rm -f "$SUDOERS_FILE"
    exit 1
fi
