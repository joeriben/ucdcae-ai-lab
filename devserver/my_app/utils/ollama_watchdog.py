"""
Ollama Watchdog — Self-healing for Ollama service failures.

When the circuit breaker trips (Ollama unreachable), the watchdog
attempts to restart Ollama via systemctl. Requires passwordless sudo
for the ollama service (see setup below).

Setup (one-time, as root):
    echo "USERNAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart ollama, \\
    /usr/bin/systemctl start ollama, /usr/bin/systemctl stop ollama" \\
    | sudo tee /etc/sudoers.d/ai4artsed-ollama

If the sudoers rule is not configured, the watchdog degrades gracefully
to a human-readable error message — no crash, no exception.
"""

import logging
import subprocess
import time
import threading

import requests

logger = logging.getLogger(__name__)

# Restart cooldown: max 1 restart attempt per 5 minutes
_RESTART_COOLDOWN_SECONDS = 300
_RESTART_TIMEOUT_SECONDS = 15
_HEALTH_CHECK_INTERVAL = 2.0
_HEALTH_CHECK_MAX_WAIT = 30.0

_last_restart_attempt = 0.0
_restart_lock = threading.Lock()


def _ollama_healthy(ollama_url: str) -> bool:
    """Quick health check — can Ollama respond?"""
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _can_restart() -> bool:
    """Check if enough time has passed since the last restart attempt."""
    return (time.time() - _last_restart_attempt) >= _RESTART_COOLDOWN_SECONDS


def attempt_restart() -> bool:
    """
    Attempt to restart Ollama via systemctl.

    Returns True if Ollama is healthy after the restart attempt.
    Returns False if restart failed or is on cooldown.

    Thread-safe: only one restart attempt at a time.
    """
    global _last_restart_attempt

    if not _restart_lock.acquire(blocking=False):
        logger.info("[OLLAMA-WATCHDOG] Restart already in progress, skipping")
        return False

    try:
        if not _can_restart():
            remaining = _RESTART_COOLDOWN_SECONDS - (time.time() - _last_restart_attempt)
            logger.info(f"[OLLAMA-WATCHDOG] Restart on cooldown ({remaining:.0f}s remaining)")
            return False

        _last_restart_attempt = time.time()
        logger.warning("[OLLAMA-WATCHDOG] Attempting Ollama restart via systemctl...")

        try:
            result = subprocess.run(
                ["sudo", "systemctl", "restart", "ollama"],
                capture_output=True,
                text=True,
                timeout=_RESTART_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            logger.warning("[OLLAMA-WATCHDOG] systemctl not found — not a systemd system")
            return False
        except subprocess.TimeoutExpired:
            logger.error("[OLLAMA-WATCHDOG] systemctl restart timed out after "
                         f"{_RESTART_TIMEOUT_SECONDS}s")
            return False

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "password" in stderr.lower() or "sudo" in stderr.lower():
                logger.warning(
                    "[OLLAMA-WATCHDOG] sudo requires password — passwordless rule not configured. "
                    "Setup: echo 'USERNAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart ollama' "
                    "| sudo tee /etc/sudoers.d/ai4artsed-ollama"
                )
            else:
                logger.error(f"[OLLAMA-WATCHDOG] systemctl restart failed: {stderr}")
            return False

        logger.info("[OLLAMA-WATCHDOG] systemctl restart succeeded, waiting for health...")

        # Wait for Ollama to become healthy
        from config import OLLAMA_API_BASE_URL
        ollama_url = OLLAMA_API_BASE_URL.rstrip('/')

        start = time.time()
        while (time.time() - start) < _HEALTH_CHECK_MAX_WAIT:
            if _ollama_healthy(ollama_url):
                duration = time.time() - start
                logger.info(f"[OLLAMA-WATCHDOG] Ollama healthy after {duration:.1f}s")
                return True
            time.sleep(_HEALTH_CHECK_INTERVAL)

        logger.error(f"[OLLAMA-WATCHDOG] Ollama not healthy after {_HEALTH_CHECK_MAX_WAIT}s")
        return False

    finally:
        _restart_lock.release()
