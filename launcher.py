#!/usr/bin/env python3
"""
Twitch Live Auto-Launcher
- Polls Twitch Helix for a streamer's live status
- Starts your clip bot when live; stops it when offline
- Safe shutdown, token refresh, jitter/backoff, and simple debouncing
"""

import os
import sys
import time
import json
import signal
import random
import requests
import subprocess
from typing import Optional

# --------------------------
# Helpers
# --------------------------
def _fmt_dur(secs: float) -> str:
    secs = int(max(0, secs))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# --------------------------
# Env loading
# --------------------------
try:
    from dotenv import load_dotenv
    # Load a dedicated automation env first (if present), then a generic .env fallback.
    # Use override=True so shell/CI can still supersede values explicitly.
    if os.path.exists("automate.env"):
        load_dotenv(dotenv_path="automate.env", override=True)
    else:
        load_dotenv(override=True)
except Exception:
    # dotenv not installed is fine if envs are provided by the environment.
    pass

# --------------------------
# Config (env or defaults)
# --------------------------
STREAMER_LOGIN = os.getenv("STREAMER_LOGIN", "some_streamer").strip().lower()  # e.g. xqc
CLIENT_ID      = os.getenv("TWITCH_CLIENT_ID", "").strip()
CLIENT_SECRET  = os.getenv("TWITCH_CLIENT_SECRET", "").strip()
POLL_SEC       = float(os.getenv("POLL_INTERVAL_SEC", "30"))   # how often to check live/offline
OFFLINE_GRACE  = float(os.getenv("OFFLINE_GRACE_SEC", "45"))   # require this long offline before stopping
LIVE_GRACE     = float(os.getenv("LIVE_GRACE_SEC", "10"))      # small delay after going live before start
LAUNCH_CMD     = os.getenv("LAUNCH_CMD", f"{sys.executable} bot.py").strip()  # command to run your bot
RESTART_BACKOFF_MAX = float(os.getenv("RESTART_BACKOFF_MAX_SEC", "60"))  # cap for restarts
USER_AGENT     = "TwitchAutoLauncher/1.0 (+https://twitch.tv)"

if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET env vars (see automate.env).")

# --------------------------
# OAuth: App Access Token
# --------------------------
_token: Optional[str] = None
_token_expiry_ts: float = 0.0

def _fetch_app_token() -> None:
    global _token, _token_expiry_ts
    url = "https://id.twitch.tv/oauth2/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    r = requests.post(url, data=data, timeout=20)
    r.raise_for_status()
    js = r.json()
    _token = js["access_token"]
    # expires_in is seconds from now
    _token_expiry_ts = time.time() + max(0, int(js.get("expires_in", 3600)) - 60)  # refresh 60s early

def _ensure_token() -> str:
    if not _token or time.time() >= _token_expiry_ts:
        _fetch_app_token()
    return _token

def _helix_headers() -> dict:
    return {
        "Client-Id": CLIENT_ID,
        "Authorization": f"Bearer {_ensure_token()}",
        "User-Agent": USER_AGENT,
    }

# --------------------------
# Stream status
# --------------------------
def is_live(user_login: str) -> bool:
    url = "https://api.twitch.tv/helix/streams"
    params = {"user_login": user_login.lower()}
    try:
        r = requests.get(url, headers=_helix_headers(), params=params, timeout=20)
        if r.status_code == 401:  # token expired/invalid -> refresh once
            _fetch_app_token()
            r = requests.get(url, headers=_helix_headers(), params=params, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[warn] Helix request failed: {e}")
        return False  # fail-closed: treat as offline to avoid flapping
    js = r.json()
    data = js.get("data", [])
    return bool(data) and data[0].get("type") == "live"

# --------------------------
# Child process management
# --------------------------
_child: Optional[subprocess.Popen] = None
_stop = False

def _spawn_child():
    global _child
    print(f"[info] Starting bot: {LAUNCH_CMD}")
    if os.name == "nt":
        _child = subprocess.Popen(LAUNCH_CMD, shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        _child = subprocess.Popen(LAUNCH_CMD, shell=True, preexec_fn=os.setsid)

def _terminate_child():
    global _child
    if not _child:
        return
    print("[info] Stopping bot...")
    try:
        if os.name == "nt":
            _child.terminate()
        else:
            os.killpg(os.getpgid(_child.pid), signal.SIGTERM)
    except Exception as e:
        print(f"[warn] Terminate failed: {e}")
    try:
        _child.wait(timeout=10)
    except Exception:
        try:
            if os.name == "nt":
                _child.kill()
            else:
                os.killpg(os.getpgid(_child.pid), signal.SIGKILL)
        except Exception:
            pass
    _child = None

def _signal_handler(sig, frame):
    global _stop
    print(f"[info] Caught signal {sig}; shutting down.")
    _stop = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# --------------------------
# Main loop
# --------------------------
def main():
    global _child
    print(json.dumps({
        "streamer": STREAMER_LOGIN,
        "poll_sec": POLL_SEC,
        "offline_grace_sec": OFFLINE_GRACE,
        "live_grace_sec": LIVE_GRACE,
        "launch_cmd": LAUNCH_CMD
    }, indent=2))

    start_ts = time.time()
    last_live = False
    offline_since: Optional[float] = None
    restart_backoff = 2.0

    while not _stop:
        live_now = is_live(STREAMER_LOGIN)
        child_running = (_child is not None and _child.poll() is None)
        print(f"[tick] uptime={_fmt_dur(time.time() - start_ts)} state={'LIVE' if live_now else 'OFF'} child={'RUNNING' if child_running else 'STOPPED'}")

        # --- handle going live
        if live_now and not last_live:
            print("[info] Detected LIVE. Arming start after grace...")
            time.sleep(LIVE_GRACE)
            # re-check to avoid brief blips
            if is_live(STREAMER_LOGIN):
                if _child is None or _child.poll() is not None:
                    _spawn_child()
                    restart_backoff = 2.0  # reset backoff after a successful start
                last_live = True
                offline_since = None
            else:
                print("[info] Live blip ended during grace; not starting.")
                last_live = False

        # --- while live: ensure child is running; restart if it crashed
        if live_now and last_live:
            if _child is not None and _child.poll() is not None:
                delay = min(restart_backoff, RESTART_BACKOFF_MAX) + random.uniform(0, 1.25)
                print(f"[warn] Bot exited unexpectedly while live. Restarting in {delay:.1f}s...")
                time.sleep(delay)
                _spawn_child()
                restart_backoff = min(RESTART_BACKOFF_MAX, restart_backoff * 2)

        # --- handle going offline
        if not live_now:
            if offline_since is None:
                offline_since = time.time()
            elapsed = time.time() - offline_since
            if elapsed >= OFFLINE_GRACE and _child is not None:
                _terminate_child()
            last_live = False

        # small sleep
        time.sleep(POLL_SEC)

    # graceful shutdown
    _terminate_child()
    print("[info] Exited.")

if __name__ == "__main__":
    main()
