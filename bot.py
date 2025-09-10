import os
import json
import time
import ssl
import socket
import threading
import webbrowser
import requests
import subprocess
import re
import ffmpeg
import numpy as np
from collections import deque
from pathlib import Path
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from streamlink import Streamlink

TOKENS_PATH = Path("tokens.json")
COOLDOWN_SEC = 60
CLIP_URL_POLL_TRIES = 6
CLIP_URL_POLL_SLEEP = 2.0

# === Load env ===
load_dotenv(dotenv_path="client.env", override=True)

CLIENT_ID = os.getenv("CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "").strip()
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost").strip() or "http://localhost"
CODE = os.getenv("CODE", "").strip()
TARGET_LOGIN = os.getenv("TARGET_LOGIN", "").strip().lower()

# Chat thresholds
CHAT_WINDOW_SEC = int(os.getenv("CHAT_WINDOW_SEC", "10"))
CHAT_MSG_THRESHOLD = int(os.getenv("CHAT_MSG_THRESHOLD", "30"))
CHAT_UNIQUE_THRESHOLD = int(os.getenv("CHAT_UNIQUE_THRESHOLD", "10"))
AUTOCLIP_ENABLED = os.getenv("AUTOCLIP_ENABLED", "true").lower() in ("1", "true", "yes", "y")

# Voice (HLS) trigger settings
VOICE_FROM_STREAM = os.getenv("VOICE_FROM_STREAM", "false").lower() in ("1","true","yes","y")
VOICE_PHRASES = [p.strip() for p in os.getenv("VOICE_PHRASES", "clip this").split(",") if p.strip()]
VOICE_DELAY_SEC = int(os.getenv("VOICE_DELAY_SEC", "10"))
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "").strip()

# Loudness (HLS) trigger env
LOUD_ENABLED = os.getenv("LOUD_ENABLED", "true").lower() in ("1","true","yes","y")
LOUD_WINDOW_SEC = float(os.getenv("LOUD_WINDOW_SEC", "8"))
LOUD_MIN_DB_ABOVE_AVG = float(os.getenv("LOUD_MIN_DB_ABOVE_AVG", "8"))
LOUD_MIN_DURATION = float(os.getenv("LOUD_MIN_DURATION", "1.0"))
LOUD_MIN_GAP = float(os.getenv("LOUD_MIN_GAP", "20"))
VOICE_DEBUG = os.getenv("VOICE_DEBUG","false").lower() in ("1","true","yes","y")
LOUD_DEBUG = os.getenv("LOUD_DEBUG","false").lower() in ("1","true","yes","y")

# ---------- OAuth helpers ----------
def _fail_if_missing_core():
    miss = []
    if not CLIENT_ID: miss.append("CLIENT_ID")
    if not CLIENT_SECRET: miss.append("CLIENT_SECRET")
    if not REDIRECT_URI: miss.append("REDIRECT_URI")
    if miss:
        raise SystemExit(f"Missing envs: {', '.join(miss)}. Fix client.env.")

def auth_url():
    base = "https://id.twitch.tv/oauth2/authorize"
    scopes = ["clips:edit", "chat:read", "chat:edit"]
    scope_str = "+".join(scopes)
    return (
        f"{base}?client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={scope_str}"
    )

def exchange_code_for_tokens(code: str):
    url = "https://id.twitch.tv/oauth2/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }
    r = requests.post(url, data=data, timeout=30)
    return r.status_code, r.json()

def refresh_tokens(refresh_token: str):
    url = "https://id.twitch.tv/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    r = requests.post(url, data=data, timeout=30)
    return r.status_code, r.json()

def validate_token(access_token: str):
    url = "https://id.twitch.tv/oauth2/validate"
    headers = {"Authorization": f"OAuth {access_token}"}
    r = requests.get(url, headers=headers, timeout=15)
    return r.status_code, r.json()

def save_tokens(tokens: dict):
    TOKENS_PATH.write_text(json.dumps(tokens, indent=2))

def load_tokens():
    if TOKENS_PATH.exists():
        return json.loads(TOKENS_PATH.read_text())
    return None

def ensure_tokens():
    _fail_if_missing_core()
    t = load_tokens()
    if t and "access_token" in t:
        status, payload = validate_token(t["access_token"])
        if status == 200:
            return t
        if t.get("refresh_token"):
            rs, pr = refresh_tokens(t["refresh_token"])
            if rs == 200 and "access_token" in pr:
                save_tokens(pr); return pr
    if not CODE:
        url = auth_url()
        print("Authorize at:", url)
        raise SystemExit("Add CODE=<value> to client.env and re-run.")
    status, payload = exchange_code_for_tokens(CODE)
    if status == 200 and "access_token" in payload:
        save_tokens(payload); return payload
    raise SystemExit("Token error: " + str(payload))

# ---------- Helix helpers ----------
def helix_users(access_token: str, login: str = None):
    headers = {"Authorization": f"Bearer {access_token}", "Client-Id": CLIENT_ID}
    params = {"login": login} if login else None
    r = requests.get("https://api.twitch.tv/helix/users", headers=headers, params=params, timeout=15)
    return r.json()

def get_broadcaster_id(access_token: str, login: str) -> str:
    data = helix_users(access_token, login=login)
    return data["data"][0]["id"]

def get_login_self(access_token: str) -> str:
    data = helix_users(access_token)
    return data["data"][0]["login"].lower()

def is_channel_live(access_token: str, user_id: str) -> bool:
    headers = {"Authorization": f"Bearer {access_token}", "Client-Id": CLIENT_ID}
    r = requests.get("https://api.twitch.tv/helix/streams",
                     params={"user_id": user_id}, headers=headers, timeout=15)
    return bool(r.json().get("data"))

def create_clip(access_token: str, broadcaster_id: str):
    headers = {"Authorization": f"Bearer {access_token}", "Client-Id": CLIENT_ID}
    r = requests.post("https://api.twitch.tv/helix/clips",
                      params={"broadcaster_id": broadcaster_id},
                      headers=headers, timeout=20)
    return r.status_code, r.text

def get_clip_url(clip_id: str, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}", "Client-Id": CLIENT_ID}
    r = requests.get("https://api.twitch.tv/helix/clips",
                     params={"id": clip_id}, headers=headers, timeout=15)
    d = r.json().get("data", [])
    if d: return d[0]["url"], d[0].get("vod_offset")
    return None, None

def print_clip_result(access_token: str, status: int, text: str):
    """Print basic clip creation result and, on success, the final clip link.

    Parses the clip ID from the create response and polls Helix for the
    published clip URL, printing a final link when available.
    """
    try:
        print("Clip →", status, text)
        if status not in (200, 202):
            return

        clip_id = None
        edit_url = None
        try:
            payload = json.loads(text)
            data0 = (payload.get("data") or [{}])[0]
            clip_id = data0.get("id")
            edit_url = data0.get("edit_url")
        except Exception:
            pass

        if edit_url:
            print("Edit clip:", edit_url)

        if clip_id:
            final_url = None
            vod_offset = None
            for _ in range(CLIP_URL_POLL_TRIES):
                u, off = get_clip_url(clip_id, access_token)
                if u:
                    final_url, vod_offset = u, off
                    break
                time.sleep(CLIP_URL_POLL_SLEEP)
            if final_url:
                print("Final clip link:", final_url)
            else:
                # Fallback best-guess link shape
                print("Final clip link:", f"https://clips.twitch.tv/{clip_id}")
        else:
            print("Note: Could not parse clip ID from response.")
    except Exception as e:
        # Never crash on printing; just show raw response
        print("Clip (raw) →", status, text, "error:", e)

# ---------- IRC Chat Stats ----------
class ChatStats:
    def __init__(self, window_sec=10):
        self.window = window_sec
        self.msg_times = deque()
        self.user_times = deque()
        self.lock = threading.Lock()

    def add_message(self, user, ts):
        with self.lock:
            self.msg_times.append(ts)
            self.user_times.append((ts, user))
            cutoff = ts - self.window
            while self.msg_times and self.msg_times[0] < cutoff:
                self.msg_times.popleft()
            while self.user_times and self.user_times[0][0] < cutoff:
                self.user_times.popleft()

    def snapshot(self):
        with self.lock:
            now = time.time()
            cutoff = now - self.window
            while self.msg_times and self.msg_times[0] < cutoff:
                self.msg_times.popleft()
            while self.user_times and self.user_times[0][0] < cutoff:
                self.user_times.popleft()
            return len(self.msg_times), len({u for _, u in self.user_times})

class TwitchIRC(threading.Thread):
    def __init__(self, oauth_token, login_name, channel_login, stats):
        super().__init__(daemon=True)
        self.oauth_token = oauth_token
        self.nick = login_name
        self.channel = f"#{channel_login}"
        self.stats = stats
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                self._connect()
            except Exception as e:
                print("IRC disconnected:", e)
                time.sleep(3)

    def stop(self):
        self.stop_event.set()

    def _connect(self):
        host, port = "irc.chat.twitch.tv", 6697
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port)) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as s:
                s.sendall(f"PASS oauth:{self.oauth_token}\r\n".encode())
                s.sendall(f"NICK {self.nick}\r\n".encode())
                s.sendall(f"JOIN {self.channel}\r\n".encode())
                print(f"IRC connected → {self.channel}")
                buf = b""
                while not self.stop_event.is_set():
                    data = s.recv(4096)
                    if not data: break
                    buf += data
                    while b"\r\n" in buf:
                        line, buf = buf.split(b"\r\n", 1)
                        self._handle_line(s, line.decode(errors="ignore"))

    def _handle_line(self, s, line: str):
        if line.startswith("PING"):
            s.sendall(b"PONG :tmi.twitch.tv\r\n")
        elif "PRIVMSG" in line:
            try:
                prefix, _, _ = line.partition(" :")
                user = prefix.split("!", 1)[0].lstrip(":").split("@", 1)[0].lower()
                self.stats.add_message(user, time.time())
            except: pass

# ---------- Voice Trigger from HLS ----------
class StreamAudioListener(threading.Thread):
    def __init__(self, access_token, channel_login, model_path, phrases):
        super().__init__(daemon=True)
        self.access_token = access_token
        self.channel_login = channel_login
        self.model_path = model_path
        self.phrases = [p.lower() for p in phrases]
        self._hits = deque()
        self._stop = threading.Event()
        # Loudness state
        self.loud_enabled = LOUD_ENABLED
        self.loud_window = LOUD_WINDOW_SEC
        self.loud_thresh_db = LOUD_MIN_DB_ABOVE_AVG
        self.loud_min_dur = LOUD_MIN_DURATION
        self._loud_buf = deque()      # stores (timestamp, dBFS)
        self._loud_hits = deque()     # timestamps when spike detected
        self._above_since = None
        self._loud_lock = threading.Lock()

    def stop(self): self._stop.set()
    def has_hit(self): return bool(self._hits)
    def pop_hit(self): return self._hits.popleft() if self._hits else None

    def _norm_text(self, s):
        try: t = json.loads(s).get("text", "")
        except: t = s
        return re.sub(r"[^a-z0-9\s]", " ", t.lower()).strip()

    def _rms_db(self, pcm_bytes: bytes) -> float:
        """Return RMS in dBFS for int16 mono PCM; safe for silence."""
        if not pcm_bytes:
            return -120.0
        s = np.frombuffer(pcm_bytes, dtype=np.int16)
        if s.size == 0:
            return -120.0
        s = s.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(s * s))
        if rms <= 1e-8:
            return -120.0
        return 20.0 * np.log10(rms)  # 0 dBFS is full scale

    def _update_loudness(self, dbfs: float, ts: float):
        """Maintain rolling average and detect spikes sustained over loud_min_dur."""
        if not self.loud_enabled:
            return
        with self._loud_lock:
            self._loud_buf.append((ts, dbfs))
            cutoff = ts - self.loud_window
            while self._loud_buf and self._loud_buf[0][0] < cutoff:
                self._loud_buf.popleft()

            if len(self._loud_buf) < 2:
                self._above_since = None
                return

            avg = sum(v for _, v in self._loud_buf) / len(self._loud_buf)
            if LOUD_DEBUG:
                try:
                    delta = dbfs - avg
                    print(f"[LOUD] dbfs={dbfs:.1f} avg={avg:.1f} Δ={delta:.1f}")
                except Exception:
                    pass

            above = dbfs >= (avg + self.loud_thresh_db)
            if above:
                if self._above_since is None:
                    self._above_since = ts
                if ts - self._above_since >= self.loud_min_dur:
                    self._loud_hits.append(ts)
                    self._above_since = None
            else:
                self._above_since = None

    def has_loud_spike(self) -> bool:
        with self._loud_lock:
            return bool(self._loud_hits)

    def pop_loud_spike(self):
        with self._loud_lock:
            return self._loud_hits.popleft() if self._loud_hits else None

    def run(self):
        if not os.path.isdir(self.model_path):
            print("No Vosk model at", self.model_path); return
        model = Model(self.model_path)
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(False)

        # Resolve HLS audio URL with Streamlink.
        # Prefer anonymous/public access first (works for most public streams).
        try:
            session = Streamlink()
            # Try without any auth first — most public channels don't need a token.
            streams = session.streams(f"https://twitch.tv/{self.channel_login}")
            stream = streams.get("audio_only") or streams.get("worst")
            if not stream:
                # Fallback path: try to attach Authorization header if available,
                # useful for some edge cases / older streamlink builds.
                try:
                    session = Streamlink()
                    session.set_option("http-headers", {"Authorization": f"OAuth {self.access_token}"})
                    streams = session.streams(f"https://twitch.tv/{self.channel_login}")
                    stream = streams.get("audio_only") or streams.get("worst")
                except Exception:
                    stream = None
            if not stream:
                print("StreamAudioListener: no audio stream available (channel offline or auth required).")
                print("Tip: If the channel is sub-only, Streamlink needs a web OAuth token or cookies.")
                return
            hls_url = stream.to_url()
        except Exception as e:
            print("StreamAudioListener: failed to resolve HLS:", e)
            print("Tip: `pip install --upgrade streamlink` (or use a public/non sub-only stream).")
            return

        try:
            proc = (ffmpeg.input(hls_url, vn=None)
                          .output("pipe:1", format="s16le", acodec="pcm_s16le", ac=1, ar="16000")
                          .compile())
            p = subprocess.Popen(proc, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            print("ffmpeg error:", e); return

        print(f"🎧 Listening to Twitch audio for '{self.channel_login}'")
        try:
            while not self._stop.is_set():
                chunk = p.stdout.read(32000)
                if not chunk: break
                # Update loudness metrics every ~1s chunk
                db = self._rms_db(chunk)
                self._update_loudness(db, time.time())
                if rec.AcceptWaveform(chunk):
                    text = self._norm_text(rec.Result())
                    if VOICE_DEBUG and text:
                        print("📝 voice:", text)
                    for phrase in self.phrases:
                        if phrase in text:
                            print("🎙️ Heard voice trigger:", text)
                            self._hits.append(time.time())
                            break
        finally:
            try: p.kill()
            except: pass

# ---------- Loops ----------
def wait_until_live(access_token, broadcaster_id):
    while not is_channel_live(access_token, broadcaster_id):
        print("…waiting for channel to go live")
        time.sleep(5)
    print("✅ Channel is live")

def interactive_clip_loop(access_token, broadcaster_id, stats, hls_voice):
    # Do not block on live status; allow user to quit anytime.
    print("Press Enter to clip (when live), q to quit.")
    last_clip = 0.0

    def watcher():
        nonlocal last_clip
        while True:
            time.sleep(1)
            now = time.time()

            # Loudness spike trigger (from HLS)
            if hls_voice and LOUD_ENABLED and hls_voice.has_loud_spike() and is_channel_live(access_token, broadcaster_id):
                if now - last_clip >= max(COOLDOWN_SEC, LOUD_MIN_GAP):
                    hls_voice.pop_loud_spike()
                    print(f"\n🔊 Loudness spike (>{LOUD_MIN_DB_ABOVE_AVG} dB over avg) → waiting {VOICE_DELAY_SEC}s…")
                    time.sleep(VOICE_DELAY_SEC)
                    if is_channel_live(access_token, broadcaster_id):
                        st, txt = create_clip(access_token, broadcaster_id)
                        print_clip_result(access_token, st, txt)
                        if st in (200, 202):
                            last_clip = time.time()
                    continue

            # Voice trigger
            if hls_voice and hls_voice.has_hit() and is_channel_live(access_token, broadcaster_id):
                if now - last_clip >= COOLDOWN_SEC:
                    hls_voice.pop_hit()
                    print("🔔 Voice trigger → waiting", VOICE_DELAY_SEC, "s…")
                    time.sleep(VOICE_DELAY_SEC)
                    if is_channel_live(access_token, broadcaster_id):
                        st, txt = create_clip(access_token, broadcaster_id)
                        print_clip_result(access_token, st, txt)
                        if st in (200,202): last_clip = time.time()

            # Chat spike trigger
            if AUTOCLIP_ENABLED and is_channel_live(access_token, broadcaster_id):
                msgs, uniq = stats.snapshot()
                if msgs >= CHAT_MSG_THRESHOLD and uniq >= CHAT_UNIQUE_THRESHOLD:
                    if now - last_clip >= COOLDOWN_SEC:
                        print("🔥 Chat spike → waiting 10s…")
                        time.sleep(10)
                        if is_channel_live(access_token, broadcaster_id):
                            st, txt = create_clip(access_token, broadcaster_id)
                            print_clip_result(access_token, st, txt)
                            if st in (200,202): last_clip = time.time()

    threading.Thread(target=watcher, daemon=True).start()

    while True:
        try:
            s = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if s == "q":
            print("Bye.")
            break

        # Proper cooldown check (no walrus/precedence bug)
        elapsed = time.time() - last_clip
        if elapsed < COOLDOWN_SEC:
            wait = int(COOLDOWN_SEC - elapsed)
            print(f"⏳ Cooldown. Try again in ~{wait}s.")
            continue

        # Ensure channel is live before attempting a manual clip
        if not is_channel_live(access_token, broadcaster_id):
            print("Channel is not live yet. Try again later or press q to quit.")
            continue

        print("Manual clip → waiting 10s…")
        time.sleep(10)
        st, txt = create_clip(access_token, broadcaster_id)
        print_clip_result(access_token, st, txt)
        if st in (200, 202):
            last_clip = time.time()

def main():
    tokens = ensure_tokens()
    access = tokens["access_token"]
    my_login = get_login_self(access)
    channel_login = TARGET_LOGIN or my_login
    broadcaster_id = get_broadcaster_id(access, channel_login)
    print("Target:", channel_login, broadcaster_id)

    stats = ChatStats(window_sec=CHAT_WINDOW_SEC)
    irc = TwitchIRC(access, my_login, channel_login, stats); irc.start()

    hls_voice = None
    if VOICE_FROM_STREAM and VOSK_MODEL_PATH:
        hls_voice = StreamAudioListener(access, channel_login, VOSK_MODEL_PATH, VOICE_PHRASES)
        hls_voice.start()

    try:
        interactive_clip_loop(access, broadcaster_id, stats, hls_voice)
    finally:
        irc.stop()
        if hls_voice: hls_voice.stop()

if __name__ == "__main__":
    main()
