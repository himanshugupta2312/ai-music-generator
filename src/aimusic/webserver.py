"""Local web server for the 3D web UI (standard library only).

    aimusic ui

Security model: it binds to 127.0.0.1 only, rejects requests whose Host header is not
this server (defends against DNS rebinding), requires JSON bodies for POST (a plain
cross-site form cannot send those), and rejects cross-origin POSTs.
"""

from __future__ import annotations

import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__
from .backends import BACKENDS
from .jobs import Busy, JobManager
from .pipeline import MAX_DURATION_S, available_backends, build_settings, default_backend, describe, validate
from .vibes import VIBES, surprise_prompt

WEB_DIR = Path(__file__).with_name("web")

BACKEND_INFO = {
    "acestep": ("ACE-Step", "one long, continuous piece"),
    "musicgen": ("MusicGen", "short 30 s clips"),
    "synth": ("Quick synth", "instant, toy-synth take on your style"),
}

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self' 'unsafe-inline'; media-src 'self'; connect-src 'self'; "
       "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")

_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9._-]+\.(wav|mp3)$")
_MAX_BODY = 200_000


class Handler(BaseHTTPRequestHandler):
    server_version = "aimusic"
    protocol_version = "HTTP/1.1"

    # ---- plumbing --------------------------------------------------------
    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass

    @property
    def mgr(self) -> JobManager:
        return self.server.manager  # type: ignore[attr-defined]

    def _allowed_hosts(self) -> set[str]:
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), MIME[".json"], {"Cache-Control": "no-store"})

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"error": message})

    def _guard(self) -> bool:
        """Common request checks. Returns False (after replying) if the request is refused."""
        if self.headers.get("Host", "") not in self._allowed_hosts():
            self._error(403, "Forbidden host.")
            return False
        if self.command == "POST":
            origin = self.headers.get("Origin")
            if origin and origin.replace("http://", "") not in self._allowed_hosts():
                self._error(403, "Cross-origin requests are not allowed.")
                return False
            if not (self.headers.get("Content-Type", "").lower().startswith("application/json")):
                self._error(415, "Expected application/json.")
                return False
        return True

    def _body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > _MAX_BODY:
            self._error(413, "Request too large.")
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            self._error(400, "Invalid JSON.")
            return None
        if not isinstance(data, dict):
            self._error(400, "Expected a JSON object.")
            return None
        return data

    def _file(self, path: Path, cache: str) -> None:
        """Serve a file, with HTTP Range support so the audio player can seek."""
        size = path.stat().st_size
        ctype = MIME.get(path.suffix.lower(), "application/octet-stream")
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if not m or (m.group(1) == "" and m.group(2) == ""):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if m.group(1) == "":                     # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            else:
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser closed the connection (normal when seeking)

    # ---- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._file(WEB_DIR / "index.html", "no-cache")
        if path.startswith("/static/"):
            target = (WEB_DIR / path[len("/static/"):]).resolve()
            if WEB_DIR.resolve() in target.parents and target.is_file():
                return self._file(target, "no-cache")
            return self._error(404, "Not found.")
        if path.startswith("/output/"):
            name = path[len("/output/"):]
            target = (self.mgr.out_dir / name).resolve()
            if _OUTPUT_NAME.match(name) and self.mgr.out_dir.resolve() in target.parents and target.is_file():
                return self._file(target, "no-store")
            return self._error(404, "Not found.")
        if path == "/api/info":
            avail = available_backends()
            default = default_backend()
            return self._json(200, {
                "version": __version__,
                "max_duration": MAX_DURATION_S,
                "default_backend": default,
                "backends": [
                    {"id": b, "name": BACKEND_INFO[b][0], "hint": BACKEND_INFO[b][1], "available": avail[b]}
                    for b in BACKENDS
                ],
                "vibes": VIBES,
            })
        if path == "/api/history":
            return self._json(200, {"tracks": self.mgr.history()})
        m = re.fullmatch(r"/api/jobs/([a-f0-9]{12})", path)
        if m:
            job = self.mgr.get(m.group(1))
            return self._json(200, job.snapshot()) if job else self._error(404, "Unknown job.")
        return self._error(404, "Not found.")

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        data = self._body()
        if data is None:
            return
        if path == "/api/surprise":
            return self._json(200, {"prompt": surprise_prompt()})
        if path == "/api/preview":
            try:
                s = build_settings(
                    str(data.get("prompt", "")),
                    duration=_num(data.get("duration")), tuning=_num(data.get("tuning")),
                    ambience=bool(data.get("ambience", True)), ambience_level=_num(data.get("level")),
                )
                warnings = []
                try:
                    warnings = validate(s, str(data.get("backend", "synth")))
                except ValueError as e:
                    return self._json(200, {"rows": [], "warnings": [], "error": str(e)})
            except (ValueError, TypeError) as e:
                return self._json(200, {"rows": [], "warnings": [], "error": str(e)})
            return self._json(200, {"rows": [{"label": k, "value": v} for k, v in describe(s)],
                                    "warnings": warnings, "error": None})
        if path == "/api/generate":
            backend = str(data.get("backend", "synth"))
            if backend not in BACKENDS:
                return self._error(400, f"Unknown backend '{backend}'.")
            try:
                job = self.mgr.start(
                    str(data.get("prompt", "")), backend,
                    duration=_num(data.get("duration")) or None, seed=data.get("seed"),
                    lyrics=str(data.get("lyrics") or ""), tuning=_num(data.get("tuning")) or None,
                    ambience=bool(data.get("ambience", True)), ambience_level=_num(data.get("level")),
                )
            except Busy as e:
                return self._error(409, str(e))
            except (ValueError, TypeError) as e:
                return self._error(400, str(e))
            return self._json(202, {"job": job.id})
        m = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/cancel", path)
        if m:
            job = self.mgr.get(m.group(1))
            if not job:
                return self._error(404, "Unknown job.")
            job.cancel()
            return self._json(200, job.snapshot())
        return self._error(404, "Not found.")


def _num(v):
    """Optional number from JSON: None/''/0 stay falsy-friendly; bad values raise ValueError."""
    if v is None or v == "":
        return None
    return float(v)


def make_server(port: int = 7860, out_dir: Path | None = None) -> ThreadingHTTPServer:
    out_dir = (out_dir or Path("output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    srv.manager = JobManager(out_dir)  # type: ignore[attr-defined]
    return srv


def run(port: int = 7860, open_browser: bool = True) -> None:
    srv = make_server(port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"aimusic is running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        job = srv.manager.current  # type: ignore[attr-defined]
        if job is not None and job.alive():
            job.cancel()
        srv.server_close()
