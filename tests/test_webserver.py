import http.client
import json
import random
import threading
import time

import numpy as np
import pytest

from aimusic import jobs, pipeline, webserver
from aimusic.backends.base import Backend
from aimusic.vibes import VIBES, surprise_prompt


@pytest.fixture()
def server(tmp_path):
    srv = webserver.make_server(0, tmp_path / "output")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def call(srv, method, path, body=None, headers=None, raw=None):
    port = srv.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    hdrs = {"Host": f"127.0.0.1:{port}"}
    if body is not None:
        raw = json.dumps(body)
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    conn.request(method, path, body=raw, headers=hdrs)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, dict(r.getheaders()), data


def jcall(srv, method, path, body=None, headers=None):
    status, hdrs, data = call(srv, method, path, body, headers)
    return status, json.loads(data.decode("utf-8"))


def wait_done(srv, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, snap = jcall(srv, "GET", f"/api/jobs/{job_id}")
        if snap["state"] != "running":
            return snap
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_index_static_and_info(server):
    status, hdrs, body = call(server, "GET", "/")
    assert status == 200 and b"<canvas" in body and "text/html" in hdrs["Content-Type"]
    assert "frame-ancestors 'none'" in hdrs["Content-Security-Policy"]
    status, hdrs, _ = call(server, "GET", "/static/vendor/three.module.min.js")
    assert status == 200 and "javascript" in hdrs["Content-Type"]
    status, info = jcall(server, "GET", "/api/info")
    assert status == 200 and len(info["vibes"]) == 8 and info["default_backend"] in ("acestep", "musicgen", "synth")
    assert {b["id"] for b in info["backends"]} == {"acestep", "musicgen", "synth"}


def test_preview_and_surprise(server):
    status, out = jcall(server, "POST", "/api/preview", {"prompt": "2 min calm piano with rain at 450hz"})
    labels = {r["label"]: r["value"] for r in out["rows"]}
    assert status == 200 and labels["duration"].startswith("120s") and "rain" in labels["ambience"]
    assert "450" in labels["tuning"]
    _, over = jcall(server, "POST", "/api/preview", {"prompt": "11 min calm piano"})
    assert "10-minute" in over["error"]
    _, s = jcall(server, "POST", "/api/surprise", {})
    assert s["prompt"].strip()


def test_generate_with_synth_then_serve_audio_with_ranges(server):
    status, out = jcall(server, "POST", "/api/generate",
                        {"prompt": "calm piano with rain", "backend": "synth", "duration": 5, "seed": 3})
    assert status == 202
    snap = wait_done(server, out["job"])
    assert snap["state"] == "done" and snap["result"]["settings"]["seed"] == 3
    url = snap["result"]["audio_url"]
    status, hdrs, full = call(server, "GET", url)
    assert status == 200 and full[:4] == b"RIFF" and hdrs["Accept-Ranges"] == "bytes"
    status, hdrs, part = call(server, "GET", url, headers={"Range": "bytes=0-99"})
    assert status == 206 and len(part) == 100 and hdrs["Content-Range"].startswith("bytes 0-99/")
    status, _, tail = call(server, "GET", url, headers={"Range": "bytes=-10"})
    assert status == 206 and tail == full[-10:]
    status, _, _ = call(server, "GET", url, headers={"Range": "bytes=999999999-"})
    assert status == 416
    _, hist = jcall(server, "GET", "/api/history")
    assert [t["name"] for t in hist["tracks"]] == [snap["result"]["name"]]


def test_bad_input_and_busy(server, monkeypatch):
    assert jcall(server, "POST", "/api/generate", {"prompt": "", "backend": "synth"})[0] == 400
    assert jcall(server, "POST", "/api/generate", {"prompt": "x", "backend": "nope"})[0] == 400
    assert jcall(server, "POST", "/api/generate", {"prompt": "calm", "backend": "synth", "duration": 700})[0] == 400

    release = threading.Event()

    class Slow(Backend):
        name, sample_rate = "slow", 8000

        def generate(self, s):
            release.wait(10)
            return np.zeros((8000, 1), dtype=np.float32)

    monkeypatch.setattr(jobs, "get_backend", lambda name: Slow())
    status, first = jcall(server, "POST", "/api/generate", {"prompt": "calm piano", "backend": "synth", "duration": 5})
    assert status == 202
    assert jcall(server, "POST", "/api/generate", {"prompt": "calm piano", "backend": "synth"})[0] == 409
    release.set()
    assert wait_done(server, first["job"])["state"] == "done"


def test_cancel_calls_backend_cancel(server, monkeypatch):
    cancelled = threading.Event()

    class Interruptible(Backend):
        name, sample_rate = "slow", 8000

        def generate(self, s):
            for _ in range(200):
                if cancelled.is_set():
                    raise RuntimeError("Generation was cancelled.")
                time.sleep(0.05)
            return np.zeros((8000, 1), dtype=np.float32)

        def cancel(self):
            cancelled.set()

    monkeypatch.setattr(jobs, "get_backend", lambda name: Interruptible())
    _, out = jcall(server, "POST", "/api/generate", {"prompt": "calm piano", "backend": "synth", "duration": 5})
    time.sleep(0.5)
    status, snap = jcall(server, "POST", f"/api/jobs/{out['job']}/cancel", {})
    assert status == 200 and snap["state"] == "cancelled"
    assert cancelled.wait(3)
    snap = wait_done(server, out["job"])
    assert snap["state"] == "cancelled" and snap["result"] is None
    assert not list((server.manager.out_dir).glob("*.wav"))  # nothing half-written is left behind


def test_security_checks(server, tmp_path):
    port = server.server_address[1]
    # wrong Host header (DNS rebinding)
    assert call(server, "GET", "/api/info", headers={"Host": "evil.example"})[0] == 403
    # cross-origin POST
    assert call(server, "POST", "/api/surprise", {}, headers={"Origin": "http://evil.example"})[0] == 403
    assert call(server, "POST", "/api/surprise", {}, headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200
    # must be JSON (blocks cross-site simple form posts)
    assert call(server, "POST", "/api/surprise", raw="a=1", headers={"Content-Type": "text/plain"})[0] == 415
    # path traversal and non-audio files
    (tmp_path / "secret.txt").write_text("x")
    assert call(server, "GET", "/static/../../../etc/passwd")[0] == 404
    assert call(server, "GET", "/static/%2e%2e/%2e%2e/pyproject.toml")[0] == 404
    assert call(server, "GET", "/output/../secret.txt")[0] == 404
    (server.manager.out_dir / "notes.txt").write_text("x")
    assert call(server, "GET", "/output/notes.txt")[0] == 404
    # malformed bodies
    assert call(server, "POST", "/api/preview", raw="{not json", headers={"Content-Type": "application/json"})[0] == 400
    assert call(server, "POST", "/api/preview", raw="[1,2]", headers={"Content-Type": "application/json"})[0] == 400


def test_vibes_and_surprise_prompts_are_valid():
    rng = random.Random(0)
    prompts = [surprise_prompt(rng) for _ in range(200)] + [v["prompt"] for v in VIBES]
    for p in prompts:
        pipeline.validate(pipeline.build_settings(p), "acestep")
    assert len(set(prompts)) > 20
    assert all(v["emoji"] and v["name"] and v["c1"].startswith("#") and v["c2"].startswith("#") for v in VIBES)
    assert len({v["id"] for v in VIBES}) == len(VIBES)
