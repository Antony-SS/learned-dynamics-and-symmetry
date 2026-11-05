"""Local page: scrub a collection, play a trained unicycle open-loop."""

from __future__ import annotations

import json
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from unicycle.experiments.data import load_all, load_split
from unicycle.experiments.models import PHYSICS, TRAIN_FAMILIES, make_model
from unicycle.experiments.train.lib import _configure
from unicycle.experiments.utils.params import RUNS, VIZ

PAGE = VIZ / "interactive" / "index.html"
HOST = "127.0.0.1"
PORT = 8765


def list_checkpoints() -> list[dict]:
    if not RUNS.is_dir():
        return []
    rows = []
    for path in sorted(RUNS.glob("*/*/*.pt")):
        if "eval" in path.parts:
            continue
        family = path.parent.parent.name
        if family not in TRAIN_FAMILIES:
            continue
        rows.append(
            {
                "id": str(path.relative_to(RUNS)),
                "label": f"{family} / {path.parent.name} / {path.name}",
                "family": family,
                "path": str(path),
            }
        )
    return rows


@lru_cache(maxsize=8)
def _logs():
    return load_all()


@lru_cache(maxsize=16)
def _load_model(path: str, family: str):
    train, _val, _test = load_split(augment_train=family == "quotient_aug")
    model = _configure(family, train, torch.device("cpu"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _index_at(log, start_s: float) -> int:
    idx = int(round(float(start_s) / log.dt))
    return min(max(idx, 0), log.n_steps - 1)


@torch.no_grad()
def _roll(model, log, start: int) -> np.ndarray:
    pose = torch.tensor(log.poses[start], dtype=torch.float32).unsqueeze(0)
    rates = torch.tensor(log.rates[start], dtype=torch.float32).unsqueeze(0)
    commands = torch.tensor(log.commands[start:], dtype=torch.float32).unsqueeze(0)
    pred = model.rollout(pose, rates, commands, log.dt).squeeze(0).numpy()
    return np.vstack((log.poses[start], pred))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[interactive] {fmt % args}")

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._file(PAGE, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/models":
            self._json({"models": list_checkpoints()})
            return
        if parsed.path == "/api/collections":
            rows = [
                {"name": name, "duration": float((log.n_poses - 1) * log.dt), "dt": log.dt}
                for name, log in _logs().items()
            ]
            self._json({"collections": rows})
            return
        if parsed.path == "/api/collection":
            name = parse_qs(parsed.query).get("name", [""])[0]
            logs = _logs()
            if name not in logs:
                self._json({"error": f"unknown collection {name}"}, 404)
                return
            log = logs[name]
            self._json(
                {
                    "name": log.name,
                    "dt": log.dt,
                    "duration": float((log.n_poses - 1) * log.dt),
                    "x": log.poses[:, 0].tolist(),
                    "y": log.poses[:, 1].tolist(),
                    "yaw": log.poses[:, 2].tolist(),
                }
            )
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/rollout":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        logs = _logs()
        name = body.get("collection", "")
        if name not in logs:
            self._json({"error": f"unknown collection {name}"}, 404)
            return
        log = logs[name]
        start = _index_at(log, float(body.get("start_s", 0.0)))
        model_id = body.get("model", "")
        ckpt = RUNS / model_id
        family = ckpt.parent.parent.name if model_id else ""
        pred = None
        if model_id and ckpt.is_file() and family in TRAIN_FAMILIES:
            pred = _roll(_load_model(str(ckpt), family), log, start)
        phys = _roll(make_model(PHYSICS), log, start) if body.get("physics") else None
        payload = {
            "start": start,
            "dt": log.dt,
            "vicon": {
                "x": log.poses[start:, 0].tolist(),
                "y": log.poses[start:, 1].tolist(),
                "yaw": log.poses[start:, 2].tolist(),
            },
        }
        if pred is not None:
            payload["pred"] = {"x": pred[:, 0].tolist(), "y": pred[:, 1].tolist(), "yaw": pred[:, 2].tolist()}
        if phys is not None:
            payload["physics"] = {"x": phys[:, 0].tolist(), "y": phys[:, 1].tolist(), "yaw": phys[:, 2].tolist()}
        self._json(payload)


def main() -> None:
    if not PAGE.is_file():
        raise SystemExit(f"missing {PAGE}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"unicycle interactive  {url}", flush=True)
    print(f"models  {RUNS}", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
