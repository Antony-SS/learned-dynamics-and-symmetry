"""Local page: Melon scrub, two-model Plotly rollout, matching channel plots."""

from __future__ import annotations

import json
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from quadrotor.geometry import _integrate
from quadrotor.idsia.baselines import load_phys_res, load_physics
from quadrotor.idsia.data import load_idsia_prepared
from quadrotor.idsia.train.lib import load_trained
from quadrotor.idsia.utils.idsia_training_reproduction_params import PHYS_RES, TRAIN_FAMILIES, run_roots

PAGE = Path(__file__).resolve().parent / "index.html"
HOST = "127.0.0.1"
PORT = 8766
HISTORY_S = 1.0
DT = 0.01
AXIS_M = 0.12


def list_models() -> list[dict]:
    rows = [
        {"id": "physics", "label": "physics (analytic)"},
        {"id": PHYS_RES, "label": "phys_res (published checkpoint)"},
    ]
    for root in run_roots():
        for family in TRAIN_FAMILIES:
            folder = root / family
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("seed_*/best.pt")):
                rows.append(
                    {
                        "id": f"{family}/{path.parent.name}/{path.name}",
                        "label": f"{family} / {path.parent.name} / {path.name}",
                    }
                )
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        unique.append(row)
    return unique


@lru_cache(maxsize=1)
def _melon():
    _train, _val, test = load_idsia_prepared()
    return {flight.name: flight for flight in test}


@lru_cache(maxsize=16)
def _load(model_id: str):
    if model_id == PHYS_RES:
        return load_phys_res(device="cpu")
    if model_id == "physics":
        model = load_physics("cpu")
        model.eval()
        return model
    family, seed_name, _name = model_id.split("/")
    seed = int(seed_name.split("_")[1])
    return load_trained(family, seed, device=torch.device("cpu"))


def _rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-rotation[:, 2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
    yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return np.unwrap(np.stack((roll, pitch, yaw), axis=-1), axis=0)


def _xyz(points: np.ndarray) -> dict:
    return {"x": points[:, 0].tolist(), "y": points[:, 1].tolist(), "z": points[:, 2].tolist()}


def _drone_axes(position: np.ndarray, rotation: np.ndarray) -> dict:
    """FLU axis tips at the last pose only (body-to-world columns), meters."""
    p = position[-1]
    axes = rotation[-1] * AXIS_M
    return {
        "p": {"x": [float(p[0])], "y": [float(p[1])], "z": [float(p[2])]},
        "f": {"x": [float(axes[0, 0])], "y": [float(axes[1, 0])], "z": [float(axes[2, 0])]},
        "l": {"x": [float(axes[0, 1])], "y": [float(axes[1, 1])], "z": [float(axes[2, 1])]},
        "u": {"x": [float(axes[0, 2])], "y": [float(axes[1, 2])], "z": [float(axes[2, 2])]},
    }


@torch.no_grad()
def _roll(model, flight, start: int, steps: int, offset: np.ndarray):
    device = torch.device("cpu")
    origin = (flight.position[start] + offset).astype(np.float32)
    motors = torch.as_tensor(flight.motors[start : start + steps], dtype=torch.float32, device=device).unsqueeze(0)
    dt = torch.as_tensor(flight.dt[start : start + steps], dtype=torch.float32, device=device).unsqueeze(0)
    pos, rot, vel, omega = _integrate(
        model,
        motors,
        dt,
        torch.as_tensor(origin, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(flight.rotation[start], dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(flight.velocity_world[start], dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(flight.omega_body[start], dtype=torch.float32, device=device).unsqueeze(0),
    )
    position = np.vstack((origin, pos.squeeze(0).cpu().numpy()))
    rotation = np.concatenate((flight.rotation[start : start + 1], rot.squeeze(0).cpu().numpy()), axis=0)
    velocity = np.vstack((flight.velocity_world[start], vel.squeeze(0).cpu().numpy()))
    rates = np.vstack((flight.omega_body[start], omega.squeeze(0).cpu().numpy()))
    return position, rotation, velocity, rates


def _channels(flight, start: int, pred, steps: int, offset: np.ndarray) -> dict:
    steps = min(int(steps), flight.n_steps - start, pred[0].shape[0] - 1)
    truth_p = flight.position[start : start + steps + 1] + offset
    truth_v = flight.velocity_world[start : start + steps + 1]
    truth_w = flight.omega_body[start : start + steps + 1]
    truth_r = _rpy(flight.rotation[start : start + steps + 1])
    pred_p, pred_R, pred_v, pred_w = pred
    pred_r = _rpy(pred_R[: steps + 1])
    t = (np.arange(steps + 1) * DT).tolist()
    names = (
        ("x", truth_p[:, 0], pred_p[: steps + 1, 0]),
        ("y", truth_p[:, 1], pred_p[: steps + 1, 1]),
        ("z", truth_p[:, 2], pred_p[: steps + 1, 2]),
        ("roll", truth_r[:, 0], pred_r[:, 0]),
        ("pitch", truth_r[:, 1], pred_r[:, 1]),
        ("yaw", truth_r[:, 2], pred_r[:, 2]),
        ("vx", truth_v[:, 0], pred_v[: steps + 1, 0]),
        ("vy", truth_v[:, 1], pred_v[: steps + 1, 1]),
        ("vz", truth_v[:, 2], pred_v[: steps + 1, 2]),
        ("wx", truth_w[:, 0], pred_w[: steps + 1, 0]),
        ("wy", truth_w[:, 1], pred_w[: steps + 1, 1]),
        ("wz", truth_w[:, 2], pred_w[: steps + 1, 2]),
    )
    return {
        name: {"t": t, "vicon": np.asarray(truth).tolist(), "pred": np.asarray(hat).tolist()}
        for name, truth, hat in names
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[idsia-interactive] {fmt % args}")

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            data = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/models":
            self._json({"models": list_models()})
            return
        if parsed.path == "/api/flights":
            rows = [
                {"name": name, "duration": float(flight.dt.sum()), "dt": float(np.median(flight.dt))}
                for name, flight in _melon().items()
            ]
            self._json({"flights": rows})
            return
        if parsed.path == "/api/flight":
            name = parse_qs(parsed.query).get("name", [""])[0]
            flights = _melon()
            if name not in flights:
                self._json({"error": f"unknown flight {name}"}, 404)
                return
            flight = flights[name]
            # Body-to-world columns are FLU: forward, left, up.
            axes = flight.rotation * AXIS_M
            self._json(
                {
                    "name": name,
                    "dt": float(np.median(flight.dt)),
                    "duration": float(flight.dt.sum()),
                    "x": flight.position[:, 0].tolist(),
                    "y": flight.position[:, 1].tolist(),
                    "z": flight.position[:, 2].tolist(),
                    "fx": axes[:, 0, 0].tolist(),
                    "fy": axes[:, 1, 0].tolist(),
                    "fz": axes[:, 2, 0].tolist(),
                    "lx": axes[:, 0, 1].tolist(),
                    "ly": axes[:, 1, 1].tolist(),
                    "lz": axes[:, 2, 1].tolist(),
                    "ux": axes[:, 0, 2].tolist(),
                    "uy": axes[:, 1, 2].tolist(),
                    "uz": axes[:, 2, 2].tolist(),
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
        flights = _melon()
        name = body.get("flight", "")
        if name not in flights:
            self._json({"error": f"unknown flight {name}"}, 404)
            return
        flight = flights[name]
        dt = float(np.median(flight.dt))
        start = min(max(int(round(float(body.get("start_s", 0.0)) / dt)), 0), flight.n_steps - 2)
        window_s = max(float(body.get("window_s", 1.0)), DT)
        steps = min(int(round(window_s / dt)), flight.n_steps - start)
        hist = min(int(round(HISTORY_S / dt)), start)
        offset = np.array(
            [float(body.get("dx", 0.0)), float(body.get("dy", 0.0)), float(body.get("dz", 0.0))],
            dtype=np.float32,
        )
        models = [body.get("model_a", "physics"), body.get("model_b", "physics")]
        series = {}
        preds = []
        frames = []
        for model_id in models:
            model = _load(model_id)
            pred = _roll(model, flight, start, steps, offset)
            preds.append(_xyz(pred[0]))
            frames.append(_drone_axes(pred[0], pred[1]))
            ch = _channels(flight, start, pred, steps, offset)
            for key, payload in ch.items():
                row = series.setdefault(key, {"t": payload["t"], "vicon": payload["vicon"]})
                slot = "a" if "a" not in row else "b"
                row[slot] = payload["pred"]
                row[slot + "_name"] = model_id
        truth_p = flight.position[start : start + steps + 1] + offset
        truth_R = flight.rotation[start : start + steps + 1]
        leadin = flight.position[start - hist : start + 1] + offset
        self._json(
            {
                "labels": models,
                "leadin": _xyz(leadin),
                "vicon": _xyz(truth_p),
                "a": preds[0],
                "b": preds[1],
                "drone_axes": {
                    "vicon": _drone_axes(truth_p, truth_R),
                    "a": frames[0],
                    "b": frames[1],
                },
                "channels": series,
                "offset": offset.tolist(),
                "start": start,
                "steps": steps,
            }
        )


def main() -> None:
    if not PAGE.is_file():
        raise SystemExit(f"missing {PAGE}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"idsia interactive  {url}", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
