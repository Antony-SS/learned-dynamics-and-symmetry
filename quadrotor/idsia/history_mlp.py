"""IDSIA delay-line MLPs on the 13-D quotient (g, v, ω, Ω).

The official leftover is hist-10 with Z2×Z2. Checkpoints stay out of
``runs/idsia_parity18k``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from quadrotor.evaluate import benchmark_mae
from quadrotor.fit import n_params
from quadrotor.idsia.baselines import load_theirs
from quadrotor.idsia.data import load_idsia_prepared
from quadrotor.sequence import _mirror_acceleration, _mirror_features
from quadrotor.train import _apply, _features, _so3_log

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADLINE_DIR = REPO_ROOT / "runs" / "idsia_mlp_hist10"
HIST5_DIR = REPO_ROOT / "runs" / "idsia_mlp_hist5w"
HORIZON = 50
EPOCHS = 20
HIDDEN = 64
BATCH = 256
STRIDE = 1
TRAIN_STRIDE = 5

# Identity, left-right, front-back, both. PWM permutation is separate.
_ORBIT_GV = (
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
)
_ORBIT_W = (
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, -1.0, 1.0),
)
_ORBIT_PWM = (
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
)
_ORBIT_ACC = (
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
    (-1.0, 1.0, 1.0, 1.0, -1.0, -1.0),
    (-1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
)


class HistoryQuotientMLP(torch.nn.Module):
    """64-64 on stacked 13-D quotient frames. Optional Z2×Z2 orbit."""

    reads_pose = False
    reads_voltage = False
    reads_height = False

    def __init__(
        self,
        history: int,
        hidden: int = HIDDEN,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        self.history = int(history)
        self.symmetric = bool(symmetric)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(13 * self.history, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 6),
        )
        self.register_buffer("x_mean", torch.zeros(13))
        self.register_buffer("x_scale", torch.ones(13))
        self.register_buffer("orbit_gv", torch.tensor(_ORBIT_GV, dtype=torch.float32))
        self.register_buffer("orbit_w", torch.tensor(_ORBIT_W, dtype=torch.float32))
        self.register_buffer("orbit_pwm", torch.tensor(_ORBIT_PWM, dtype=torch.long))
        self.register_buffer("orbit_acc", torch.tensor(_ORBIT_ACC, dtype=torch.float32))

    def set_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        with torch.no_grad():
            self.x_mean.copy_(mean.reshape(13).to(self.x_mean))
            self.x_scale.copy_(scale.reshape(13).to(self.x_scale).clamp_min(1e-6))

    def _mirror_stack(self, stacked: torch.Tensor, name: str) -> torch.Tensor:
        slots = stacked.reshape(stacked.shape[0], self.history, 13)
        return _mirror_features(slots, name).reshape(stacked.shape[0], -1)

    def _predict(self, stacked: torch.Tensor) -> torch.Tensor:
        if not self.symmetric:
            return self.net(stacked)
        batch = stacked.shape[0]
        slots = stacked.reshape(batch, self.history, 13)
        pwm = slots[..., 9:13]
        index = self.orbit_pwm.view(4, 1, 1, 4).expand(4, batch, self.history, 4)
        orbit = torch.cat(
            (
                slots[..., 0:6].unsqueeze(0) * self.orbit_gv.view(4, 1, 1, 6),
                slots[..., 6:9].unsqueeze(0) * self.orbit_w.view(4, 1, 1, 3),
                pwm.unsqueeze(0).expand(4, -1, -1, -1).gather(-1, index),
            ),
            dim=-1,
        )
        raw = self.net(orbit.reshape(4 * batch, -1)).reshape(4, batch, 6)
        return (raw * self.orbit_acc.unsqueeze(1)).sum(0) / 4.0

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        return self._predict(stacked)

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, self.history, 13)

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        buf = torch.cat((hidden[:, 1:], features.unsqueeze(1)), dim=1)
        flat = ((buf - self.x_mean) / self.x_scale).reshape(buf.shape[0], -1)
        return self._predict(flat), buf


def feat_row(flight, index: int) -> np.ndarray:
    index = int(np.clip(index, 0, flight.n_steps - 1))
    pwm_i = min(index, len(flight.pwm) - 1)
    g = flight.rotation[index].T @ np.array((0.0, 0.0, 1.0), np.float32)
    v = flight.rotation[index].T @ flight.velocity_world[index]
    return np.concatenate((g, v, flight.omega_body[index], flight.pwm[pwm_i])).astype(
        np.float32
    )


def feat_t(rotation, vel_world, omega, pwm) -> torch.Tensor:
    dummy = type(
        "M",
        (),
        {"reads_pose": False, "reads_voltage": False, "reads_height": False},
    )()
    return _features(dummy, None, rotation, vel_world, omega, pwm, None)


def starts(flights, horizon: int = HORIZON, stride: int = STRIDE):
    return [
        (i, s)
        for i, f in enumerate(flights)
        for s in range(0, f.n_steps - horizon, stride)
    ]


def state_scales(flights):
    p = np.concatenate([f.position for f in flights])
    v = np.concatenate([f.velocity_world for f in flights])
    w = np.concatenate([f.omega_body for f in flights])
    r = _so3_log(torch.as_tensor(np.concatenate([f.rotation for f in flights]))).numpy()
    return tuple(max(float(np.std(x)), 1e-6) for x in (p, r, v, w))


def feat_stats(flights):
    rows = np.concatenate([flight_features(f) for f in flights], axis=0)
    return (
        torch.as_tensor(rows.mean(0), dtype=torch.float32),
        torch.as_tensor(np.clip(rows.std(0), 1e-6, None), dtype=torch.float32),
    )


def flight_features(flight) -> np.ndarray:
    n = flight.n_steps
    rotation = flight.rotation[:n]
    up = np.array((0.0, 0.0, 1.0), np.float32)
    gravity = np.einsum("nji,j->ni", rotation, up)
    velocity = np.einsum("nji,nj->ni", rotation, flight.velocity_world[:n])
    return np.concatenate(
        (gravity, velocity, flight.omega_body[:n], flight.pwm), axis=1
    ).astype(np.float32)


def cache_flights(flights, device: torch.device) -> list[dict[str, torch.Tensor]]:
    cached = []
    for flight in flights:
        feat = flight_features(flight)
        cached.append(
            {
                "feat": torch.as_tensor(feat, device=device),
                "position": torch.as_tensor(flight.position, device=device),
                "rotation": torch.as_tensor(flight.rotation, device=device),
                "velocity": torch.as_tensor(flight.velocity_world, device=device),
                "omega": torch.as_tensor(flight.omega_body, device=device),
                "pwm": torch.as_tensor(flight.pwm, device=device),
                "dt": torch.as_tensor(flight.dt, device=device),
            }
        )
    return cached


def pack_windows(
    cached: list[dict[str, torch.Tensor]],
    pairs: list[tuple[int, int]],
    history: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    p0, r0, v0, w0, pwm, dt = [], [], [], [], [], []
    pt, rt, vt, wt, hist = [], [], [], [], []
    for fi, s in pairs:
        f = cached[fi]
        n_feat = f["feat"].shape[0]
        idx = torch.arange(s - history + 1, s + 1, device=f["feat"].device)
        hist.append(f["feat"][idx.clamp(0, n_feat - 1)])
        stop = s + horizon
        p0.append(f["position"][s])
        r0.append(f["rotation"][s])
        v0.append(f["velocity"][s])
        w0.append(f["omega"][s])
        pwm.append(f["pwm"][s:stop])
        dt.append(f["dt"][s:stop])
        pt.append(f["position"][s + 1 : stop + 1])
        rt.append(f["rotation"][s + 1 : stop + 1])
        vt.append(f["velocity"][s + 1 : stop + 1])
        wt.append(f["omega"][s + 1 : stop + 1])
    return {
        "p": torch.stack(p0),
        "r": torch.stack(r0),
        "v": torch.stack(v0),
        "w": torch.stack(w0),
        "pwm": torch.stack(pwm),
        "dt": torch.stack(dt),
        "pt": torch.stack(pt),
        "rt": torch.stack(rt),
        "vt": torch.stack(vt),
        "wt": torch.stack(wt),
        "hist": torch.stack(hist),
    }


def warm(flights, batch, history: int, device):
    rows = []
    for fi, s in batch:
        rows.append(np.stack([feat_row(flights[fi], s - history + 1 + j) for j in range(history)]))
    return torch.as_tensor(np.stack(rows), device=device, dtype=torch.float32)


def roll(model, p, r, v_w, w, pwm, dt, hist, horizon: int = HORIZON):
    ps, rs, vs, ws = [], [], [], []
    buf = hist.clone()
    for k in range(horizon):
        flat = ((buf - model.x_mean) / model.x_scale).reshape(buf.shape[0], -1)
        acc = model(flat)
        p, r, v_w, w = _apply(p, r, v_w, w, acc, dt[:, k])
        ps.append(p)
        rs.append(r)
        vs.append(v_w)
        ws.append(w)
        if k + 1 < horizon:
            nxt = feat_t(r, v_w, w, pwm[:, k + 1]).unsqueeze(1)
            buf = torch.cat((buf[:, 1:], nxt), dim=1)
    return torch.stack(ps, 1), torch.stack(rs, 1), torch.stack(vs, 1), torch.stack(ws, 1)


def _batch_states(flights, batch, horizon: int, device):
    p0, r0, v0, w0, pwm, dt = [], [], [], [], [], []
    pt, rt, vt, wt = [], [], [], []
    for fi, s in batch:
        f = flights[fi]
        stop = s + horizon
        p0.append(f.position[s])
        r0.append(f.rotation[s])
        v0.append(f.velocity_world[s])
        w0.append(f.omega_body[s])
        pwm.append(f.pwm[s:stop])
        dt.append(f.dt[s:stop])
        pt.append(f.position[s + 1 : stop + 1])
        rt.append(f.rotation[s + 1 : stop + 1])
        vt.append(f.velocity_world[s + 1 : stop + 1])
        wt.append(f.omega_body[s + 1 : stop + 1])
    return (
        torch.as_tensor(np.stack(p0), device=device),
        torch.as_tensor(np.stack(r0), device=device),
        torch.as_tensor(np.stack(v0), device=device),
        torch.as_tensor(np.stack(w0), device=device),
        torch.as_tensor(np.stack(pwm), device=device),
        torch.as_tensor(np.stack(dt), device=device),
        torch.as_tensor(np.stack(pt), device=device),
        torch.as_tensor(np.stack(rt), device=device),
        torch.as_tensor(np.stack(vt), device=device),
        torch.as_tensor(np.stack(wt), device=device),
    )


def _packed_mean_loss(model, packed, scales, index: torch.Tensor | None = None):
    sl = slice(None) if index is None else index
    pred_p, pred_r, pred_v, pred_w = roll(
        model,
        packed["p"][sl],
        packed["r"][sl],
        packed["v"][sl],
        packed["w"][sl],
        packed["pwm"][sl],
        packed["dt"][sl],
        packed["hist"][sl],
    )
    rel = torch.matmul(packed["rt"][sl].transpose(-1, -2), pred_r)
    sp, sr, sv, sw = scales
    return torch.stack(
        (
            (pred_p - packed["pt"][sl]).div(sp).pow(2).mean(),
            _so3_log(rel.reshape(-1, 3, 3)).div(sr).pow(2).mean(),
            (pred_v - packed["vt"][sl]).div(sv).pow(2).mean(),
            (pred_w - packed["wt"][sl]).div(sw).pow(2).mean(),
        )
    )


@torch.no_grad()
def _packed_split_loss(model, packed, scales) -> np.ndarray:
    model.eval()
    parts = np.zeros(4)
    seen = 0
    n = packed["p"].shape[0]
    for start in range(0, n, BATCH):
        block = _packed_mean_loss(model, packed, scales, slice(start, start + BATCH))
        count = min(BATCH, n - start)
        parts += block.detach().cpu().numpy() * count
        seen += count
    return parts / max(seen, 1)


def train_history(
    train,
    *,
    history: int,
    symmetric: bool,
    device: torch.device,
    seed: int = 0,
    tag: str = "",
    stride: int = TRAIN_STRIDE,
    val=None,
) -> HistoryQuotientMLP:
    torch.manual_seed(seed)
    mean, scale = feat_stats(train)
    model = HistoryQuotientMLP(history=history, symmetric=symmetric).to(device)
    model.set_stats(mean.to(device), scale.to(device))
    pairs = starts(train, stride=stride)
    kind = "Z2×Z2" if symmetric else "quotient-only"
    print(
        f"{tag or f'hist-{history}'}  {kind}  params {n_params(model)}  "
        f"windows {len(pairs)}  train stride {stride}  stacked ω  seed {seed}",
        flush=True,
    )
    packed = pack_windows(cache_flights(train, device), pairs, history, HORIZON)
    packed_val = None
    if val is not None:
        val_pairs = starts(val, stride=stride)
        packed_val = pack_windows(cache_flights(val, device), val_pairs, history, HORIZON)
        print(f"  cached {len(pairs)} train / {len(val_pairs)} val windows on {device}", flush=True)
    else:
        print(f"  cached {len(pairs)} windows on {device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scales = state_scales(train)
    n = len(pairs)
    curve = []
    for epoch in range(1, EPOCHS + 1):
        print(f"  {tag or f'h{history}'}  epoch {epoch:02d}  starting  {n} windows", flush=True)
        model.train()
        order = torch.randperm(n)
        parts = np.zeros(4)
        seen = 0
        for start in range(0, n, BATCH):
            index = order[start : start + BATCH]
            block = _packed_mean_loss(model, packed, scales, index)
            opt.zero_grad()
            block.mean().backward()
            opt.step()
            parts += block.detach().cpu().numpy() * int(index.numel())
            seen += int(index.numel())
        m = parts / max(seen, 1)
        row = {
            "epoch": epoch,
            "train": float(m.mean()),
            "train_p": float(m[0]),
            "train_R": float(m[1]),
            "train_v": float(m[2]),
            "train_w": float(m[3]),
        }
        if packed_val is not None:
            vm = _packed_split_loss(model, packed_val, scales)
            row.update(
                {
                    "val": float(vm.mean()),
                    "val_p": float(vm[0]),
                    "val_R": float(vm[1]),
                    "val_v": float(vm[2]),
                    "val_w": float(vm[3]),
                }
            )
        curve.append(row)
        extra = f"  val {row['val']:.5f}" if "val" in row else ""
        print(
            f"  {tag or f'h{history}'}  epoch {epoch:02d}  train {m.mean():.5f}  "
            f"p {m[0]:.5f}  R {m[1]:.5f}  v {m[2]:.5f}  w {m[3]:.5f}{extra}",
            flush=True,
        )
    model.train_curve = curve
    model.eval()
    return model


@torch.no_grad()
def score_history(model, flights, device) -> dict[str, np.ndarray]:
    pairs = [
        (fi, s)
        for fi, flight in enumerate(flights)
        for s in range(0, flight.n_steps - HORIZON + 1)
    ]
    packed = pack_windows(cache_flights(flights, device), pairs, model.history, HORIZON)
    totals = {k: np.zeros(HORIZON) for k in ("position", "velocity", "rotation", "omega")}
    print(f"  scoring Melon  {len(pairs)} windows  batch {BATCH}", flush=True)
    for offset in range(0, len(pairs), BATCH):
        index = slice(offset, offset + BATCH)
        pred_p, pred_r, pred_v, pred_w = roll(
            model,
            packed["p"][index],
            packed["r"][index],
            packed["v"][index],
            packed["w"][index],
            packed["pwm"][index],
            packed["dt"][index],
            packed["hist"][index],
        )
        pt, rt, vt, wt = (
            packed["pt"][index],
            packed["rt"][index],
            packed["vt"][index],
            packed["wt"][index],
        )
        rel = torch.matmul(rt.transpose(-1, -2), pred_r)
        totals["position"] += (
            torch.linalg.vector_norm(pred_p - pt, dim=-1).sum(0).cpu().numpy()
        )
        totals["rotation"] += (
            torch.linalg.vector_norm(
                _so3_log(rel.reshape(-1, 3, 3)).reshape(rel.shape[0], HORIZON, 3), dim=-1
            )
            .sum(0)
            .cpu()
            .numpy()
        )
        totals["velocity"] += (
            torch.linalg.vector_norm(pred_v - vt, dim=-1).sum(0).cpu().numpy()
        )
        totals["omega"] += (
            torch.linalg.vector_norm(pred_w - wt, dim=-1).sum(0).cpu().numpy()
        )
        print(f"  melon {min(offset + BATCH, len(pairs))}/{len(pairs)}", flush=True)
    return {k: v / max(len(pairs), 1) for k, v in totals.items()}


def row(name: str, mae: dict[str, np.ndarray]) -> None:
    print(
        f"{name:28s}  p {mae['position'][9]:.4f}/{mae['position'][49]:.4f}  "
        f"R {mae['rotation'][9]:.4f}/{mae['rotation'][49]:.4f}  "
        f"v {mae['velocity'][9]:.4f}/{mae['velocity'][49]:.4f}  "
        f"w {mae['omega'][9]:.4f}/{mae['omega'][49]:.4f}"
    )


def save_model(model: HistoryQuotientMLP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "x_mean": model.x_mean.detach().cpu(),
            "x_scale": model.x_scale.detach().cpu(),
            "history": model.history,
            "symmetric": model.symmetric,
            "params": n_params(model),
            "curve": list(getattr(model, "train_curve", [])),
        },
        path,
    )


def load_model(path: Path, device: torch.device | str = "cpu") -> HistoryQuotientMLP:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HistoryQuotientMLP(
        history=int(payload["history"]),
        symmetric=bool(payload.get("symmetric", False)),
    )
    model.load_state_dict(payload["state"], strict=False)
    model.set_stats(payload["x_mean"], payload["x_scale"])
    return model.to(device).eval()


def load_or_train_headline(device: str | torch.device = "cpu") -> HistoryQuotientMLP:
    """Official IDSIA leftover: hist-10 Z2×Z2. Does not touch idsia_parity18k."""
    ckpt = HEADLINE_DIR / "hist10_parity.pt"
    device = torch.device(device)
    if not ckpt.is_file():
        run(history=10, symmetric=True, out=HEADLINE_DIR, seed=0, tag="hist-10 Z2×Z2")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = HistoryQuotientMLP(history=int(payload.get("history", 10)), symmetric=True)
    model.load_state_dict(payload["state"], strict=False)
    model.set_stats(payload["x_mean"], payload["x_scale"])
    return model.to(device).eval()


def run(
    *,
    history: int,
    symmetric: bool,
    out: Path,
    seed: int,
    tag: str,
    device: str = "cpu",
) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train_device = torch.device(device)
    if train_device.type == "cpu" and torch.backends.mps.is_available():
        train_device = torch.device("mps")
    train, _val, test = load_idsia_prepared()
    ckpt = out / f"hist{history}{'_parity' if symmetric else '_quotient'}.pt"
    print(
        f"IDSIA hist-{history}  {'Z2×Z2' if symmetric else 'quotient, no orbit'}  "
        f"train {len(train)}  melon {len(test)}  -> {ckpt}",
        flush=True,
    )
    if ckpt.is_file():
        model = load_model(ckpt, train_device)
        print(f"loaded {ckpt}  params {n_params(model)}", flush=True)
    else:
        model = train_history(
            train,
            history=history,
            symmetric=symmetric,
            device=train_device,
            seed=seed,
            tag=tag,
        )
        save_model(model, ckpt)
        print(f"saved {ckpt}", flush=True)

    print("\nMelon MAE  h10/h50  (physics / phys+res only)", flush=True)
    row(f"{tag} ({n_params(model)})", score_history(model, test, train_device))
    theirs = load_theirs(train_flights=train, device=train_device)
    for name in ("physics", "phys+res"):
        if name in theirs:
            row(name, benchmark_mae(theirs[name], test, horizon=HORIZON, start_stride=1))
