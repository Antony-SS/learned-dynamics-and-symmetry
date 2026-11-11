"""Long-exposure stills of Robotarium collection videos.

    python -m unicycle.experiments.long_exposure figure8
    python -m unicycle.experiments.long_exposure figure8 slalom --stride 1
    python -m unicycle.experiments.long_exposure --montage     # 2x2 PDF of the test set

For each collection, the overhead video is streamed through ffmpeg (no OpenCV
needed). A clean arena background is the per-pixel median of sampled frames.
Frames with a projected overlay ("INITIALIZING", the end logo) differ from that
background over a large area and are skipped.

The image is a light painting: the arena is dimmed, the robot's green LED
(green excess over the background) and its dark body are accumulated over every
frame into a streak, tone-mapped so repeated passes saturate instead of
clipping, and given a multi-scale bloom. Per frame, the signal is thresholded
and kept only inside a disc around the robot (its strongest blob), so sensor
noise elsewhere never accumulates. The robot is pasted sharp at its last clean
position.

Decoding is the slow part (minutes per video). The per-pixel sums are cached in
``visualizations/long_exposure/.cache/<collection>.npz``, so changing the look
constants re-renders in seconds. Pass ``--recompute`` after changing anything
in the per-frame stage (thresholds, radius, stride).

Output: ``visualizations/long_exposure/<collection>.png``. ``--montage`` places the
four test collections in a 2x2 grid in ``test_montage.pdf`` (images embedded at
full resolution; ``montage_size()`` gives its size so other panels can match).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unicycle.experiments.data import DEFAULT_ROOT, TEST_LOGS
from unicycle.experiments.utils.params import VIZ

OUT = VIZ / "long_exposure"

# Look. Tune here.
BACKGROUND_DIM = 0.50         # arena brightness under the streak
GREEN_LO, GREEN_HI = 0.06, 0.25   # green excess over background: ignored below, full LED above
BODY_LO, BODY_HI = 0.12, 0.40     # luminance drop: ignored below, full robot body above
ROBOT_RADIUS = 45.0               # px at 1080p kept around the robot each frame
BODY_WEIGHT = 0.35            # body contribution to the streak relative to the LED
EXPOSURE = 2.5                # streak = 1 − exp(−EXPOSURE · summed signal / its 99th pct)
BLOOM = ((4.0, 0.9), (18.0, 0.6), (60.0, 0.35))  # (sigma px at 1080p, gain)
STREAK_RGB = (0.35, 1.0, 0.55)
OVERLAY_FRACTION = 0.02       # skip frames differing from the background over >2% of pixels

# Montage layout (inches). The height follows from the image aspect.
MONTAGE_WIDTH_IN = 6.0
MONTAGE_GAP_IN = 0.06
MONTAGE_COLS = 2              # 2 -> 2x2 grid for the four test collections


def find_video(collection: str, root: Path = DEFAULT_ROOT) -> Path:
    videos = sorted((Path(root) / collection).glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no .mp4 under {Path(root) / collection}")
    return videos[0]


def probe(video: Path) -> tuple[int, int, float, float]:
    info = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(video)],
            check=True, capture_output=True, text=True,
        ).stdout
    )
    stream = info["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return int(stream["width"]), int(stream["height"]), float(num) / float(den), float(info["format"]["duration"])


def frames(video: Path, width: int, height: int, fps: float | None = None, start: float = 0.0):
    """Yield float32 RGB frames in [0, 1], shape (H, W, 3)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video)]
    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    size = width * height * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3).astype(np.float32) / 255.0


def grab(video: Path, width: int, height: int, t: float) -> np.ndarray:
    """One frame at time ``t`` seconds."""
    buf = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(buf, np.uint8).reshape(height, width, 3).astype(np.float32) / 255.0


def background(video: Path, width: int, height: int, duration: float, samples: int = 61) -> np.ndarray:
    """Per-pixel median of frames sampled across the middle 80% of the video."""
    stack = [grab(video, width, height, float(t)) for t in np.linspace(0.1 * duration, 0.9 * duration, samples)]
    return np.median(np.stack(stack), axis=0).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def green_excess(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of an (H, W) or (H, W, C) array."""
    t = torch.from_numpy(np.ascontiguousarray(image))
    t = t.unsqueeze(-1) if t.ndim == 2 else t
    t = t.permute(2, 0, 1).unsqueeze(1)  # (C, 1, H, W)
    radius = int(3 * sigma)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    t = F.conv2d(F.pad(t, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    out = t.squeeze(1).permute(1, 2, 0).numpy()
    return out[..., 0] if image.ndim == 2 else out


def ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def robot_signal(frame: np.ndarray, bg_lum: np.ndarray, bg_green: np.ndarray, radius: float):
    """Thresholded LED and body maps, zeroed outside a disc around the robot."""
    led = ramp(green_excess(frame) - bg_green, GREEN_LO, GREEN_HI)
    body = ramp(bg_lum - luminance(frame), BODY_LO, BODY_HI)
    # Locate the robot as the strongest blob on an 8x-downsampled map.
    h, w = led.shape
    k = 8
    coarse = (led + body)[: h // k * k, : w // k * k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    if coarse.max() <= 0.02:
        return None
    cy, cx = np.unravel_index(int(np.argmax(coarse)), coarse.shape)
    cy, cx = (cy + 0.5) * k, (cx + 0.5) * k
    y0, y1 = int(max(cy - radius, 0)), int(min(cy + radius + 1, h))
    x0, x1 = int(max(cx - radius, 0)), int(min(cx + radius + 1, w))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2).astype(np.float32)
    window = (slice(y0, y1), slice(x0, x1))
    return window, led[window] * disc, body[window] * disc


def accumulate(collection: str, stride: int, root: Path, recompute: bool) -> dict:
    cache = OUT / ".cache" / f"{collection}.npz"
    if cache.is_file() and not recompute:
        data = dict(np.load(cache))
        if int(data["stride"]) == stride:
            print(f"  using cache {cache.name} (pass --recompute to rebuild)", flush=True)
            return data
    video = find_video(collection, root)
    width, height, fps, duration = probe(video)
    print(f"  {video.name}  {width}x{height}  {fps:g} fps  {duration:.1f} s", flush=True)
    bg = background(video, width, height, duration)
    bg_lum, bg_green = luminance(bg), green_excess(bg)
    radius = ROBOT_RADIUS * height / 1080.0

    led_sum = np.zeros((height, width), np.float32)
    body_sum = np.zeros((height, width), np.float32)
    last_frame = np.zeros_like(bg)
    last_mask = np.zeros((height, width), np.float32)
    used = skipped = missing = 0
    for frame in frames(video, width, height, fps=fps / stride):
        if (np.abs(frame - bg).max(axis=-1) > 0.12).mean() > OVERLAY_FRACTION:
            skipped += 1
            continue
        found = robot_signal(frame, bg_lum, bg_green, radius)
        if found is None:
            missing += 1
            continue
        window, led, body = found
        led_sum[window] += led
        body_sum[window] += body
        used += 1
        last_frame = frame
        last_mask = np.zeros_like(last_mask)
        last_mask[window] = np.maximum(led, body)
    if used == 0:
        raise RuntimeError("no usable frames; check OVERLAY_FRACTION and the thresholds")
    print(f"  frames used {used}, overlay skipped {skipped}, robot not found {missing}", flush=True)
    data = {"bg": bg, "led_sum": led_sum, "body_sum": body_sum, "last_frame": last_frame,
            "last_mask": last_mask, "stride": np.array(stride)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    return data


def render(collection: str, stride: int = 2, root: Path = DEFAULT_ROOT, recompute: bool = False) -> Path:
    print(f"{collection}", flush=True)
    data = accumulate(collection, stride, root, recompute)
    bg, last_frame, last_mask = data["bg"], data["last_frame"], data["last_mask"]
    bg_lum = luminance(bg)
    height = bg.shape[0]

    # Streak: LED plus a weaker body trace, tone-mapped so overlaps saturate smoothly.
    signal = data["led_sum"] + BODY_WEIGHT * data["body_sum"]
    ref = float(np.quantile(signal[signal > 0], 0.99)) if (signal > 0).any() else 1.0
    streak = 1.0 - np.exp(-EXPOSURE * signal / max(ref, 1e-6))

    scale = height / 1080.0
    glow = streak.copy()
    for sigma, gain in BLOOM:
        glow += gain * blur(streak, sigma * scale)
    glow_rgb = np.clip(glow, 0.0, None)[..., None] * np.array(STREAK_RGB, np.float32)

    # Dim, slightly desaturated arena, then screen the glow on top.
    base = BACKGROUND_DIM * (0.6 * bg + 0.4 * bg_lum[..., None])
    image = 1.0 - (1.0 - base) * np.exp(-glow_rgb)  # soft additive (screen-like, no hard clip)

    # Sharp robot at its last clean position.
    mask = blur(np.clip(last_mask * 2.0, 0.0, 1.0), 1.5 * scale)[..., None]
    image = image * (1.0 - mask) + last_frame * mask

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{collection}.png"
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)).save(dest)
    print(f"  wrote {dest}", flush=True)
    return dest


def montage_size(collections=TEST_LOGS) -> tuple[float, float, float, float]:
    """(figure width, figure height, panel width, panel height) in inches."""
    first = Image.open(OUT / f"{collections[0]}.png")
    aspect = first.height / first.width
    cols = min(MONTAGE_COLS, len(collections))
    rows = -(-len(collections) // cols)
    panel_w = (MONTAGE_WIDTH_IN - (cols - 1) * MONTAGE_GAP_IN) / cols
    panel_h = panel_w * aspect
    return MONTAGE_WIDTH_IN, rows * panel_h + (rows - 1) * MONTAGE_GAP_IN, panel_w, panel_h


def montage(collections=TEST_LOGS, name: str = "test_montage") -> Path:
    """Stills in a grid of ``MONTAGE_COLS`` columns, row by row in ``collections``
    order, with small gaps and no labels."""
    import matplotlib.pyplot as plt

    missing = [c for c in collections if not (OUT / f"{c}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"render these first: {missing}")
    width, height, panel_w, panel_h = montage_size(collections)
    cols = min(MONTAGE_COLS, len(collections))
    fig = plt.figure(figsize=(width, height))
    for i, collection in enumerate(collections):
        row, col = divmod(i, cols)
        left = col * (panel_w + MONTAGE_GAP_IN)
        top = row * (panel_h + MONTAGE_GAP_IN)
        ax = fig.add_axes((left / width, 1.0 - (top + panel_h) / height, panel_w / width, panel_h / height))
        # interpolation="none" embeds each image at its native resolution in the PDF
        ax.imshow(np.asarray(Image.open(OUT / f"{collection}.png").convert("RGB")), interpolation="none")
        ax.set_axis_off()
    pdf = OUT / f"{name}.pdf"
    fig.savefig(pdf)  # no bbox_inches="tight": keep the exact size montage_size() reports
    fig.savefig(pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}  ({width:.2f} x {height:.2f} in)", flush=True)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collections", nargs="*", default=[],
                        help=f"collection folder names (test set: {', '.join(TEST_LOGS)})")
    parser.add_argument("--stride", type=int, default=2, help="use every k-th video frame")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached per-pixel sums")
    parser.add_argument("--montage", action="store_true",
                        help="after rendering, write test_montage.pdf (2x2 grid of the test collections)")
    args = parser.parse_args()
    collections = args.collections or ([] if args.montage else ["figure8"])
    for name in collections:
        render(name, stride=args.stride, recompute=args.recompute)
    if args.montage:
        montage()


if __name__ == "__main__":
    main()
