"""Heatmap of commanded (v, ω) over the whole dataset (train and test), log color scale.

    python -m unicycle.experiments.command_heatmap              # heatmap only
    python -m unicycle.experiments.command_heatmap --stack figure8
    python -m unicycle.experiments.command_heatmap --match-montage

Uses every sample of every collection in ``TRAIN_LOGS`` and ``TEST_LOGS`` (full
logs, no mirror augmentation). Empty bins are left blank, and the color scale
is logarithmic so rarely visited commands stay visible next to the dense
straight-line band. Text is Times (math in STIX, which matches it).

``--stack <collection>`` also writes a figure with that collection's long
exposure (from ``long_exposure.py``) on top and the heatmap below.
``--match-montage`` also writes ``command_vw_heatmap_wide.pdf`` at exactly the
size of the long-exposure test montage, for side-by-side layout.

Writes vector PDFs (images and bins are rasterized inside them) and PNG
previews to ``visualizations/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.ticker import MultipleLocator
from PIL import Image

from unicycle.experiments.data import TEST_LOGS, TRAIN_LOGS, commanded_vw, load_all
from unicycle.experiments.utils.params import VIZ

BINS_V = 50
BINS_W = 50
# Blues without its near-white end, so single-sample bins stay visible on white.
CMAP = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.18, 1.0, 256)))
WIDTH_IN = 4.6
PDF_DPI = 400  # resolution of rasterized content inside the PDFs
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
}


def dataset_logs():
    found = load_all()
    names = list(TRAIN_LOGS) + list(TEST_LOGS)
    missing = [n for n in names if n not in found]
    if missing:
        raise FileNotFoundError(f"collections not found: {missing}")
    return [found[n] for n in names]


def draw_heatmap(ax) -> int:
    """Draw the log-scale commanded (v, ω) heatmap on ``ax``. Returns the sample count."""
    v, w = commanded_vw(dataset_logs())
    counts, v_edges, w_edges = np.histogram2d(v, w, bins=(BINS_V, BINS_W))
    counts = np.ma.masked_equal(counts, 0)  # empty bins stay blank
    ax.pcolormesh(v_edges, w_edges, counts.T, cmap=CMAP,
                  norm=LogNorm(vmin=1, vmax=counts.max()), rasterized=True)
    ax.set_xlabel(r"commanded $v$ (m/s)")
    ax.set_ylabel(r"commanded $\omega$ (rad/s)")
    ax.set_xlim(0.0, max(0.2, float(v_edges[-1])))
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return len(v)


def _save(fig, dest: Path, name: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    pdf = dest / f"{name}.pdf"
    fig.savefig(pdf, dpi=PDF_DPI)  # dpi only affects rasterized parts (photo, bins)
    fig.savefig(dest / f"{name}.png", dpi=200)
    plt.close(fig)
    return pdf


def plot_command_heatmap_log(dest: Path | None = None, name: str = "command_vw_heatmap") -> Path:
    """Heatmap alone: ``<name>.pdf`` and ``<name>.png``. Returns the PDF path."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
        n = draw_heatmap(ax)
        fig.tight_layout()
        pdf = _save(fig, VIZ if dest is None else Path(dest), name)
    print(f"{n} samples from {len(TRAIN_LOGS) + len(TEST_LOGS)} collections -> {pdf}", flush=True)
    return pdf


def plot_exposure_over_heatmap(collection: str = "figure8", dest: Path | None = None) -> Path:
    """Long exposure of ``collection`` on top, heatmap below, in one figure."""
    still = VIZ / "long_exposure" / f"{collection}.png"
    if not still.is_file():
        raise FileNotFoundError(f"{still} (run python -m unicycle.experiments.long_exposure {collection})")
    image = np.asarray(Image.open(still).convert("RGB"))
    top_h = WIDTH_IN * image.shape[0] / image.shape[1]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(WIDTH_IN, top_h + 3.0), layout="constrained")
        grid = fig.add_gridspec(2, 1, height_ratios=(top_h, 3.0))
        top = fig.add_subplot(grid[0])
        top.imshow(image, interpolation="lanczos")
        top.set_axis_off()
        draw_heatmap(fig.add_subplot(grid[1]))
        pdf = _save(fig, VIZ if dest is None else Path(dest), f"{collection}_exposure_command_vw")
    print(f"-> {pdf}", flush=True)
    return pdf


def plot_command_heatmap_matching_montage(name: str = "command_vw_heatmap_wide") -> Path:
    """Heatmap at the exact size of ``long_exposure.montage_size()``."""
    from unicycle.experiments.long_exposure import montage_size

    width, height, _pw, _ph = montage_size()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
        draw_heatmap(ax)
        pdf = _save(fig, VIZ, name)
    print(f"-> {pdf}  ({width:.2f} x {height:.2f} in)", flush=True)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stack", metavar="COLLECTION", default=None,
                        help="also write that collection's long exposure stacked above the heatmap")
    parser.add_argument("--match-montage", action="store_true",
                        help="also write a heatmap sized to the long-exposure test montage")
    args = parser.parse_args()
    plot_command_heatmap_log()
    if args.stack:
        plot_exposure_over_heatmap(args.stack)
    if args.match_montage:
        plot_command_heatmap_matching_montage()


if __name__ == "__main__":
    main()
