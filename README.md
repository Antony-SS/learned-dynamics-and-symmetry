# Learning dynamics

Incorporating continuous and morphological symmetries for learned robot dynamics.  This repository accompanies the paper available [HERE - Add Link].  

The repository is broken down by platform, with a folder for each.  Currently we have code and data for:

- Quadrotors
- Differentiable Drive Unicycle Robots

We evaluate only on real robot data and each platform comes with an associated dataset for evaluation.

## Setup

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync
source .venv/bin/activate
```

That creates `.venv`, installs the lockfile (`uv.lock`), and installs this repo editable. Dependencies are pinned in `uv.lock` from `pyproject.toml` (numpy, torch, matplotlib, pandas, plotly, pillow, jupyter).

IDSIA Melon CSVs (git + git-lfs):

```bash
# macOS: brew install git-lfs && git lfs install
uv run python -m quadrotor.idsia.data.fetch
```

Unicycle Robotarium logs are not in this repo. Download the [Robotarium dataset](https://drive.google.com/drive/folders/1clnovbZEz0lmSBDjoV2VMztKRxhU-Eog?usp=drive_link) and use **robot 11 `standard_collections`**. The loader expects that folder at

`../robotarium-capstone-data/single_robots/data_robot_11/standard_collections`.

`runs/` is gitignored. Train locally, or copy someone else's `runs/quadrotor` / `runs/unicycle` checkpoints.

## Layout

- `unicycle`: SE(2) geometry and dynamics wrappers. Robotarium experiments live in `unicycle/experiments/`.
- `quadrotor/flight.py` and `quadrotor/geometry.py`: `PreparedFlight` and SO(3) / kinematics.
- `quadrotor/idsia`: Busetto et al. Crazyflie sys-id.

## Run

```bash
uv run python -m quadrotor.idsia.train.sym_quad_dynamics --seed 0
uv run python -m quadrotor.idsia.visualizations.interactive
uv run python -m unicycle.experiments.interactive
```

Open `unicycle/experiments/train/train.ipynb` or `quadrotor/idsia/train/train.ipynb` with the `.venv` kernel.

The IDSIA benchmark is from Busetto et al., *Control Engineering Practice* 172 (2026), 106871.

## License

MIT, see [LICENSE](LICENSE).
