# NeuroBEM (experimental)

Self-contained. Shared Lie helpers stay in `quadrotor/{geometry,canonical}.py`. GRU / ParityMotorNet live here (`sequence.py`, `openloop.py`, `fit.py`). This folder must not import `quadrotor.idsia`.

Checkpoints and plots write under `quadrotor/neurobem/out/`, not the repo-root `runs/` graveyard.

```bash
python -m quadrotor.neurobem.train
python -m quadrotor.neurobem.gru_vs_coast
python -m quadrotor.neurobem.mlp_fullstate
```
