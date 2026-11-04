# IDSIA Crazyflie sys-id

Busetto et al., *Control Engineering Practice* 172 (2026). Official split: Square / Random / Chirp runs 1–3 train, run 4 val, Melon test. Fetch with `python quadrotor/fetch_idsia.py`.

We train three models with **their recipe** (stride 1, 50-step windows, WeightedMSE λ=0.1, Adam 1e-5→1e-8 cosine T=500, 20 epochs, best val):

| family | class | what it is |
| --- | --- | --- |
| `quad_dynamics` | `QuadDynamicsNet` | delay-line MLP on the 13-D quotient → body (a, α); kinematics `_apply`. Optional Z₂×Z₂ (default on). History length 10 is internal. |
| `phys_quotient` | `PhysQuotient` | frozen Crazyflie RK4 + 12-D residual on the quotient only |
| `phys_quotient_sym` | `PhysQuotientSym` | same physics, body residual with Reynolds orbit |

Their published physics and phys+res checkpoints are loaded as-is (`test_baselines.py`). Do not retrain them.

```bash
python -m quadrotor.idsia.train_quad_dynamics --seed 0
python -m quadrotor.idsia.test_quad_dynamics --all
python -m quadrotor.idsia.test_baselines
```

`--all` trains or evaluates every seed that already has a run dir / checkpoint. Checkpoints: `out/runs/<family>/seed_k/{best.pt,last.pt}`. Melon MAE: `out/eval/<family>/seed_k/{mae.json,predictions.npz}`.

`train.ipynb` trains one seed (resumes `last.pt`). `results.ipynb` prints the Melon table at h=1 / 10 / 50.
