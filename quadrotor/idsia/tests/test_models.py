import torch

from quadrotor.idsia.models import PhysQuotient, PhysQuotientSym, QuadDynamicsNet, quotient
from quadrotor.idsia.recipe import time_weights
from quadrotor.train import _so3_exp


def test_time_weights_sum_to_one():
    w = time_weights(torch.device("cpu"), horizon=50)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_quad_dynamics_step_shapes():
    model = QuadDynamicsNet(symmetric=True)
    b = 4
    feat = torch.randn(b, 13)
    hidden = model.initial_hidden(b, feat)
    acc, hidden2 = model.step(feat, hidden)
    assert acc.shape == (b, 6)
    assert hidden2.shape == (b, model.history, 13)


def test_quad_dynamics_orbit_matches_batch():
    model = QuadDynamicsNet(symmetric=True)
    stacked = torch.randn(3, 13 * model.history)
    out = model(stacked)
    assert out.shape == (3, 6)
    model_off = QuadDynamicsNet(symmetric=False)
    assert model_off(stacked).shape == (3, 6)


def test_phys_quotient_one_step():
    model = PhysQuotient()
    x = torch.zeros(2, 12)
    x[:, 2] = 1.0
    u = torch.ones(2, 4) * 1000.0
    y = model.step_real(x, u)
    assert y.shape == (2, 12)
    assert torch.isfinite(y).all()


def test_phys_quotient_sym_one_step():
    model = PhysQuotientSym()
    x = torch.zeros(2, 12)
    u = torch.ones(2, 4) * 1000.0
    y = model.one_step(x, u)
    assert y.shape == (2, 12)
    assert torch.isfinite(y).all()


def test_phys_quotient_sym_orbit_shape():
    model = PhysQuotientSym()
    canonical = torch.randn(5, 13)
    dx = model.neural.body_dx(canonical)
    assert dx.shape == (5, 12)


def test_quotient_is_13d():
    r = _so3_exp(torch.zeros(2, 3))
    v = torch.zeros(2, 3)
    w = torch.zeros(2, 3)
    pwm = torch.ones(2, 4)
    q = quotient(r, v, w, pwm)
    assert q.shape == (2, 13)
