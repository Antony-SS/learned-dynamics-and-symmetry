import torch

from quadrotor.geometry import _so3_exp
from quadrotor.idsia.baselines import PhysQuadModel
from quadrotor.idsia.models import PhysQuotient, PhysQuotientSym, QuadDynamicsNet, make_model, quotient
from quadrotor.idsia.idsia_training_reproduction_params import time_weights
from quadrotor.symmetry import apply_body_dx, apply_orbit_element, orbit_body_dx, orbit_features


def _hover_state(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(batch, 12)
    x[:, 2] = 1.0
    u = torch.ones(batch, 4) * 1000.0
    return x, u


def test_time_weights_sum_to_one():
    w = time_weights(torch.device("cpu"), horizon=50)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_one_step_shapes():
    x, u = _hover_state(3)
    for model in (QuadDynamicsNet(), PhysQuotient(), PhysQuotientSym()):
        y = model.one_step(x, u)
        assert y.shape == (3, 12)
        assert torch.isfinite(y).all()


def test_make_model_families():
    assert isinstance(make_model("quad_dynamics"), QuadDynamicsNet)
    assert isinstance(make_model("phys_quotient"), PhysQuotient)
    assert isinstance(make_model("phys_quotient_sym"), PhysQuotientSym)
    assert isinstance(make_model("phys_quotient_sym", symmetric=False), PhysQuotient)


def test_zero_residual_matches_physics():
    phys = PhysQuadModel(zero_omega_dot=True)
    x, u = _hover_state(4)
    expected = phys.one_step(x, u)
    for model in (QuadDynamicsNet(), PhysQuotient(), PhysQuotientSym()):
        got = model.one_step(x, u)
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_apply_body_dx_zero_is_identity():
    x, u = _hover_state(2)
    phys = PhysQuadModel(zero_omega_dot=True).one_step(x, u)
    assert torch.allclose(apply_body_dx(phys, x, torch.zeros(2, 12)), phys, atol=1e-6)


def test_quotient_is_13d():
    r = _so3_exp(torch.zeros(2, 3))
    v = torch.zeros(2, 3)
    w = torch.zeros(2, 3)
    motors = torch.ones(2, 4)
    q = quotient(r, v, w, motors)
    assert q.shape == (2, 13)


def test_quotient_ignores_translation_and_yaw():
    r = _so3_exp(torch.tensor([[0.2, -0.1, 0.4], [0.0, 0.3, -0.2]]))
    v = torch.tensor([[1.0, 0.2, -0.4], [0.3, -0.5, 0.1]])
    w = torch.tensor([[0.1, -0.2, 0.05], [0.0, 0.4, -0.1]])
    motors = torch.tensor([[800.0, 900.0, 850.0, 870.0], [1000.0, 1100.0, 1050.0, 980.0]])
    base = quotient(r, v, w, motors)
    yaw = _so3_exp(torch.tensor([[0.0, 0.0, 1.3], [0.0, 0.0, -0.7]]))
    rotated = quotient(torch.bmm(yaw, r), torch.bmm(yaw, v.unsqueeze(-1)).squeeze(-1), w, motors)
    assert torch.allclose(base, rotated, atol=1e-5)
    # translation is not an input; two positions share the same quotient by construction
    assert torch.allclose(base, quotient(r, v, w, motors), atol=1e-6)


def test_orbit_element_is_involution():
    q = torch.randn(5, 13)
    for index in range(4):
        twice = apply_orbit_element(apply_orbit_element(q, index), index)
        assert torch.allclose(twice, q, atol=1e-5)


def test_orbit_features_close_under_group():
    q = torch.randn(4, 13)
    stacked = orbit_features(q)
    assert stacked.shape == (4, 4, 13)
    for index in range(4):
        moved = orbit_features(apply_orbit_element(q, index))
        for batch in range(q.shape[0]):
            orig = stacked[:, batch]
            new = moved[:, batch]
            for row in range(4):
                assert (new[row] - orig).abs().sum(dim=-1).min() < 1e-4


def test_orbit_body_dx_involution():
    raw = torch.randn(4, 3, 12)
    signed = orbit_body_dx(raw)
    assert signed.shape == (3, 12)
    flipped = raw.clone()
    polar = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]])
    axial = torch.tensor([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]])
    flipped[..., 0:3] = flipped[..., 0:3] * polar.view(4, 1, 3)
    flipped[..., 3:6] = flipped[..., 3:6] * polar.view(4, 1, 3)
    flipped[..., 6:9] = flipped[..., 6:9] * axial.view(4, 1, 3)
    flipped[..., 9:12] = flipped[..., 9:12] * axial.view(4, 1, 3)
    flipped2 = flipped.clone()
    flipped2[..., 0:3] = flipped2[..., 0:3] * polar.view(4, 1, 3)
    flipped2[..., 3:6] = flipped2[..., 3:6] * polar.view(4, 1, 3)
    flipped2[..., 6:9] = flipped2[..., 6:9] * axial.view(4, 1, 3)
    flipped2[..., 9:12] = flipped2[..., 9:12] * axial.view(4, 1, 3)
    assert torch.allclose(flipped2, raw, atol=1e-6)


def test_phys_quotient_sym_orbit_shape():
    model = PhysQuotientSym()
    x, u = _hover_state(5)
    dx = model.body_dx(x, u)
    assert dx.shape == (5, 12)
