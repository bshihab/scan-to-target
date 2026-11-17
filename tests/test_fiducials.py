import numpy as np
import pytest

from s2t import fiducials as fid
from s2t import transforms as T


def test_exact_recovery_large_motion():
    rng = np.random.default_rng(0)
    P = rng.normal(0, 50, (6, 3))
    M = T.random_rigid(rng, 179.0, 900.0)
    Q = P @ M[:3, :3].T + M[:3, 3]
    est = fid.fit_rigid(P, Q)
    assert np.allclose(est, M, atol=1e-9)
    assert fid.fre(est, P, Q) < 1e-9


def test_never_returns_a_reflection():
    # nearly planar points measured with noise: the unconstrained SVD answer can be a mirror
    rng = np.random.default_rng(1)
    for _ in range(200):
        P = np.c_[rng.normal(0, 40, (4, 2)), rng.normal(0, 0.01, 4)]
        Q = P + rng.normal(0, 2.0, P.shape)
        R = fid.fit_rigid(P, Q)[:3, :3]
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_collinear_refused():
    P = np.c_[np.linspace(0, 30, 4), np.zeros(4), np.zeros(4)]
    with pytest.raises(ValueError, match="collinear"):
        fid.fit_rigid(P, P)


def test_prediction_is_frame_invariant():
    rng = np.random.default_rng(2)
    P = rng.normal(0, 60, (5, 3))
    t = np.array([10.0, -20, 40])
    M = T.random_rigid(rng, 90, 100)
    a = fid.predicted_tre_rms(P, t, 0.5)
    b = fid.predicted_tre_rms(T.apply(M, P), T.apply(M, t), 0.5)
    assert a == pytest.approx(b, rel=1e-9)
