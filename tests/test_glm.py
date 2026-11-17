import numpy as np

from s2t import glm


def test_hrf_shape():
    h = glm.spm_hrf(0.1)
    assert abs(h.sum() - 1) < 1e-12
    assert 4.5 < np.argmax(h) * 0.1 < 5.5  # SPM canonical peaks near 5 s
    assert h.min() < 0  # undershoot


def test_cosine_drift_orthonormal():
    D = glm.cosine_drift(184, 2.5, 128)
    assert D.shape[1] == int(np.floor(2 * 184 * 2.5 / 128))
    assert np.allclose(D.T @ D, np.eye(D.shape[1]), atol=1e-10)


def test_recovers_known_effect():
    rng = np.random.default_rng(0)
    ev = [{"onset": o, "duration": 15.0, "trial_type": "A", "weight": 1} for o in range(10, 400, 60)]
    d = glm.design_matrix(180, 2.5, ev)
    beta = np.zeros(d.X.shape[1]); beta[0] = 3.0; beta[-1] = 100.0
    from s2t.volume import Volume
    Y = (d.X @ beta)[None, None, None, :] + rng.normal(0, 1, (4, 4, 4, 180))
    t, _ = glm.fit_t(Volume(Y.astype(np.float32), np.eye(4)), d, glm.contrast_vector(d, {"A": 1}), np.ones((4, 4, 4), bool))
    assert t.data.min() > 5
