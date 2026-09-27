"""
PRAHARI — smoke tests. Run before every push:  pytest -q

These are deliberately about the things that break silently:
the contracts, the pipeline ORDER, and the feature count.
"""
import numpy as np
import pytest

from hub.contracts import (FS, WINDOW_SAMPLES, CLASSES, N_FEATURES,
                           FEATURE_NAMES, signal_trust_index)
from hub.preprocess import preprocess, zscore, bandpass, window_signal, Calibration
from hub.features import extract
from hub import degrade, restore, hr


def _clean_window(seed=0):
    from ml.build_dataset import synth_ppg
    return preprocess(synth_ppg(30, seed=seed))[:WINDOW_SAMPLES]


def test_contracts_frozen():
    assert N_FEATURES == 13 == len(FEATURE_NAMES)
    assert CLASSES[0] == "Clean"
    assert len(CLASSES) == 5
    assert WINDOW_SAMPLES == 500 and FS == 100


def test_feature_vector_shape_and_finiteness():
    f = extract(_clean_window())
    assert f.shape == (13,)
    assert np.all(np.isfinite(f))


def test_zscore_destroys_attenuation():
    """
    The reason the pipeline order is what it is. If this test ever fails,
    someone has changed zscore and the Attenuated class may be recoverable
    after normalisation — which would be good news, but check it deliberately.
    """
    w = _clean_window().astype(np.float64)
    a, b = zscore(w), zscore(0.15 * w)
    # identical up to float round-off: the gain is gone, not merely reduced
    assert np.max(np.abs(a - b)) < 1e-9


def test_degradation_changes_features_in_correct_order():
    """Degrade AFTER normalising: attenuation must move peak-to-peak."""
    w = _clean_window()
    rng = np.random.default_rng(0)
    att, sev = degrade.apply(w, "Attenuated", 3, rng)
    assert sev == 1.0
    p2p_i = FEATURE_NAMES.index("p2p")
    assert extract(att)[p2p_i] < extract(w)[p2p_i] * 0.5


def test_severity_values():
    w = _clean_window()
    rng = np.random.default_rng(0)
    for lvl, expected in [(1, 0.33), (2, 0.66), (3, 1.0)]:
        _, s = degrade.apply(w, "Noisy", lvl, rng)
        assert s == pytest.approx(expected)


def test_trust_index_bounds_and_direction():
    clean = np.array([0.95, 0.02, 0.01, 0.01, 0.01])
    dirty = np.array([0.05, 0.80, 0.05, 0.05, 0.05])
    assert signal_trust_index(clean, 0.0) > 90
    assert signal_trust_index(dirty, 0.9) < 10
    assert 0 <= signal_trust_index(dirty, 1.0) <= 100


def test_restoration_improves_spiked_snr():
    w = _clean_window()
    rng = np.random.default_rng(1)
    spiked, sev = degrade.apply(w, "Spiked", 3, rng)
    fixed = restore.restore_spiked(spiked, sev)
    assert restore.snr_db(w, fixed) > restore.snr_db(w, spiked)


def test_restoration_dispatch_leaves_clean_alone():
    w = _clean_window()
    out = restore.restore(w, {"class": "Clean", "severity": 0.0})
    assert np.allclose(out, w)


def test_hr_in_physiological_range():
    w = _clean_window()
    bpm, _ = hr.estimate(w)
    assert 40 <= bpm <= 180


def test_calibration_preserves_relative_amplitude():
    """The real-data path must NOT normalise attenuation away."""
    from ml.build_dataset import synth_ppg
    baseline = synth_ppg(30, seed=3)
    cal = Calibration.from_baseline(baseline, subject="test")
    full = cal.apply(baseline)
    half = cal.apply(0.5 * baseline)
    assert np.ptp(half) < np.ptp(full) * 0.6


def test_windowing_overlap():
    x = np.arange(2000, dtype=float)
    W = window_signal(x)
    assert W.shape[1] == WINDOW_SAMPLES
    assert W[1, 0] - W[0, 0] == 250
