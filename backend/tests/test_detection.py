import numpy as np

from backend.detection import adtof


def test_weights_load_strictly():
    assert adtof.load_model() is not None  # asserts no missing/unexpected keys


def test_first_frame_onsets_survive_picking():
    """Regression: upstream picker's edge-padded average cancelled onsets at frame 0 (PRD 8.3)."""
    pred = np.zeros((50, 5), np.float32)
    pred[0, 0] = pred[0, 3] = 0.9  # kick + hi-hat at sample 0
    pred[1, 1] = 0.9  # snare at frame 1
    peaks = adtof.pick(pred)
    assert peaks[35] == [0.0] and peaks[42] == [0.0] and peaks[38] == [0.01]


def test_last_frame_onset_survives_picking():
    pred = np.zeros((50, 5), np.float32)
    pred[49, 1] = 0.9
    assert adtof.pick(pred)[38] == [0.49]
