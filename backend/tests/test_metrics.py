from backend.validation.metrics import score


def test_perfect_and_shifted():
    ref = [0.0, 0.5, 1.0]
    assert score(ref, ref)["by_tolerance_ms"][10]["f1"] == 1.0
    s = score([t + 0.03 for t in ref], ref)  # 30 ms late: fails 10/25, passes 50/100
    assert s["by_tolerance_ms"][25]["f1"] == 0.0 and s["by_tolerance_ms"][50]["f1"] == 1.0
    assert s["median_error_ms"] == 30.0


def test_extra_and_missing_events():
    s = score([0.0, 0.5, 2.0], [0.0, 0.5, 1.0])["by_tolerance_ms"][50]
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 1)


def test_one_to_one_matching():
    s = score([0.50, 0.51], [0.50])["by_tolerance_ms"][50]  # two detections cannot both match
    assert (s["tp"], s["fp"]) == (1, 1)
