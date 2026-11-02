import numpy as np
import pytest

from backend.detection import articulation as art
from backend.events import Event
from backend.validation.synth import SR, articulation_clip


def _events(truth):
    return [Event(f"t{i}", t, round(t * SR), inst, 1.0, {}) for i, (t, inst, _) in enumerate(truth)]


def _annotate():
    y, truth = articulation_clip()
    ev = _events(truth)
    # the detector would also emit the flam's grace hit as its own snare event 35 ms earlier
    grace = Event("g", 4.965, round(4.965 * SR), "snare", 0.4, {})
    out, summary = art.annotate(ev + [grace], y)
    return truth, out, summary


def test_synthetic_articulations_recovered():
    truth, out, _ = _annotate()
    def at(t, inst):  # refinement moves times by a few ms: match the nearest event of that instrument
        return min((e for e in out if e.instrument == inst), key=lambda e: abs(e.time_s - t))
    for t, inst, want in truth:
        assert abs(at(t, inst).time_s - t) < 0.02
        if want == "ghost":  # measured but not marked: ghost marks were mostly wrong on real songs
            assert at(t, inst).ghost_probability >= 0.5 and at(t, inst).articulation is None
        elif want:
            assert at(t, inst).articulation == want, (t, inst, want)
    plain_snares = [e for e in out if e.instrument == "snare" and e.articulation is None]
    assert len(plain_snares) == 9  # 8 steady main hits + the quiet one (no false flam)
    assert all(e.ghost_probability < 0.5 for e in plain_snares if abs(e.time_s - 1.25) > 0.05)
    assert sum(e.articulation == "open" for e in out) == 2
    assert not any(e.articulation == "open" for e in out if abs(e.time_s - 7.0) < 0.05 or abs(e.time_s - 8.0) < 0.05)


def test_flam_grace_is_linked_and_not_double_counted():
    _, out, _ = _annotate()
    grace = [e for e in out if e.articulation == "grace"]
    main = [e for e in out if e.articulation == "flam"]
    assert len(grace) == 1 and len(main) == 1 and grace[0].grace_of == main[0].id
    assert grace[0].time_s < main[0].time_s  # raw times untouched


def test_features_are_stored_for_recalibration():
    _, out, _ = _annotate()
    assert all(e.articulation_features for e in out if e.instrument == "cymbal")
    assert all("tail_ratio" in e.articulation_features for e in out if e.instrument == "hihat" and e.time_s > 7)


def test_band_power_frame_alignment():
    y = np.zeros(SR, np.float32)
    y[22050:22050 + 200] = 1.0  # click at 0.5 s
    p = art.band_power(y)["snare"]
    env = art.Env({"snare": p})
    assert env.win("snare", 0.49, 0.52).max() > 0 and env.win("snare", 0.1, 0.3).max() == 0


def _flam_inputs(grace_activation):
    from backend.detection import adtof
    y, truth = articulation_clip()
    ev = _events(truth)  # the grace hit is NOT detected at the default threshold
    pred = np.zeros((2600, 5), np.float32)
    pred[500, 1] = 0.8  # main snare at 5.0 s
    if grace_activation:
        pred[496:498, 1] = [grace_activation, grace_activation * 0.7]  # separate low peak ~35 ms earlier
    return y, ev, pred, adtof.pick


def test_flam_created_from_low_activation_peak_plus_envelope():
    y, ev, pred, pick = _flam_inputs(0.15)
    out, _ = art.annotate(ev, y, pred, pick)
    graces = [e for e in out if e.articulation == "grace"]
    main = next(e for e in out if e.articulation == "flam")
    assert len(graces) == 1 and graces[0].grace_of == main.id and 0.025 <= main.time_s - graces[0].time_s <= 0.045


def test_envelope_alone_is_not_enough_for_a_flam():
    y, ev, pred, pick = _flam_inputs(0.0)  # audio still contains the grace hit, model shows nothing
    out, _ = art.annotate(ev, y, pred, pick)
    assert not any(e.articulation in ("flam", "grace") for e in out)


def test_refine_moves_early_detector_times_onto_the_true_attack():
    from backend.validation.synth import ROCK, render
    x, truth = render(ROCK, 100, 4)
    env = art.Env(art.band_power(x))
    ev, true_t = [], []
    for inst, ts in truth.items():
        for t in ts:  # detector-like: 12 ms before the real attack
            ev.append(Event(f"e{len(ev)}", t - 0.012, 0, inst, 1.0, {}))
            true_t.append(t)
    assert art.refine_onsets(ev, env) >= 0.9 * len(ev)
    errs = [abs(e.time_s - t) for e, t in zip(ev, true_t, strict=True)]
    assert np.median(errs) < 0.002
    unrefined = [e for e, err in zip(ev, errs, strict=True) if err > 0.006]
    # only hits at the very start of the file (search window falls off the edge) keep the detector time
    assert all("frame_time_s" not in e.model_output for e in unrefined)
    assert all(e.time_s < 0.03 for e in unrefined) and len(unrefined) <= 3
    assert all(e.model_output["frame_time_s"] == pytest.approx(t - 0.012) for e, t in zip(ev, true_t, strict=True)
               if "frame_time_s" in e.model_output)  # the detector's own time is preserved


def test_hihat_at_the_very_end_of_the_audio_does_not_crash():
    """Regression (found on a real upload): no measurable tail after the last hit used to raise."""
    y = (np.random.default_rng(0).standard_normal(int(1.0 * SR)) * 0.1).astype(np.float32)
    ev = [Event("a", 0.995, 0, "hihat", 0.5, {}), Event("b", 0.5, 0, "hihat", 0.5, {})]
    out, _ = art.annotate(ev, y)
    last = max(out, key=lambda e: e.time_s)  # ids are renumbered by annotate
    assert last.articulation != "open" and last.articulation_features["tail_ratio"] is None  # nothing to measure -> closed


def test_annotate_never_crashes_on_hits_at_the_edges_of_short_audio():
    rng = np.random.default_rng(1)
    for dur in (0.3, 0.97, 2.0):
        y = (rng.standard_normal(int(dur * SR)) * 0.1).astype(np.float32)
        for _ in range(25):
            ev = [Event(f"e{i}", float(rng.choice([0.0, 0.005, dur - 0.01, dur, dur + 0.3, rng.uniform(0, dur)])), 0,
                        str(rng.choice(["kick", "snare", "tom", "hihat", "cymbal"])), 0.5, {}) for i in range(int(rng.integers(1, 12)))]
            art.annotate(ev, y)
