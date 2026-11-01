import numpy as np
import pytest
import soundfile as sf

from backend.player import score_audio as sa

pytestmark = pytest.mark.skipif(sa.soundfont_path() is None, reason="no drum SoundFont installed")


def test_notes_follow_the_notation_incl_ghost_and_flam():
    bars = [[set() for _ in range(16)] for _ in range(2)]
    bars[0][0] = {"kick", "hihat"}
    bars[0][8] = {"snare_ghost"}
    bars[1][4] = {"snare_flam", "hihat_open"}
    notes = sa.score_notes(bars, [0.0, 2.0, 4.0])  # 2 s bars -> slot = 0.125 s
    by_time = {}
    for t, p, v in notes:
        by_time.setdefault(round(t, 3), []).append((p, v))
    assert sorted(by_time[0.0]) == [(36, 100), (42, 100)]
    assert by_time[1.0] == [(38, sa.GHOST_VEL)]  # ghost note is quiet
    assert by_time[2.5 - sa.GRACE_OFFSET_S] == [(38, sa.GRACE_VEL)]  # flam grace note just before
    assert sorted(by_time[2.5]) == [(38, 100), (46, 100)]  # main stroke + open hi-hat (GM 46)


def test_rendered_audio_is_a_percussive_kit_placed_at_the_notated_times(tmp_path):
    import librosa
    notes = [(0.5, 36, 100), (1.5, 38, 100), (2.5, 42, 100), (3.5, 49, 100)]
    out = sa.render(notes, tmp_path / "k.flac", tail_s=1.0)
    y, sr = sf.read(out, always_2d=True)
    assert sr == 44100 and y.shape[1] == 2 and 4.9 < len(y) / sr < 6.5
    mono = y.mean(1)
    env = np.abs(mono)
    for t, *_ in notes:  # silent just before, loud within 20 ms after the notated time
        assert env[int((t - 0.05) * sr):int((t - 0.005) * sr)].max() < 0.02
        assert env[int(t * sr):int((t + 0.02) * sr)].max() > 0.02
    ym = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=22050)
    h, p = librosa.effects.harmonic(ym, margin=3), librosa.effects.percussive(ym, margin=3)
    assert (h**2).sum() / ((h**2).sum() + (p**2).sum()) < 0.3  # MuseScore's own render measured 0.98


def test_ghost_is_quieter_than_a_full_hit_and_unknown_kit_rejected(tmp_path):
    out = sa.render([(0.5, 38, sa.GHOST_VEL), (1.5, 38, 100)], tmp_path / "g.flac", tail_s=0.5)
    y, sr = sf.read(out)
    y = y.mean(1) if y.ndim > 1 else y
    quiet, loud = np.abs(y[int(0.5 * sr):int(0.6 * sr)]).max(), np.abs(y[int(1.5 * sr):int(1.6 * sr)]).max()
    assert quiet < 0.7 * loud
    with pytest.raises(ValueError):
        sa.render([(0.1, 36, 100)], tmp_path / "x.flac", kit="theremin")
