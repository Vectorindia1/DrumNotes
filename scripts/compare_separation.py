"""Original-vs-separated detection comparison (PRD 11, M2 exit). Never assumes separation helps.

A) synthetic MIXES with exact truth (drums + synthetic bass/chords at 0 dB and -6 dB drum level)
B) the user's real clip: no truth, so only agreement between the two detections is reported.
Both are weak evidence; real answer-key clips (data/benchmarks/C) are still needed."""
import json
import pathlib
import sys
import tempfile

import soundfile as sf

from backend.detection import adtof
from backend.separation.demucs_sep import separate_drums
from backend.validation.metrics import match, score
from backend.validation.synth import ROCK, SR, render_mix

PATTERNS = {
    "rock_100bpm": (ROCK, 100, 6),
    "stacked_plus_offbeat_hats": ([("kick", 0), ("snare", 0), ("hihat", 0), ("kick", 4), ("snare", 4), ("hihat", 4),
                                   ("hihat", 2), ("hihat", 6)], 100, 4),
}
INSTS = ("kick", "snare", "hihat")


def times(path):
    ev, _ = adtof.detect(path)
    return {i: [e.time_s for e in ev if e.instrument == i] for i in INSTS + ("tom", "cymbal")}


def synthetic(d):
    rows = {}
    for name, (pattern, bpm, bars) in PATTERNS.items():
        for snr in (0.0, -6.0):
            mix, _, truth = render_mix(pattern, bpm, bars, snr)
            wav = pathlib.Path(d) / f"{name}_{snr:+.0f}dB.wav"
            sf.write(wav, mix, SR)
            stem = separate_drums(wav, pathlib.Path(d) / f"{wav.stem}.drums.wav")
            for label, path in (("original", wav), ("separated", stem)):
                t = times(path)
                for inst in INSTS:
                    s = score(t[inst], truth[inst])["by_tolerance_ms"][50]
                    extra = sum(len(t[k]) for k in t if k not in INSTS)  # tom/cymbal = false positives here
                    rows[f"{name} {snr:+.0f}dB | {label} | {inst}"] = {**s, "other_class_fp": extra}
                print(name, snr, label, {i: rows[f"{name} {snr:+.0f}dB | {label} | {i}"]["f1"] for i in INSTS}, flush=True)
    return rows


def real_clip(clip, stem):
    a, b = times(clip), times(stem)
    out = {}
    for inst in INSTS + ("tom", "cymbal"):
        both = len(match(a[inst], b[inst], 0.05))
        out[inst] = {"original": len(a[inst]), "stem": len(b[inst]), "matched_within_50ms": both,
                     "only_original": len(a[inst]) - both, "only_stem": len(b[inst]) - both}
    return out


if __name__ == "__main__":
    res = {}
    with tempfile.TemporaryDirectory() as d:
        res["A_synthetic_mix"] = synthetic(d)
    clip, stem = pathlib.Path("data/incoming/user_clip.mp3"), pathlib.Path("data/stems/user_clip.drums.wav")
    if clip.exists() and stem.exists():
        res["B_real_clip_agreement_no_truth"] = real_clip(clip, stem)
    pathlib.Path("outputs/json").mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs/json/separation_comparison.json").write_text(json.dumps(res, indent=1))
    json.dump(res.get("B_real_clip_agreement_no_truth"), sys.stdout, indent=1)
