"""Onset benchmarks, reported per category (PRD 7.3). Never pooled into one accuracy number.

A: synthetic controlled patterns (generated here, exact truth)       -> runs
B: isolated drum recordings, C: full mixes (real, licensed, annotated) -> read from
   data/benchmarks/{B,C}/<clip>/{audio.*, truth.json}; truth.json = {"kick": [sec,...], ...}
   Missing data is reported BLOCKED, never fabricated.
"""
import json
import pathlib
import tempfile

import soundfile as sf

from backend.detection import adtof
from backend.validation.metrics import score
from backend.validation.synth import ROCK, SR, render

PATTERNS = {
    "rock_100bpm": (ROCK, 100, 8),
    "rock_140bpm": (ROCK, 140, 8),
    "stacked_kick_snare_hat": ([("kick", 0), ("snare", 0), ("hihat", 0), ("kick", 4), ("snare", 4), ("hihat", 4)], 100, 4),
    "stacked_plus_offbeat_hats": ([("kick", 0), ("snare", 0), ("hihat", 0), ("kick", 4), ("snare", 4), ("hihat", 4),
                                   ("hihat", 2), ("hihat", 6)], 100, 4),
}
COMPARED = ("kick", "snare", "hihat")  # toms/cymbals are not synthesized


def evaluate(path, truth):
    events, _ = adtof.detect(path)
    return {inst: score([e.time_s for e in events if e.instrument == inst], truth[inst]) for inst in COMPARED if truth.get(inst)}


def category_a():
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for name, (pattern, bpm, bars) in PATTERNS.items():
            audio, truth = render(pattern, bpm, bars)
            wav = pathlib.Path(d) / f"{name}.wav"
            sf.write(wav, audio, SR)
            out[name] = evaluate(wav, truth)
    return out


def category_real(cat):
    root = pathlib.Path("data/benchmarks") / cat
    clips = sorted(p for p in root.glob("*/truth.json")) if root.exists() else []
    if not clips:
        return "BLOCKED: no licensed annotated clips in data/benchmarks/" + cat
    out = {}
    for t in clips:
        audio = next(t.parent.glob("audio.*"))
        out[t.parent.name] = evaluate(audio, json.loads(t.read_text()))
    return out


def show(title, res):
    print(f"\n== {title}")
    if isinstance(res, str):
        print(res)
        return
    for clip, per in res.items():
        for inst, s in per.items():
            row = "  ".join(f"±{ms}ms F1={v['f1']:.2f}" for ms, v in s["by_tolerance_ms"].items())
            t50 = s["by_tolerance_ms"][50]
            print(f"{clip:28s} {inst:6s} ref={s['n_reference']:3d} det={s['n_detected']:3d} P={t50['precision']:.2f} "
                  f"R={t50['recall']:.2f} med={s['median_error_ms']}ms p95={s['p95_error_ms']}ms | {row}")


if __name__ == "__main__":
    results = {"A_synthetic": category_a(), "B_isolated_real": category_real("B"), "C_full_mix_real": category_real("C")}
    for k, v in results.items():
        show(k, v)
    out = pathlib.Path("outputs/json")
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmarks.json").write_text(json.dumps(results, indent=1))
