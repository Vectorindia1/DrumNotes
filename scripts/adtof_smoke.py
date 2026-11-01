"""ADTOF-pytorch smoke test on SYNTHETIC drums (plumbing check only; not real-world accuracy)."""
import numpy as np
import soundfile as sf
import torch
from adtof_pytorch import (
    FRAME_RNN_THRESHOLDS,
    LABELS_5,
    PeakPicker,
    calculate_n_bins,
    create_frame_rnn_model,
    get_default_weights_path,
    load_audio_for_model,
)

SR, BPM, BARS = 44100, 100, 8
rng = np.random.default_rng(0)


def kick(n=int(.25 * SR)):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 30)).cumsum() / SR) * np.exp(-t * 14)


def snare(n=int(.2 * SR)):
    t = np.arange(n) / SR
    return (rng.standard_normal(n) * .7 + np.sin(2 * np.pi * 190 * t) * .5) * np.exp(-t * 22)


def hat(n=int(.06 * SR)):
    t = np.arange(n) / SR
    return np.diff(rng.standard_normal(n + 1)) * np.exp(-t * 70) * .35


step = 60 / BPM / 2  # eighth
bar = [("kick", 0), ("kick", 4), ("kick", 5), ("snare", 2), ("snare", 6)] + [("hat", i) for i in range(8)]
truth, x = {"kick": [], "snare": [], "hat": []}, np.zeros(int((BARS * 8 * step + 1) * SR), np.float32)
for b in range(BARS):
    for name, i in bar:
        t = (b * 8 + i) * step  # first kick is exactly at sample 0
        truth[name].append(t)
        s = {"kick": kick, "snare": snare, "hat": hat}[name]()
        a = round(t * SR); x[a:a + len(s)] += s
x /= np.abs(x).max() * 1.1
sf.write("outputs/synth_drums.wav", x, SR)

n_bins = calculate_n_bins()
model = create_frame_rnn_model(n_bins).eval()
sd = torch.load(get_default_weights_path(), map_location="cpu")
sd = sd["model_weights"]
res = model.load_state_dict(sd, strict=False)
print("weights missing:", res.missing_keys, "unexpected:", res.unexpected_keys)
assert not res.missing_keys, "weights did not fully load"
with torch.no_grad():
    pred = model(load_audio_for_model("outputs/synth_drums.wav")).numpy()
print("activations", pred.shape, "max per class", pred[0].max(0).round(3))
PAD = 10  # frames; ADTOF's picker edge-pads its moving average and so drops onsets at frame 0 (PRD 8.3)
padded = np.concatenate([np.zeros((1, PAD, pred.shape[2]), pred.dtype), pred], axis=1)
raw = PeakPicker(thresholds=FRAME_RNN_THRESHOLDS, fps=100).pick(padded, labels=LABELS_5)[0]
peaks = {k: [t - PAD / 100 for t in v if t >= PAD / 100] for k, v in raw.items()}
print("detected counts (35 kick,38 snare,47 tom,42 hat,49 cym):", {k: len(v) for k, v in peaks.items()})


def f1(det, ref, tol=.05):
    det, used, tp = sorted(det), set(), 0
    for r in ref:
        m = [j for j, d in enumerate(det) if j not in used and abs(d - r) <= tol]
        if m: used.add(min(m, key=lambda j: abs(det[j] - r))); tp += 1
    p, r_ = tp / max(len(det), 1), tp / len(ref)
    return tp, len(det), len(ref), round(p, 3), round(r_, 3), round(2 * p * r_ / max(p + r_, 1e-9), 3)


for name, lab in (("kick", 35), ("snare", 38), ("hat", 42)):
    print(name, "(tp, det, ref, P, R, F1 @50ms):", f1(peaks.get(lab, []), truth[name]))
print("first kick truth t=0.0; earliest kick detection:", sorted(peaks.get(35, [None]))[:3])

print("activation frames 0-5 (kick, snare, tom, hat, cym):")
print(pred[0, :6].round(3))
print("peak-picker thresholds:", FRAME_RNN_THRESHOLDS)
