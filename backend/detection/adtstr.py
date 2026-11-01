"""ADT_STR (third_party/adt-str, 26 drum classes) with fast cached decoding.

The model's own greedy decoder recomputes the whole target sequence at every step (no key/value cache),
which made it >12x slower than real time on this CPU. Here the SAME weights are run step by step with a
cache: the cross-attention keys/values over the audio are computed once per chunk and each decoder layer
keeps the self-attention keys/values of earlier tokens. The arithmetic is the same as
nn.TransformerDecoderLayer (post-norm, GELU, dropout off in eval), so the output tokens are identical; a test
compares against the model's own `sample()`. Chunking and token decoding are the original code.
Licence: CC BY-SA 4.0 per the ADT_STR GitHub README (unconfirmed on the model card), see MODEL_REGISTRY.md.
"""
import math
import pathlib
import sys
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "adt-str"
VARIANT = "setting-tau-0.8"

# ADT_STR class id -> (our instrument, articulation)
CLASSES = {
    35: ("kick", None), 36: ("kick", None), 37: ("snare", "side_stick"), 38: ("snare", None), 39: ("snare", "clap"),
    40: ("snare", None), 41: ("tom", None), 45: ("tom", None), 47: ("tom", None),
    42: ("hihat", None), 43: ("hihat", "pedal"), 44: ("hihat", "open"),
    46: ("cymbal", "crash"), 48: ("cymbal", "ride"), 49: ("cymbal", "china"), 51: ("cymbal", "splash"),
    50: ("percussion", "tambourine"), 55: ("percussion", "shaker"), 52: ("percussion", "cowbell"), 60: ("percussion", "triangle"),
}


def available():
    return (ROOT / VARIANT / "model.safetensors").exists()


@lru_cache(maxsize=1)
def load():
    sys.path.insert(0, str(ROOT))
    from adt_transcriber import ADTTranscriber
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return ADTTranscriber.from_pretrained(str(ROOT), variant=VARIANT, device="cpu")


def _split_in_proj(attn, x, which):
    w, b = attn.in_proj_weight, attn.in_proj_bias
    d = attn.embed_dim
    sl = {"q": slice(0, d), "k": slice(d, 2 * d), "v": slice(2 * d, 3 * d), "kv": slice(d, 3 * d)}[which]
    return F.linear(x, w[sl], b[sl])


def _heads(x, h):
    b, t, d = x.shape
    return x.view(b, t, h, d // h).transpose(1, 2)  # (b, h, t, dh)


def _attend(q, k, v, attn):
    out = F.scaled_dot_product_attention(q, k, v)  # (b, h, tq, dh); no mask: the cache only holds the past
    b, h, t, dh = out.shape
    return F.linear(out.transpose(1, 2).reshape(b, t, h * dh), attn.out_proj.weight, attn.out_proj.bias)


@torch.no_grad()
def greedy_cached(model, src, max_length, start_token, end_token):
    """Same result as ADTModel.sample(), with a key/value cache."""
    memory = model.encoder(model.project_to_mel(model.compute_spectrogram(src)))
    dec = model.decoder
    layers = dec.decoder.layers
    h = layers[0].self_attn.num_heads
    cross = []  # per layer: (k, v) of the encoder memory, computed once
    for lyr in layers:
        kv = _split_in_proj(lyr.multihead_attn, memory, "kv")
        k, v = kv.chunk(2, dim=-1)
        cross.append((_heads(k, h), _heads(v, h)))
    cache = [None] * len(layers)
    bsz = src.shape[0]
    generated = torch.full((bsz, 1), start_token, dtype=torch.long)
    finished = torch.zeros(bsz, dtype=torch.bool)
    scale = math.sqrt(dec.num_features)
    for step in range(max_length - 1):
        tok = generated[:, -1:]
        x = dec.tgt_tok_emb.embedding(tok) * scale + dec.positional_encoding.pos_embedding[:, step:step + 1, :]
        for i, lyr in enumerate(layers):
            q, k, v = (_heads(_split_in_proj(lyr.self_attn, x, n), h) for n in ("q", "k", "v"))
            if cache[i] is not None:
                k, v = torch.cat([cache[i][0], k], 2), torch.cat([cache[i][1], v], 2)
            cache[i] = (k, v)
            x = lyr.norm1(x + _attend(q, k, v, lyr.self_attn))
            q2 = _heads(_split_in_proj(lyr.multihead_attn, x, "q"), h)
            x = lyr.norm2(x + _attend(q2, *cross[i], lyr.multihead_attn))
            x = lyr.norm3(x + lyr.linear2(lyr.activation(lyr.linear1(x))))
        nxt = torch.argmax(dec.generator(x[:, -1, :]), dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, end_token), nxt)
        generated = torch.cat([generated, nxt[:, None]], 1)
        finished |= nxt == end_token
        if bool(finished.all()):
            break
    return generated


def transcribe(y, sr, batch_size=16):
    """y: mono float audio at `sr`. Returns array of (onset_s, pitch_class, velocity), sorted by time."""
    import librosa
    tr = load()
    if sr != tr.sample_rate:
        y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=tr.sample_rate)
    sys.path.insert(0, str(ROOT))
    from adt_transcriber import _chunk_audio
    wav = torch.from_numpy(np.asarray(y, np.float32))[None]
    chunks = _chunk_audio(wav, tr.chunk_samples)
    max_length = tr.cfg.get("inference", {}).get("max_length", 1024)
    notes = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tokens = greedy_cached(tr.model, torch.cat([c[1] for c in batch]), max_length, tr.tokenizer.BOS_token, tr.tokenizer.EOS_token)
        for (start, _), seq in zip(batch, tokens, strict=True):
            n = tr.tokenizer.decode(seq.numpy())
            if n.numel():
                n[:, 0:2] += start / tr.sample_rate
                notes.append(n.numpy())
    if not notes:
        return np.zeros((0, 3))
    allnotes = np.unique(np.concatenate(notes), axis=0)  # same de-duplication as the original
    out = allnotes[:, [0, 2, 3]]
    return out[np.argsort(out[:, 0], kind="stable")]


MATCH_S = 0.04


def relabel(events, hits):
    """Take the LABELS from ADT_STR, keep our hits and timing. hits: (onset_s, class, velocity).
    - hi-hat: open / closed from the nearest ADT_STR hi-hat note; a 'hi-hat' that ADT_STR hears only as
      tambourine/shaker (no hat within MATCH_S) becomes instrument 'percussion' with that articulation
    - cymbal: crash / ride / china / splash from the nearest ADT_STR cymbal note
    Returns counts of what changed. Events without an ADT_STR counterpart keep their current label."""
    import bisect
    by = {}
    for t, c, _v in hits:
        if int(c) in CLASSES:
            by.setdefault(CLASSES[int(c)][0], []).append((float(t), CLASSES[int(c)][1]))
    for v in by.values():
        v.sort()

    def nearest(kind, t):
        arr = by.get(kind, [])
        i = bisect.bisect_left(arr, (t,))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(arr) and abs(arr[j][0] - t) <= MATCH_S and (best is None or abs(arr[j][0] - t) < abs(best[0] - t)):
                best = arr[j]
        return best
    changed = {"hat_open": 0, "hat_closed": 0, "to_percussion": 0, "cymbal_typed": 0}
    for e in events:
        if e.grace_of is not None:
            continue
        if e.instrument == "hihat":
            h = nearest("hihat", e.time_s)
            if h:
                e.articulation = "open" if h[1] == "open" else None
                changed["hat_open" if h[1] == "open" else "hat_closed"] += 1
            else:
                p = nearest("percussion", e.time_s)
                if p and p[1] in ("tambourine", "shaker"):
                    e.instrument, e.articulation = "percussion", p[1]
                    changed["to_percussion"] += 1
        elif e.instrument == "cymbal":
            c = nearest("cymbal", e.time_s)
            if c:
                e.articulation = c[1]
                changed["cymbal_typed"] += 1
        if e.instrument in ("hihat", "cymbal", "percussion"):
            e.articulation_features = {**(e.articulation_features or {}), "label_source": "adt_str"}
    return changed
