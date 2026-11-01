"""Download + verify model weights once (PRD 4.6). Demucs htdemucs only; ADTOF weights ship in its repo.
Demucs 4.1.0 pulls weights from the Hugging Face repo adefossez/HTDemucs (falls back to dl.fbaipublicfiles.com)."""
import hashlib
import pathlib

from demucs.pretrained import get_model

model = get_model("htdemucs")
print("sources", model.sources, "samplerate", model.samplerate)
for p in sorted((pathlib.Path.home() / ".cache/huggingface/hub").glob("models--adefossez--HTDemucs/snapshots/*/*.safetensors")):
    print(p.name, p.stat().st_size, "bytes", "sha256", hashlib.sha256(p.resolve().read_bytes()).hexdigest())
