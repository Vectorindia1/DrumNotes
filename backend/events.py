"""Internal event representation (versioned JSON schema). Raw timestamps are never overwritten."""
from dataclasses import asdict, dataclass

SCHEMA_VERSION = "1.0"
INSTRUMENTS = ("kick", "snare", "tom", "hihat", "cymbal")


@dataclass
class Event:
    id: str
    time_s: float  # raw detected time; quantization writes only to the fields below
    sample: int  # round(time_s * sample_rate)
    instrument: str
    confidence: float  # model activation at the peak; NOT a calibrated probability
    model_output: dict  # {"model", "label", "activation"} as emitted by the detector
    articulation: str | None = None  # M4
    articulation_confidence: float | None = None
    ghost_probability: float | None = None  # heuristic score, NOT a calibrated probability
    articulation_features: dict | None = None  # raw measurements behind the articulation label
    bar: int | None = None  # 0-based
    slot: int | None = None  # 16th-note slot within the bar
    beat_position: float | None = None  # beats from grid start, unquantized
    quantization_error_s: float | None = None
    grace_of: str | None = None  # id of the main event when this is a flam grace note
    notated: bool = True  # False when merged into a same-instrument event in the same slot
    user_edited: bool = False
    provenance: str = "detected"

    def to_dict(self):
        return asdict(self)
