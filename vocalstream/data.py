"""Direct MUSDB18-HQ WAV reader. Does not import musdb/stempeg/FFmpeg."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import soundfile as sf

STEMS = ("mixture", "vocals", "drums", "bass", "other")


def dataset_info(root):
    root = Path(root)
    if (root / 'SYNTHETIC_ONLY.json').is_file():
        return {"kind": "synthetic_engineering_fixture", "formal_benchmark": False}
    marker = root / '_DATASET_INFO.json'
    if marker.is_file():
        value = json.loads(marker.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or not isinstance(value.get('kind'), str):
            raise ValueError('Invalid dataset metadata')
        return value
    return {"kind": "MUSDB18-HQ", "formal_benchmark": True,
            "source_verification": "user-provided folder; confirm official release provenance"}


@dataclass(frozen=True)
class HQTrack:
    path: Path
    subset: str
    name: str
    sample_rate: int
    frames: int
    channels: int

    def as_dict(self):
        return {"name": self.name, "subset": self.subset,
                "sample_rate": self.sample_rate, "frames": self.frames,
                "duration_seconds": self.frames / self.sample_rate,
                "channels": self.channels}


def discover_hq(root, subset="test", expected_rate=44100):
    if subset not in ("train", "test"):
        raise ValueError("subset must be train or test")
    folder = Path(root).expanduser().resolve() / subset
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing {folder}. Expected root/train/<song>/mixture.wav and root/test/<song>/mixture.wav")
    tracks = []
    for path in sorted(p for p in folder.iterdir() if p.is_dir()):
        infos = {}
        for stem in STEMS:
            wav = path / f"{stem}.wav"
            if not wav.is_file():
                raise FileNotFoundError(f"Missing stem: {wav}")
            infos[stem] = sf.info(wav)
        first = infos["mixture"]
        if first.samplerate != expected_rate or first.channels != 2:
            raise ValueError(f"{path.name}: expected stereo {expected_rate} Hz, found {first.channels} channels {first.samplerate} Hz")
        if first.frames <= 0:
            raise ValueError(f"{path.name}: empty audio")
        for stem, info in infos.items():
            if (info.frames, info.channels, info.samplerate) != (first.frames, first.channels, first.samplerate):
                raise ValueError(f"{path.name}/{stem}: stem length/rate/channels do not match mixture")
        tracks.append(HQTrack(path, subset, path.name, first.samplerate, first.frames, first.channels))
    if not tracks:
        raise ValueError(f"No track directories found in {folder}")
    return tracks


def load_segment(track, start=0.0, duration=None):
    if not np.isfinite(start) or start < 0:
        raise ValueError("start must be finite and >= 0")
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and > 0")
    begin = int(round(start * track.sample_rate))
    if begin >= track.frames:
        raise ValueError(f"start {start}s is past track {track.name}")
    end = track.frames if duration is None else min(track.frames, begin + int(round(duration * track.sample_rate)))
    if end <= begin:
        raise ValueError("Segment has zero samples")
    arrays = {}
    for stem in STEMS:
        audio, rate = sf.read(track.path / f"{stem}.wav", start=begin, stop=end,
                              dtype="float32", always_2d=True)
        if rate != track.sample_rate or not np.isfinite(audio).all():
            raise ValueError(f"Invalid audio in {track.name}/{stem}")
        arrays[stem] = audio
    arrays["accompaniment"] = arrays["drums"] + arrays["bass"] + arrays["other"]
    # Official mixture is retained, rather than silently replacing it by a sum.
    arrays["linear_mixture"] = arrays["vocals"] + arrays["accompaniment"]
    error = arrays["mixture"].astype(np.float64) - arrays["linear_mixture"]
    info = {"start_sample": begin, "samples": end - begin,
            "duration_seconds": (end - begin) / track.sample_rate,
            "mixture_sum_max_abs_difference": float(np.max(np.abs(error))),
            "mixture_sum_rms_difference": float(np.sqrt(np.mean(error ** 2)))}
    return arrays, info
