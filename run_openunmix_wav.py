"""Run Open-Unmix on a 16-bit PCM WAV with newer torchaudio releases.

Open-Unmix 1.3 calls torchaudio.info/load/save APIs removed in torchaudio 2.9.
This small adapter uses Python's standard-library WAV reader/writer for audio
I/O, while leaving Open-Unmix's model and separation code unchanged.
"""

import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio

import openunmix.data as openunmix_data


def _read_info(path: str) -> dict:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
            raise ValueError("This adapter currently supports uncompressed 16-bit PCM WAV files.")
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        channels = wav_file.getnchannels()
    return {
        "samplerate": sample_rate,
        "samples": frames,
        "channels": channels,
        "duration": frames / sample_rate,
    }


def _read_audio(path: str, start: float = 0.0, dur=None, info=None):
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
            raise ValueError("This adapter currently supports uncompressed 16-bit PCM WAV files.")
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        start_frame = max(0, int(start * sample_rate))
        wav_file.setpos(min(start_frame, wav_file.getnframes()))
        frame_count = -1 if dur is None else max(0, int(dur * sample_rate))
        raw = wav_file.readframes(frame_count)

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    samples = samples.reshape(-1, channels).T.copy()
    return torch.from_numpy(samples), sample_rate


def _write_audio(path, source, sample_rate, channels_first=True, **kwargs):
    samples = source.detach().to("cpu").numpy()
    if samples.ndim == 1:
        samples = samples[:, None]
    elif channels_first:
        samples = samples.T
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())


# Supply the audio I/O functions Open-Unmix 1.3 expects, without changing its model.
openunmix_data.load_info = _read_info
openunmix_data.load_audio = _read_audio
torchaudio.save = _write_audio

from openunmix.cli import separate


if __name__ == "__main__":
    separate()
