"""Reproducible additive conditions for aligned vocals and accompaniment.

Inputs and returned audio have shape (frames, channels); mono inputs are expanded
to one channel. Outputs are float32. No normalization or clipping is applied.
The returned clean reference is a copy. Source-separation and dry-vocal-restoration
conditions are different tasks and must be grouped separately in reports.

The RIR is an artificial decaying noise kernel, not a measured/simulated physical
room. None of these conditions models acoustic feedback's closed loop.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _audio_copy(value: np.ndarray, name: str) -> np.ndarray:
    audio = np.array(value, dtype=np.float32, copy=True)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or not all(audio.shape):
        raise ValueError(f"{name} must be a nonempty (frames, channels) array")
    if not np.isfinite(audio).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return audio


def _energy(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sum(values * values))


def _ratio_db(target: np.ndarray, interference: np.ndarray) -> tuple[float | None, str | None]:
    target_energy = _energy(target)
    interference_energy = _energy(interference)
    if target_energy <= 1e-16 * target.size:
        return None, "target_silent"
    if interference_energy <= 1e-16 * interference.size:
        return None, "interference_silent"
    return float(10.0 * np.log10(target_energy / interference_energy)), None


def synthetic_rir(
    sample_rate: int,
    rt60_seconds: float = 0.25,
    seed: int = 0,
    direct_to_reverb_db: float = 6.0,
) -> np.ndarray:
    """Create an aligned direct tap plus a deterministic decaying noise tail.

    The tail begins at 12 ms and its *amplitude envelope* reaches -60 dB at
    rt60_seconds. This parameter does not claim the output's measured RT60. The
    direct/tail energy ratio is normalized over the RIR itself, not the song.
    A single RIR is applied to both channels to keep this condition simple.
    """
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not np.isfinite(rt60_seconds) or not 0.02 <= rt60_seconds <= 5.0:
        raise ValueError("rt60_seconds must be between 0.02 and 5.0")
    if not np.isfinite(direct_to_reverb_db) or not -60 <= direct_to_reverb_db <= 60:
        raise ValueError("direct_to_reverb_db must be finite and within [-60, 60]")
    length = max(2, int(round(rt60_seconds * sample_rate)) + 1)
    times = np.arange(length, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    tail = rng.standard_normal(length) * np.exp(-np.log(1000.0) * times / rt60_seconds)
    tail[: max(1, int(round(0.012 * sample_rate)))] = 0.0
    tail_energy = float(np.dot(tail, tail))
    if tail_energy <= 0:
        raise ValueError("sample_rate is too low to construct the reverberant tail")
    tail *= np.sqrt(10.0 ** (-direct_to_reverb_db / 10.0) / tail_energy)
    tail[0] = 1.0
    return tail


def make_condition(
    vocals: np.ndarray,
    accompaniment: np.ndarray | None,
    sample_rate: int,
    mode: str = "mixture",
    seed: int = 0,
    snr_db: float = 10.0,
    bleed_db: float = -20.0,
    rt60_seconds: float = 0.25,
    direct_to_reverb_db: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(input_audio, clean_vocal_reference, condition_metadata)``.

    mixture: vocals + accompaniment, conventional vocal source separation.
    clean: vocals only, an identity sanity check.
    noise: vocals + Gaussian noise at target-relative SNR; dry-vocal restoration.
    reverb: artificial reverberated vocals; dry-vocal restoration.
    bleed: vocals + 10**(bleed_db/20)*accompaniment; close-mic contamination
           stress test. This does not simulate physical microphone acoustics.
    """
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    reference = _audio_copy(vocals, "vocals")
    backing = None if accompaniment is None else _audio_copy(accompaniment, "accompaniment")
    if backing is not None and backing.shape != reference.shape:
        raise ValueError("vocals and accompaniment must have identical aligned shapes")
    if mode not in {"mixture", "clean", "noise", "reverb", "bleed"}:
        raise ValueError(f"unknown condition mode: {mode}")
    metadata = {
        "mode": mode,
        "seed": int(seed),
        "sample_rate": int(sample_rate),
        "frames": int(reference.shape[0]),
        "channels": int(reference.shape[1]),
        "reference": "unaltered_original_vocal_stem",
        "reference_is_stem_not_guaranteed_dry_close_mic": True,
        "normalization_applied": False,
        "clipping_applied": False,
        "closed_loop_acoustic_feedback_simulated": False,
    }
    if mode in {"mixture", "bleed"}:
        if backing is None:
            raise ValueError(f"accompaniment is required for mode={mode}")
        gain = 1.0
        if mode == "bleed":
            if not np.isfinite(bleed_db) or not -120 <= bleed_db <= 60:
                raise ValueError("bleed_db must be finite and within [-120, 60]")
            gain = float(10.0 ** (bleed_db / 20.0))
            metadata.update({
                "task": "close_microphone_vocal_extraction_synthetic",
                "bleed_amplitude_gain_db": float(bleed_db),
                "bleed_amplitude_gain": gain,
                "simulation": "scaled_aligned_accompaniment; no physical_microphone_model",
            })
        else:
            metadata["task"] = "music_vocal_source_separation"
        interference = backing.astype(np.float64) * gain
        result = reference.astype(np.float64) + interference
    elif mode == "clean":
        interference = np.zeros_like(reference)
        result = reference.copy()
        metadata["task"] = "clean_vocal_identity_sanity_check"
    elif mode == "noise":
        if not np.isfinite(snr_db) or not -60 <= snr_db <= 120:
            raise ValueError("snr_db must be finite and within [-60, 120]")
        target_energy = _energy(reference)
        if target_energy <= reference.size * 1e-16:
            raise ValueError("target-relative SNR is undefined for silent vocals")
        interference = np.random.default_rng(seed).standard_normal(reference.shape)
        interference *= np.sqrt(target_energy / (_energy(interference) * 10.0 ** (snr_db / 10.0)))
        result = reference.astype(np.float64) + interference
        actual_snr, _ = _ratio_db(reference, interference)
        metadata.update({
            "task": "dry_vocal_restoration",
            "noise_distribution": "independent_gaussian_per_sample_per_channel",
            "snr_reference": "uncentered_clean_vocal_energy_jointly_over_channels",
            "requested_snr_db": float(snr_db),
            "actual_component_snr_db": actual_snr,
        })
    else:
        rir = synthetic_rir(sample_rate, rt60_seconds, seed, direct_to_reverb_db)
        result = np.column_stack([
            fftconvolve(reference[:, channel].astype(np.float64), rir, mode="full")[: reference.shape[0]]
            for channel in range(reference.shape[1])
        ])
        interference = result - reference
        metadata.update({
            "task": "dry_vocal_restoration",
            "rir_type": "synthetic_decaying_gaussian_tail_plus_aligned_direct_tap",
            "rir_shared_across_channels": True,
            "rir_length_samples": int(rir.size),
            "tail_start_ms": 12.0,
            "rt60_amplitude_envelope_seconds": float(rt60_seconds),
            "rir_direct_to_tail_energy_ratio_db": float(direct_to_reverb_db),
            "tail_truncated_at_signal_end": True,
            "measured_or_physical_room_rir": False,
        })
    ratio, ratio_reason = _ratio_db(reference, interference)
    result = np.asarray(result, dtype=np.float32)
    metadata.update({
        "input_target_to_interference_db": ratio,
        "input_target_to_interference_undefined_reason": ratio_reason,
        "peak_absolute_amplitude": float(np.max(np.abs(result))),
        "peak_exceeds_pcm_unit_range": bool(np.max(np.abs(result)) > 1.0),
        "comparison_group": metadata["task"],
    })
    return result, reference, metadata
