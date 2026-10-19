"""Explicit, whole-segment quality metrics and block timing summaries.

Audio arrays are ``(frames, channels)``. A one-dimensional array is accepted as
mono. SI-SDR removes each channel's mean over the whole supplied segment, then
flattens all channels and fits ONE projection gain. This preserves relative
stereo channel gains. There is no alignment search or reference filtering.

These whole-segment scores are not museval / BSSEval windowed SDR and must not
be used to reproduce a paper's SDR result unless its protocol matches exactly.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


SILENCE_RMS = 1e-8
NUMERICAL_ENERGY_FLOOR_RATIO = 1e-12
SCORE_LIMIT_DB = 120.0


def _audio_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not all(array.shape):
        raise ValueError(f"{name} must be a nonempty (frames, channels) array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _matching_audio(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = _audio_array(reference, "reference")
    est = _audio_array(estimate, "estimate")
    if ref.shape != est.shape:
        raise ValueError(f"reference and estimate shapes differ: {ref.shape} vs {est.shape}")
    return ref, est


def _finite_db(numerator: float, denominator: float, reference_energy: float) -> float:
    # A relative floor and an explicit cap make perfect/orthogonal cases JSON
    # safe. No score is affected except at extreme, numerically fragile ratios.
    floor = NUMERICAL_ENERGY_FLOOR_RATIO * reference_energy
    value = 10.0 * np.log10(max(numerator, floor) / max(denominator, floor))
    return float(np.clip(value, -SCORE_LIMIT_DB, SCORE_LIMIT_DB))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> tuple[float | None, str | None]:
    """Return ``(SI-SDR in dB, undefined reason)`` for a supplied segment.

    Formula: alpha = <estimate, reference> / ||reference||^2;
    SI-SDR = 10 log10(||alpha*reference||^2 / ||estimate-alpha*reference||^2).
    All inner products jointly include the centered stereo channels. Silent
    reference and silent estimate are undefined rather than plausible scores.
    """
    ref, est = _matching_audio(reference, estimate)
    ref = (ref - ref.mean(axis=0, keepdims=True)).reshape(-1)
    est = (est - est.mean(axis=0, keepdims=True)).reshape(-1)
    reference_energy = float(np.dot(ref, ref))
    if reference_energy / ref.size <= SILENCE_RMS**2:
        return None, "reference_silent_after_per_channel_zero_mean"
    estimate_energy = float(np.dot(est, est))
    if estimate_energy / est.size <= SILENCE_RMS**2:
        return None, "estimate_silent_after_per_channel_zero_mean"
    alpha = float(np.dot(est, ref) / reference_energy)
    projected = alpha * ref
    residual = est - projected
    return _finite_db(float(np.dot(projected, projected)), float(np.dot(residual, residual)), reference_energy), None


def global_residual_sdr(reference: np.ndarray, estimate: np.ndarray) -> tuple[float | None, str | None]:
    """Raw energy/error ratio, WITHOUT centering or fitted scale/filter.

    This diagnostic penalizes volume errors and is deliberately named differently
    from museval's BSSEval SDR. It is not interchangeable with paper SDR values.
    """
    ref, est = _matching_audio(reference, estimate)
    ref = ref.reshape(-1)
    est = est.reshape(-1)
    reference_energy = float(np.dot(ref, ref))
    if reference_energy / ref.size <= SILENCE_RMS**2:
        return None, "reference_silent"
    residual = ref - est
    return _finite_db(reference_energy, float(np.dot(residual, residual)), reference_energy), None


def separation_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    mixture: np.ndarray | None = None,
) -> dict:
    """Return JSON-safe quality scores with definitions and undefined reasons."""
    ref, est = _matching_audio(reference, estimate)
    score, reason = si_sdr(ref, est)
    residual_score, residual_reason = global_residual_sdr(ref, est)
    ref_energy = float(np.sum(ref * ref))
    if ref_energy / ref.size > SILENCE_RMS**2:
        normalized_rmse = float(np.sqrt(np.sum((est-ref)**2) / ref_energy))
        projected_gain = float(np.sum(est*ref) / ref_energy)
    else:
        normalized_rmse = projected_gain = None
    mixture_score = None
    mixture_reason = "mixture_not_supplied"
    improvement = None
    improvement_reason = "mixture_not_supplied"
    if mixture is not None:
        mixture_score, mixture_reason = si_sdr(ref, mixture)
        if score is not None and mixture_score is not None:
            improvement = float(score - mixture_score)
            improvement_reason = None
        else:
            improvement_reason = reason or mixture_reason
    return {
        "si_sdr_db": score,
        "si_sdr_undefined_reason": reason,
        "mixture_si_sdr_db": mixture_score,
        "mixture_si_sdr_undefined_reason": mixture_reason,
        "si_sdri_db": improvement,
        "si_sdri_undefined_reason": improvement_reason,
        "global_residual_sdr_db": residual_score,
        "global_residual_sdr_undefined_reason": residual_reason,
        "normalized_waveform_rmse": normalized_rmse,
        "reference_projection_gain": projected_gain,
        "preservation_diagnostic_scope": "unaligned raw waveform error; no gain fitting; interpret especially on original-vocal identity inputs",
        "frames": int(ref.shape[0]),
        "channels": int(ref.shape[1]),
        "protocol": {
            "segment": "entire_supplied_segment",
            "si_sdr_centering": "subtract_whole_segment_mean_per_channel",
            "si_sdr_channel_reduction": "flatten_channels_then_fit_one_shared_gain",
            "alignment": "none; caller must preserve time alignment",
            "global_residual_sdr": "10log10(raw_reference_energy/raw_error_energy)",
            "silence_rms_threshold": SILENCE_RMS,
            "numerical_energy_floor_ratio": NUMERICAL_ENERGY_FLOOR_RATIO,
            "score_cap_db": SCORE_LIMIT_DB,
            "museval_bsseval_sdr": False,
        },
    }


def timing_summary(processing_seconds: Iterable[float], audio_seconds: Iterable[float]) -> dict:
    """Summarize already-measured processing times for complete audio blocks.

    The caller owns device synchronization and the timing scope. This function
    never measures, synchronizes, or infers algorithmic latency. Deadline misses
    mean processing_time > block_audio_duration. Warmup/model loading must be
    excluded by the caller if reporting model inference throughput.
    """
    processing = np.asarray(list(processing_seconds), dtype=np.float64)
    durations = np.asarray(list(audio_seconds), dtype=np.float64)
    if processing.ndim != 1 or durations.ndim != 1 or not processing.size:
        raise ValueError("timing arrays must be nonempty, one-dimensional sequences")
    if processing.shape != durations.shape:
        raise ValueError("timing arrays must contain the same number of blocks")
    if not np.isfinite(processing).all() or not np.isfinite(durations).all():
        raise ValueError("timing arrays contain NaN or infinity")
    if (processing < 0).any() or (durations <= 0).any():
        raise ValueError("processing seconds must be >= 0; audio seconds must be > 0")
    ratios = processing / durations
    misses = int(np.count_nonzero(processing > durations))
    return {
        "block_count": int(processing.size),
        "mean_rtr": float(ratios.mean()),
        "p95_rtr": float(np.percentile(ratios, 95)),
        "p99_rtr": float(np.percentile(ratios, 99)),
        "max_rtr": float(ratios.max()),
        "total_rtr": float(processing.sum() / durations.sum()),
        "deadline_miss_count": misses,
        "deadline_miss_fraction": float(misses / processing.size),
        "total_processing_seconds": float(processing.sum()),
        "total_audio_seconds": float(durations.sum()),
        "percentile_method": "numpy_linear",
        "latency_inferred_from_rtr": False,
    }
