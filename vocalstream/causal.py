"""Educational causal vocal separator with explicit streaming state.

This is a small research baseline, not a reproduction of RT-STT or BS-RoFormer.
The GRU only sees the current and preceding STFT frames.  A frame contains a
trailing window of samples; no centered STFT or bidirectional recurrence is used.

Timing convention: each input hop produces one equally sized output hop.  In
that *untrimmed* output stream, sample j represents input sample j - (N - H).
Thus the sample-stream displacement is D = n_fft - hop_length.  The first
original sample is returned when N input samples have been collected: collecting
the current hop adds H samples to the availability time.  These two quantities
are reported separately.  Neither is a measurement of sound-device round-trip
latency.  Removing the leading D samples in an offline file does not remove the
delay experienced by a live listener.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CausalConfig:
    sample_rate: int = 44_100
    n_fft: int = 1_024
    hop_length: int = 256
    channels: int = 2
    hidden_size: int = 64
    num_layers: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft < 4 or self.n_fft % 2:
            raise ValueError("n_fft must be an even integer of at least four")
        if self.hop_length <= 0 or self.n_fft % self.hop_length:
            raise ValueError("n_fft must be divisible by hop_length")
        if self.hop_length > self.n_fft // 2:
            raise ValueError("Hann reconstruction requires hop_length <= n_fft / 2")
        if self.channels not in (1, 2):
            raise ValueError("this baseline supports mono or stereo")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")

    @property
    def bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def delay_samples(self) -> int:
        return self.n_fft - self.hop_length

    @property
    def flush_hops(self) -> int:
        return self.delay_samples // self.hop_length

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def latency_info(self) -> dict[str, Any]:
        return {
            "algorithmic_delay_samples": self.delay_samples,
            "algorithmic_latency_ms": 1_000 * self.delay_samples / self.sample_rate,
            "latency_definition": "untrimmed output[j] corresponds to input[j - (n_fft - hop_length)]",
            "first_sample_ready_after_input_samples": self.n_fft,
            "first_sample_ready_ms": 1_000 * self.n_fft / self.sample_rate,
            "input_block_duration_ms": 1_000 * self.hop_length / self.sample_rate,
            "lookahead_frames": 0,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "sample_rate": self.sample_rate,
            "end_to_end_latency_measured": False,
        }


def _window(config: CausalConfig, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.hann_window(config.n_fft, periodic=True, device=device, dtype=dtype)


def _waveform_batch(waveform: Tensor, config: CausalConfig) -> Tensor:
    waveform = torch.as_tensor(waveform)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3 or waveform.shape[1] != config.channels:
        raise ValueError(f"waveform must have shape [channels, samples] or [batch, {config.channels}, samples]")
    if waveform.shape[-1] == 0:
        raise ValueError("waveform must not be empty")
    if not waveform.is_floating_point():
        waveform = waveform.float()
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform contains non-finite samples")
    return waveform


def analysis_frames(waveform: Tensor, config: CausalConfig) -> Tensor:
    """Return [batch, frames, channels, bins] complex causal spectra.

    Left padding is previous silence, not future context.  Right padding only
    flushes the finite recording; those frames must arrive as zero input hops
    when this same recording is processed incrementally.
    """
    batch = _waveform_batch(waveform, config)
    length = batch.shape[-1]
    rounded = math.ceil(length / config.hop_length) * config.hop_length
    padded = F.pad(batch, (config.delay_samples, rounded - length + config.delay_samples))
    frames = padded.unfold(-1, config.n_fft, config.hop_length)
    window = _window(config, device=batch.device, dtype=batch.dtype)
    # unfold gives [B, C, T, N]; the model's time axis is explicitly T.
    return torch.fft.rfft(frames * window, dim=-1).permute(0, 2, 1, 3).contiguous()


def synthesize_frames(
    spectrum: Tensor,
    config: CausalConfig,
    original_length: int | None = None,
    *,
    aligned: bool = True,
) -> Tensor:
    """Differentiable overlap-add, using the same window/normalizer as streaming.

    Input is [B, T, C, F]; output is [B, C, samples].  aligned=False retains
    leading silence and the documented sample delay.  Original length limits
    the returned file but never changes what the live processor has to wait for.
    """
    if spectrum.ndim != 4 or spectrum.shape[2:] != (config.channels, config.bins):
        raise ValueError("spectrum must have shape [batch, frames, channels, bins]")
    if not spectrum.is_complex() or spectrum.shape[1] == 0:
        raise ValueError("spectrum must contain complex frames")
    batch_size, frames_count = spectrum.shape[:2]
    window = _window(config, device=spectrum.device, dtype=spectrum.real.dtype)
    frames = torch.fft.irfft(spectrum, n=config.n_fft, dim=-1) * window
    columns = frames.permute(0, 2, 3, 1).reshape(batch_size * config.channels, config.n_fft, frames_count)
    total = (frames_count - 1) * config.hop_length + config.n_fft
    summed = F.fold(columns, (1, total), (1, config.n_fft), stride=(1, config.hop_length))
    norm_columns = window.square().view(1, config.n_fft, 1).expand(1, -1, frames_count)
    normalizer = F.fold(norm_columns, (1, total), (1, config.n_fft), stride=(1, config.hop_length))
    output = (summed / normalizer.clamp_min(1e-8)).reshape(batch_size, config.channels, total)
    # Each push emits H samples.  The remaining OLA tail is padding, not a new
    # input hop, and is also discarded by CausalStreamProcessor.flush().
    output = output[..., : frames_count * config.hop_length]
    if aligned:
        output = output[..., config.delay_samples :]
        if original_length is not None:
            output = output[..., :original_length]
    elif original_length is not None:
        output = output[..., : original_length + config.delay_samples]
    return output


class CausalMaskGRU(nn.Module):
    """Shared stereo features -> one-way GRU -> per-channel real magnitude mask."""

    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        features = config.channels * config.bins
        self.feature_norm = nn.LayerNorm(features)
        self.gru = nn.GRU(features, config.hidden_size, config.num_layers, batch_first=True)
        self.mask_head = nn.Linear(config.hidden_size, features)
        self.trained_optimizer_steps = 0
        self.checkpoint_metadata: dict[str, Any] = {}

    def forward(self, spectrum: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if spectrum.ndim != 4 or spectrum.shape[2:] != (self.config.channels, self.config.bins):
            raise ValueError("spectrum must have shape [batch, frames, channels, bins]")
        magnitudes = spectrum.abs()
        features = torch.log1p(magnitudes).flatten(2)
        features = self.feature_norm(features)
        sequence, next_state = self.gru(features, state)
        mask = torch.sigmoid(self.mask_head(sequence)).reshape_as(magnitudes)
        return spectrum * mask, next_state

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CausalStreamProcessor:
    """One input hop -> one output hop; history, GRU and OLA state persist.

    push() accepts exactly [C, H].  flush() supplies D/H zero hops and closes the
    processor.  identity_mask=True is a DSP diagnostic; it is not a separator.
    A randomly initialized model is deliberately rejected during inference.
    """

    def __init__(
        self,
        model: CausalMaskGRU | None = None,
        config: CausalConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        identity_mask: bool = False,
    ):
        if config is None:
            if model is None:
                raise ValueError("config is required for the identity DSP diagnostic")
            config = model.config
        if model is not None and model.config != config:
            raise ValueError("model configuration differs from stream configuration")
        if not identity_mask and (model is None or model.trained_optimizer_steps <= 0):
            raise ValueError("separation requires a trained checkpoint; use identity_mask=True only to diagnose reconstruction")
        if identity_mask and model is not None:
            raise ValueError("identity diagnostics must not be confused with model separation")
        self.config = config
        self.device = torch.device(device)
        self.model = None if identity_mask else model.to(self.device).eval()
        self.dtype = torch.float32 if self.model is None else next(self.model.parameters()).dtype
        self.window = _window(config, device=self.device, dtype=self.dtype)
        self.window_squared = self.window.square()
        self.history = torch.zeros(config.channels, config.delay_samples, device=self.device, dtype=self.dtype)
        self.ola = torch.zeros(config.channels, config.n_fft, device=self.device, dtype=self.dtype)
        self.normalizer = torch.zeros(config.n_fft, device=self.device, dtype=self.dtype)
        self.gru_state: Tensor | None = None
        self.closed = False
        self.input_hops = 0

    @torch.inference_mode()
    def push(self, samples: Tensor) -> Tensor:
        if self.closed:
            raise RuntimeError("cannot push after flush; create a new processor to reset state")
        samples = torch.as_tensor(samples, dtype=self.dtype, device=self.device)
        if samples.shape != (self.config.channels, self.config.hop_length):
            raise ValueError(f"push requires shape [{self.config.channels}, {self.config.hop_length}]")
        if not torch.isfinite(samples).all():
            raise ValueError("input hop contains non-finite samples")
        frame = torch.cat((self.history, samples), dim=-1)
        self.history = frame[:, self.config.hop_length :].clone()
        spectrum = torch.fft.rfft(frame * self.window, dim=-1)
        if self.model is not None:
            prediction, self.gru_state = self.model(spectrum[None, None, ...], self.gru_state)
            spectrum = prediction[0, 0]
        reconstructed = torch.fft.irfft(spectrum, n=self.config.n_fft, dim=-1) * self.window
        self.ola.add_(reconstructed)
        self.normalizer.add_(self.window_squared)
        hop = self.config.hop_length
        output = self.ola[:, :hop] / self.normalizer[:hop].clamp_min(1e-8)
        # Return an independent block; the next callback mutates the state.
        output = output.clone()
        self.ola = F.pad(self.ola[:, hop:], (0, hop))
        self.normalizer = F.pad(self.normalizer[hop:], (0, hop))
        self.input_hops += 1
        return output

    @torch.inference_mode()
    def flush(self) -> Tensor:
        if self.closed:
            return torch.empty(self.config.channels, 0, device=self.device, dtype=self.dtype)
        zeros = torch.zeros(self.config.channels, self.config.hop_length, device=self.device, dtype=self.dtype)
        output = torch.cat([self.push(zeros) for _ in range(self.config.flush_hops)], dim=-1)
        self.closed = True
        return output


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[CausalMaskGRU, CausalConfig, dict[str, Any]]:
    """Load project-format weights without silently using a random separator."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"trained checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format; train_causal.py creates format_version=1")
    config = CausalConfig(**checkpoint["config"])
    training = checkpoint.get("training", {})
    steps = int(training.get("optimizer_steps", 0))
    if steps <= 0:
        raise ValueError("checkpoint has no recorded optimizer steps; it cannot be presented as a trained separator")
    model = CausalMaskGRU(config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.trained_optimizer_steps = steps
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    metadata["checkpoint_path"] = str(path.resolve())
    model.checkpoint_metadata = metadata
    model.eval()
    return model, config, metadata


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@torch.inference_mode()
def benchmark_stream(
    waveform: Tensor,
    model: CausalMaskGRU,
    config: CausalConfig | None = None,
    *,
    device: str | torch.device = "cpu",
    warmup_hops: int = 4,
) -> tuple[Tensor, dict[str, Any]]:
    """Separate [C,L] input and return aligned [C,L] CPU output plus timing.

    Times include STFT, one-frame model inference and overlap-add, exclude file
    IO/model loading, and use the full H/fs deadline even for the padded final
    input block.  Flush times are recorded separately; whole-file RTF includes
    them.  Warm-up uses an independent processor so it cannot contaminate state.
    """
    config = model.config if config is None else config
    batch = _waveform_batch(waveform, config)
    if batch.shape[0] != 1:
        raise ValueError("stream benchmark processes one recording at a time")
    audio = batch[0].to(device=device, dtype=next(model.parameters()).dtype)
    if warmup_hops < 0:
        raise ValueError("warmup_hops must be nonnegative")
    device_obj = torch.device(device)

    def synchronize() -> None:
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)

    if warmup_hops:
        warmup = CausalStreamProcessor(model, config, device=device_obj)
        silence = torch.zeros(config.channels, config.hop_length, device=device_obj, dtype=audio.dtype)
        for _ in range(warmup_hops):
            warmup.push(silence)
        synchronize()
    processor = CausalStreamProcessor(model, config, device=device_obj)
    output_blocks: list[Tensor] = []
    times: list[float] = []
    length = audio.shape[-1]
    hop = config.hop_length
    for start in range(0, length, hop):
        block = audio[:, start : start + hop]
        if block.shape[-1] < hop:
            block = F.pad(block, (0, hop - block.shape[-1]))
        synchronize()
        started = time.perf_counter()
        output = processor.push(block)
        synchronize()
        times.append(time.perf_counter() - started)
        output_blocks.append(output)
    synchronize()
    started = time.perf_counter()
    tail = processor.flush()
    synchronize()
    flush_seconds = time.perf_counter() - started
    output_blocks.append(tail)
    raw = torch.cat(output_blocks, dim=-1)
    output = raw[:, config.delay_samples : config.delay_samples + length].cpu()
    deadline = hop / config.sample_rate
    ratios = [seconds / deadline for seconds in times]
    processing_seconds = sum(times)
    misses = sum(seconds > deadline for seconds in times)
    report: dict[str, Any] = {
        **config.latency_info(),
        "model_name": "educational_causal_stft_gru",
        "model_parameter_count": model.parameter_count,
        "trained_optimizer_steps": model.trained_optimizer_steps,
        "device": str(device_obj),
        "audio_samples": length,
        "audio_duration_seconds": length / config.sample_rate,
        "input_blocks": len(times),
        "padded_input_samples": math.ceil(length / hop) * hop - length,
        "warmup_hops": warmup_hops,
        "block_processing_seconds": times,
        "rtr_mean": sum(ratios) / len(ratios),
        "rtr_p95": _percentile(ratios, 0.95),
        "rtr_p99": _percentile(ratios, 0.99),
        "rtr_max": max(ratios),
        "deadline_misses": misses,
        "deadline_miss_fraction": misses / len(times),
        "processing_seconds": processing_seconds,
        "flush_seconds": flush_seconds,
        "flush_blocks": config.flush_hops,
        "whole_file_rtf_including_flush": (processing_seconds + flush_seconds) / (length / config.sample_rate),
        "timing_includes": ["causal_stft", "single_frame_gru", "overlap_add"],
        "timing_excludes": ["file_io", "checkpoint_loading", "warmup", "device_audio_buffers"],
        "final_block_uses_full_hop_deadline": True,
        "is_live_audio_device_measurement": False,
    }
    return output, report


@torch.inference_mode()
def separate_waveform(waveform: Tensor, model: CausalMaskGRU, *, device: str | torch.device = "cpu") -> Tensor:
    """Convenience wrapper with the same streaming path, returning [C,L]."""
    output, _ = benchmark_stream(waveform, model, device=device, warmup_hops=0)
    return output
