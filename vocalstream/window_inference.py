"""One BS-RoFormer forward per rolling context, without nested offline OLA.

This changes the inference policy, not the network or checkpoint. It is not
output-equivalent to the original overlap-add adapter and must be evaluated
alongside it on the same input/output schedule before claiming a speedup.
"""
from __future__ import annotations

import time

import torch

from .bsroformer import BSRoFormerBaseline


class BSRoFormerWindowSeparator(BSRoFormerBaseline):
    """Accept exactly one context and return its aligned vocal prediction.

    There is one batch-one network call, with the publisher's own STFT/ISTFT.
    There is no extra chunk padding, overlap averaging, TTA, or neural cache.
    The outer bounded-future adapter remains responsible for startup/EOF zeros
    and cropping the emitted samples. Context length must match chunk_samples.
    """

    window_inference_mode = "single"

    def __init__(self, device="cuda", precision="amp", chunk_seconds=2.0,
                 warmup=True, heads="vocals"):
        # Strict full-checkpoint loading, head selection, output checks and
        # synchronized processing/memory accounting are inherited unchanged.
        # Parent warmup dispatches through this class's _demix implementation.
        super().__init__(device=device, overlap=2, precision=precision,
                         tta=False, chunk_seconds=chunk_seconds,
                         warmup=warmup, heads=heads)
        self.overlap = 1
        self.step_samples = self.chunk_samples
        self.fade_samples = 0
        self.border_samples = 0

    def _demix(self, audio):
        if tuple(audio.shape) != (2, self.chunk_samples):
            raise ValueError(
                f"Single-window inference requires exactly {self.chunk_samples} "
                "stereo samples; choose an STFT-hop-aligned stream context")
        batch = audio.unsqueeze(0).to(self.device)
        self._sync()
        begin = time.perf_counter()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=self.precision == "amp"):
            prediction = self.model(batch)
        self._sync()
        forward_seconds = time.perf_counter() - begin
        vocals = self._vocal_prediction(prediction)[0].to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(vocals).all():
            raise ValueError("Non-finite model output; retry explicit fp32 if using amp")
        return vocals, 1, False, forward_seconds

    def metadata(self):
        result = super().metadata()
        result.update({
            "window_inference_mode": self.window_inference_mode,
            "overlap": 1, "step_samples": self.chunk_samples,
            "fade_samples": 0, "reflect_border_samples": 0,
            "nested_offline_overlap_add": False,
            "network_state_cached": False, "stft_state_cached": False,
            "output_equivalence_to_offline_ola_claimed": False,
            "processing_timing_scope": "one context; CPU preparation, transfers, one synchronized model forward and CPU waveform return; excludes load/warmup, file I/O and outer stream assembly",
            "latency_reason": "Bidirectional attention within the supplied window; waiting is determined by the outer packet schedule, with additional measured compute and virtual queue time",
        })
        return result
