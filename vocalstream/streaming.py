"""Bounded-future, file-simulated streaming around a synchronous separator.

This adapter changes the *inference schedule*, not the network architecture.
It recomputes an offline separator on rolling windows; it has no hidden full-file
input, recurrent neural state, audio-device clock, or real-time guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class StreamEmission:
    """Time-aligned output interval and its timing/dependency accounting."""

    audio: np.ndarray
    record: dict


class BoundedFutureSeparator:
    """Feed small stereo packets and emit consecutive, non-overlapping blocks.

    ``context_seconds = past_seconds + emit_seconds + lookahead_seconds``.
    Past context is cached input, so it is NOT counted as future waiting. A
    regular output interval [s, s+E) uses input only in [s-P, s+E+L), including
    zero padding before the stream starts. Its sample-boundary availability is
    s+E+L, plus any explicit input-packet delivery delay. At EOF, unavailable
    right context is zero and trailing outputs are released on the EOF event.

    The supplied separator must return ``(aligned_audio, forward_seconds)``;
    it must complete GPU work before returning. Existing project wrappers do
    this. Their optional ``_sync`` hook is also called around the wall timer.
    File I/O, checkpoint loading, and caller-controlled warmup are external.
    """

    def __init__(self, separator, *, sample_rate=44100, context_seconds=2.0,
                 emit_seconds=0.5, lookahead_seconds=0.25):
        if isinstance(sample_rate, bool) or int(sample_rate) != sample_rate or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        self.sample_rate = int(sample_rate)
        for name, value in (("context_seconds", context_seconds),
                            ("emit_seconds", emit_seconds),
                            ("lookahead_seconds", lookahead_seconds)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.context_samples = round(context_seconds * self.sample_rate)
        self.emit_samples = round(emit_seconds * self.sample_rate)
        self.lookahead_samples = round(lookahead_seconds * self.sample_rate)
        if self.emit_samples < 1 or self.context_samples < self.emit_samples + self.lookahead_samples:
            raise ValueError("context must contain at least one emit interval plus lookahead")
        self.past_samples = self.context_samples - self.emit_samples - self.lookahead_samples
        if getattr(separator, "sample_rate", self.sample_rate) != self.sample_rate:
            raise ValueError("Separator and stream sample rates must match")
        self.separator = separator
        self._buffer = np.empty((0, 2), dtype=np.float32)
        self._buffer_start = 0
        self._received = 0
        self._next_start = 0
        self._finished = False
        self._virtual_worker_finish = 0.0
        self._max_buffer_samples = 0
        self.records = []

    @property
    def next_input_frames(self):
        """Read at most this many samples to hit the next dependency boundary.

        Callers may supply smaller packets. Larger packets up to emit_samples
        are accepted, but their packetization wait is recorded separately.
        """
        if self._finished:
            return 0
        needed = self._next_start + self.emit_samples + self.lookahead_samples - self._received
        return max(1, min(self.emit_samples, needed))

    def push(self, block):
        """Accept ONLY the next input packet; no full-file future is accepted."""
        if self._finished:
            raise RuntimeError("Cannot push after finish")
        block = np.asarray(block)
        if (block.ndim != 2 or block.shape[1] != 2 or not 0 < len(block) <= self.emit_samples
                or np.iscomplexobj(block) or not np.isfinite(block).all()):
            raise ValueError("Input must be finite real stereo, 1..emit_samples frames per push")
        self._buffer = np.concatenate((self._buffer, np.asarray(block, dtype=np.float32)))
        self._received += len(block)
        self._max_buffer_samples = max(self._max_buffer_samples, len(self._buffer))
        emissions = []
        while self._received >= self._next_start + self.emit_samples + self.lookahead_samples:
            emissions.append(self._emit(eof=False))
        return emissions

    def finish(self):
        """Signal that no more real input exists; flush with explicit right zeros."""
        if self._finished:
            raise RuntimeError("finish must be called exactly once")
        self._finished = True
        emissions = []
        while self._next_start < self._received:
            emissions.append(self._emit(eof=True))
        return emissions

    def _emit(self, *, eof):
        start = self._next_start
        count = min(self.emit_samples, self._received - start) if eof else self.emit_samples
        window_start = start - self.past_samples
        window_end = start + self.emit_samples + self.lookahead_samples
        required_input = min(window_end, self._received) if eof else window_end
        actual_available = self._received
        sync = getattr(self.separator, "_sync", None)
        if sync is not None:
            sync()
        begin = time.perf_counter()
        window = np.zeros((self.context_samples, 2), dtype=np.float32)
        real_begin, real_end = max(0, window_start), min(self._received, window_end)
        if real_begin < self._buffer_start or real_end > self._buffer_start + len(self._buffer):
            raise RuntimeError("Rolling input buffer does not cover required real context")
        window[real_begin - window_start:real_end - window_start] = self._buffer[
            real_begin - self._buffer_start:real_end - self._buffer_start]
        prediction, network_seconds = self.separator.separate(window)
        prediction = np.asarray(prediction)
        if (prediction.shape != window.shape or np.iscomplexobj(prediction)
                or not np.isfinite(prediction).all()):
            raise ValueError("Separator must return a finite, aligned stereo context waveform")
        output = np.array(prediction[self.past_samples:self.past_samples + count], dtype=np.float32, copy=True)
        if sync is not None:
            sync()
        wall_seconds = time.perf_counter() - begin
        if not np.isfinite(network_seconds) or network_seconds < 0:
            raise ValueError("Separator forward time must be finite and nonnegative")

        # These are virtual audio-clock times. No sleeping, soundcard callback,
        # file read, or output write is represented as processing work here.
        available_seconds = actual_available / self.sample_rate
        worker_start = max(available_seconds, self._virtual_worker_finish)
        worker_finish = worker_start + wall_seconds
        period_seconds = self.emit_samples / self.sample_rate
        target_end = start + count
        record = {
            "emission_index": len(self.records),
            "output_start_sample": start, "output_end_sample_exclusive": target_end,
            "output_samples": count, "output_audio_seconds": count / self.sample_rate,
            "context_start_sample": window_start, "context_end_sample_exclusive": window_end,
            "real_context_start_sample": real_begin, "real_context_end_sample_exclusive": real_end,
            "left_zero_padding_samples": max(0, -window_start),
            "right_zero_padding_samples": max(0, window_end - self._received),
            "required_real_input_sample_boundary": required_input,
            "input_received_sample_boundary": actual_available,
            "packetization_wait_samples": actual_available - required_input,
            "eof_flush": eof, "full_emission": count == self.emit_samples,
            "availability_delay_from_first_output_boundary_seconds": (required_input - start) / self.sample_rate,
            "availability_delay_from_output_end_boundary_seconds": (required_input - target_end) / self.sample_rate,
            "processing_seconds": wall_seconds,
            "network_forward_seconds": float(network_seconds),
            "processing_rtf_vs_emitted_audio": wall_seconds / (count / self.sample_rate),
            "nominal_compute_budget_seconds": period_seconds,
            "compute_exceeds_nominal_emit_period": wall_seconds > period_seconds,
            "compute_exceeds_output_duration": wall_seconds > count / self.sample_rate,
            "virtual_input_release_seconds": available_seconds,
            "virtual_worker_start_seconds": worker_start,
            "virtual_worker_finish_seconds": worker_finish,
            "virtual_queue_wait_seconds": worker_start - available_seconds,
            "virtual_deadline_seconds": available_seconds + period_seconds,
            "virtual_deadline_missed": worker_finish > available_seconds + period_seconds,
            "virtual_ready_delay_from_first_output_boundary_seconds": worker_finish - start / self.sample_rate,
            "virtual_ready_delay_from_output_end_boundary_seconds": worker_finish - target_end / self.sample_rate,
        }
        self._virtual_worker_finish = worker_finish
        self.records.append(record)
        self._next_start = target_end
        keep_from = max(0, self._next_start - self.past_samples)
        discard = min(len(self._buffer), max(0, keep_from - self._buffer_start))
        self._buffer = self._buffer[discard:].copy()
        self._buffer_start += discard
        return StreamEmission(output, dict(record))

    def summary(self):
        """JSON-safe observations; warmup, device latency and causality are not inferred."""
        def describe(records):
            if not records:
                return {"emission_count": 0}
            processing = np.array([row["processing_seconds"] for row in records])
            duration = sum(row["output_audio_seconds"] for row in records)
            delay = np.array([row["virtual_ready_delay_from_first_output_boundary_seconds"] for row in records])
            return {
                "emission_count": len(records),
                "total_processing_seconds": float(processing.sum()),
                "total_emitted_audio_seconds": duration,
                "total_processing_rtf": float(processing.sum() / duration),
                "processing_seconds_median": float(np.median(processing)),
                "processing_seconds_p95": float(np.percentile(processing, 95)),
                "processing_seconds_max": float(processing.max()),
                "compute_budget_miss_count": sum(row["compute_exceeds_nominal_emit_period"] for row in records),
                "compute_budget_miss_fraction": sum(row["compute_exceeds_nominal_emit_period"] for row in records) / len(records),
                "virtual_deadline_miss_count": sum(row["virtual_deadline_missed"] for row in records),
                "virtual_deadline_miss_fraction": sum(row["virtual_deadline_missed"] for row in records) / len(records),
                "virtual_queue_wait_seconds_max": max(row["virtual_queue_wait_seconds"] for row in records),
                "virtual_first_boundary_ready_delay_seconds_p95": float(np.percentile(delay, 95)),
            }

        regular = [row for row in self.records if not row["eof_flush"]]
        # Left-padded startup and right-padded EOF observations are not silently
        # mixed into the full-real-context, steady-state summary.
        steady = [row for row in regular if row["left_zero_padding_samples"] == 0]
        return {
            "protocol": "bounded_future_rolling_offline_recompute_v1",
            "sample_rate": self.sample_rate, "channels": 2,
            "context_samples": self.context_samples, "past_samples": self.past_samples,
            "emit_samples": self.emit_samples, "lookahead_samples": self.lookahead_samples,
            "context_seconds": self.context_samples / self.sample_rate,
            "cached_past_seconds": self.past_samples / self.sample_rate,
            "emit_seconds": self.emit_samples / self.sample_rate,
            "lookahead_seconds": self.lookahead_samples / self.sample_rate,
            "regular_first_boundary_availability_delay_seconds": (self.emit_samples + self.lookahead_samples) / self.sample_rate,
            "regular_output_end_boundary_availability_delay_seconds": self.lookahead_samples / self.sample_rate,
            "input_samples": self._received, "output_samples": self._next_start,
            "finished": self._finished, "max_retained_input_buffer_samples": self._max_buffer_samples,
            "rolling_window_padding": "zero before start and after explicit EOF; internal separator padding may also apply",
            "output_overlap_crossfade": False,
            "timing_scope": "rolling window creation, synchronized separator call and output crop; excludes input reads/push copy, model load/warmup, sink writes and sleep",
            "virtual_scheduler": "single serial worker; audio input released at packet sample boundary / fs; finish=max(input_release, previous_finish)+measured_compute",
            "deadline_definition": "available packet audio-clock time + nominal emit period; EOF flush is a separate burst and is excluded from steady-state",
            "availability_definition": "sample interval boundaries; first sample is index output_start; final sample precedes output_end boundary by 1/fs",
            "network_architecture_causal": False,
            "hardware_end_to_end_latency_measured": False,
            "audio_device_real_time_test": False,
            "no_real_time_sleep": True,
            "all_emissions": describe(self.records),
            "regular_emissions": describe(regular),
            "full_real_context_steady_state_emissions": describe(steady),
        }
