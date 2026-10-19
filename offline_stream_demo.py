"""Short, dependency-bounded BS-RoFormer file-stream experiment (not live audio)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import soundfile as sf

from vocalstream.streaming import BoundedFutureSeparator


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference", type=Path, help="Optional aligned vocal WAV, scored only after streaming finishes")
    parser.add_argument("--output", required=True, type=Path, help="A NEW output directory")
    parser.add_argument("--start", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--emit-seconds", type=float, default=0.5)
    parser.add_argument("--lookahead-seconds", type=float, default=0.25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "amp"), default="amp")
    parser.add_argument("--heads", choices=("all", "vocals"), default="vocals")
    args = parser.parse_args()
    if not math.isfinite(args.start) or args.start < 0 or not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("start must be finite >=0; duration must be finite >0")
    if not 0.25 <= args.context_seconds <= 11:
        parser.error("The pinned BS-RoFormer wrapper supports context-seconds in [0.25, 11]")
    if (not math.isfinite(args.emit_seconds) or args.emit_seconds <= 0
            or not math.isfinite(args.lookahead_seconds) or args.lookahead_seconds < 0):
        parser.error("emit-seconds must be finite >0; lookahead-seconds must be finite >=0")
    if (round(args.emit_seconds * 44100) < 1
            or round(args.context_seconds * 44100) < round(args.emit_seconds * 44100) + round(args.lookahead_seconds * 44100)):
        parser.error("At sample resolution, context must contain at least one emit interval plus lookahead")
    if args.device == "cpu" and args.precision == "amp":
        parser.error("Use --precision fp32 with --device cpu")
    project = Path(__file__).resolve().parent
    source_files = [Path(__file__), project / "vocalstream/streaming.py",
                    project / "vocalstream/bsroformer.py", project / "vocalstream/roformer.py",
                    project / "vocalstream/metrics.py"]
    source_files.extend(sorted((project / "vocalstream/vendor/zfturbo_bsroformer").glob("*.py")))
    source_hashes = {str(p.relative_to(project)): sha256(p) for p in source_files}
    # No implicit dataset traversal; only the explicitly supplied file is read.
    input_path = args.input.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve() if args.reference else None
    with sf.SoundFile(input_path) as source:
        if source.samplerate != 44100 or source.channels != 2:
            parser.error("Input must be stereo 44100 Hz; resample explicitly before this experiment")
        rate = source.samplerate
        start_sample = round(args.start * rate)
        requested_samples = round(args.duration * rate)
        if not 0 <= start_sample < source.frames or requested_samples < 1:
            parser.error("Selected input segment is empty or begins after EOF")
        actual_samples = min(requested_samples, source.frames - start_sample)
        if reference_path:
            ref_info = sf.info(reference_path)
            if (ref_info.samplerate, ref_info.channels, ref_info.frames) != (rate, 2, source.frames):
                parser.error("Reference and input must have identical full-file rate, channels and frame count")
        if args.output.exists():
            parser.error("Output already exists; select a new directory to preserve earlier evidence")

        import torch
        from vocalstream.bsroformer import BSRoFormerBaseline

        seed = 2026
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        separator = BSRoFormerBaseline(device=args.device, precision=args.precision,
                                      overlap=2, heads=args.heads,
                                      chunk_seconds=args.context_seconds, warmup=True)
        stream = BoundedFutureSeparator(separator, sample_rate=rate,
                                        context_seconds=args.context_seconds,
                                        emit_seconds=args.emit_seconds,
                                        lookahead_seconds=args.lookahead_seconds)
        args.output.mkdir(parents=True, exist_ok=False)
        output_path = args.output / "vocals.wav"
        source.seek(start_sample)
        remaining = actual_samples
        with sf.SoundFile(output_path, mode="w", samplerate=rate, channels=2, subtype="FLOAT") as sink:
            while remaining:
                block = source.read(min(remaining, stream.next_input_frames), dtype="float32", always_2d=True)
                if not len(block):
                    raise RuntimeError("Unexpected EOF while reading the selected segment")
                remaining -= len(block)
                for emission in stream.push(block):
                    sink.write(emission.audio)
            for emission in stream.finish():
                sink.write(emission.audio)

    summary = stream.summary()
    if summary["input_samples"] != actual_samples or summary["output_samples"] != actual_samples:
        raise RuntimeError("Stream output length differs from the selected input segment")
    quality = None
    steady_quality = None
    steady_intervals = [
        [row["output_start_sample"], row["output_end_sample_exclusive"]]
        for row in stream.records
        if not row["eof_flush"] and row["left_zero_padding_samples"] == 0
    ]
    if reference_path:
        from vocalstream.metrics import separation_metrics
        reference, _ = sf.read(reference_path, start=start_sample, frames=actual_samples,
                               dtype="float32", always_2d=True)
        mixture, _ = sf.read(input_path, start=start_sample, frames=actual_samples,
                             dtype="float32", always_2d=True)
        vocals, _ = sf.read(output_path, dtype="float32", always_2d=True)
        quality = separation_metrics(reference, vocals, mixture)
        if steady_intervals:
            def concatenate_intervals(audio):
                return np.concatenate([audio[begin:end] for begin, end in steady_intervals], axis=0)
            steady_quality = separation_metrics(concatenate_intervals(reference),
                                                concatenate_intervals(vocals),
                                                concatenate_intervals(mixture))
    if {str(p.relative_to(project)): sha256(p) for p in source_files} != source_hashes:
        raise RuntimeError("Source files changed during the experiment; output cannot be treated as frozen evidence")
    report = {
        "status": "complete",
        "experiment_kind": "short_file_simulated_streaming_demo_not_formal_test",
        "input_path": str(input_path), "input_sha256": sha256(input_path),
        "reference_path": str(reference_path) if reference_path else None,
        "reference_sha256": sha256(reference_path) if reference_path else None,
        "output_path": str(output_path.resolve()), "output_sha256": sha256(output_path),
        "clip_start_sample": start_sample, "clip_start_seconds": start_sample / rate,
        "clip_samples": actual_samples, "clip_seconds": actual_samples / rate,
        "clip_end_is_explicit_stream_eof": True,
        "source_files_sha256": source_hashes,
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "random_seed": seed, "seed_scope": "numpy.random.seed and torch.manual_seed; deterministic kernels not enforced",
        "device_name": torch.cuda.get_device_name(separator.device) if separator.device.type == "cuda" else platform.processor(),
        "model": separator.metadata(), "stream": summary,
        "quality_whole_clip_includes_startup_and_eof": quality,
        "quality_full_real_context_steady_state": steady_quality,
        "steady_state_quality_intervals_clip_relative_samples": steady_intervals,
        "steady_state_quality_interval_convention": "start inclusive, end exclusive; concatenate non-EOF emissions with zero left padding; a shared interval across different configurations is not enforced",
        "steady_state_quality_undefined_reason": ("reference_not_supplied" if not reference_path else
                                                   "no_full_real_context_emissions" if not steady_intervals else None),
        "emissions": stream.records,
        "limitations": [
            "No audio hardware, callback, driver or input/output device latency was measured.",
            "Offline model is recomputed for each emitted block; it is not converted to a causal architecture.",
            "Non-overlapping output crops have no crossfade; listening must check boundary artifacts.",
            "The selected clip starts with zero history and ends with explicit EOF, not real song continuation.",
            "Per-emission times are one short run, not a repeated timing benchmark.",
            "SI-SDR is the project whole-segment metric, not the original paper SDR protocol.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "summary": summary, "quality": quality}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
