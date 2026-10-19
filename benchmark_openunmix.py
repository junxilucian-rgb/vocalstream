"""Run Open-Unmix and save a reproducible JSON run report.

The reported wall time includes Python startup, model loading, inference, output
writing, and any first-run checkpoint download. It is intentionally distinct
from the model-only inference time printed by Open-Unmix.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
import wave
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SEPARATOR_SCRIPT = PROJECT_DIR / "run_openunmix_wav.py"


def wav_metadata(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
            raise ValueError(
                f"{path.name}: expected an uncompressed 16-bit PCM WAV file."
            )
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        return {
            "sample_rate_hz": sample_rate,
            "channels": wav_file.getnchannels(),
            "frames": frames,
            "duration_seconds": frames / sample_rate,
        }


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return slug or "openunmix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Open-Unmix on a 16-bit PCM WAV and save a JSON benchmark report. "
            "The total wall time includes process startup and model loading."
        )
    )
    parser.add_argument("input", type=Path, help="Input 16-bit PCM WAV file")
    parser.add_argument("--start", type=float, default=0.0, help="Crop start in seconds")
    parser.add_argument(
        "--duration", type=float, default=15.0, help="Crop duration in seconds"
    )
    parser.add_argument("--model", default="umxhq", help="Open-Unmix model name")
    parser.add_argument("--targets", default="vocals", help="Target stem(s)")
    parser.add_argument(
        "--residual", default="accompaniment", help="Residual stem name"
    )
    parser.add_argument("--niter", type=int, default=0, help="Wiener filter iterations")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="Inference device"
    )
    parser.add_argument(
        "--run-name", default=None, help="Optional short label included in output names"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if args.start < 0 or args.duration <= 0 or args.niter < 0:
        raise SystemExit("start must be >= 0, duration must be > 0, and niter must be >= 0.")

    source_info = wav_metadata(input_path)
    if args.start >= source_info["duration_seconds"]:
        raise SystemExit("Crop start must be shorter than the input audio duration.")
    crop_duration = min(
        args.duration, source_info["duration_seconds"] - args.start
    )

    started_at = datetime.now(timezone.utc)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    label = safe_slug(args.run_name or args.model)
    run_id = f"{timestamp}_{label}_niter{args.niter}_{uuid.uuid4().hex[:6]}"
    output_dir = PROJECT_DIR / "results" / "benchmark_runs" / run_id
    report_dir = PROJECT_DIR / "results" / "benchmark_reports"
    model_cache_dir = PROJECT_DIR / "model_cache" / "torch"
    output_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(SEPARATOR_SCRIPT),
        str(input_path),
        "--model",
        args.model,
        "--targets",
        args.targets,
        "--residual",
        args.residual,
        "--niter",
        str(args.niter),
        "--start",
        str(args.start),
        "--duration",
        str(crop_duration),
        "--outdir",
        str(output_dir),
    ]
    if args.device == "cpu":
        command.append("--no-cuda")

    wall_started = time.perf_counter()
    child_env = os.environ.copy()
    # Keep Torch's checkpoint cache inside the project so runs work in
    # restricted environments where the user's home directory is read-only.
    child_env["TORCH_HOME"] = str(model_cache_dir)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=child_env,
    )
    wall_seconds = time.perf_counter() - wall_started

    output_files = []
    for wav_path in sorted(output_dir.rglob("*.wav")):
        try:
            file_info = wav_metadata(wav_path)
        except (OSError, wave.Error, ValueError):
            file_info = None
        output_files.append(
            {
                "path": str(wav_path.relative_to(PROJECT_DIR)),
                "metadata": file_info,
            }
        )

    stdout_log = completed.stdout
    stderr_log = completed.stderr
    replacements = (
        (str(input_path), input_path.name),
        (input_path.as_posix(), input_path.name),
        (str(output_dir), str(output_dir.relative_to(PROJECT_DIR))),
        (output_dir.as_posix(), str(output_dir.relative_to(PROJECT_DIR))),
    )
    for sensitive_path, replacement in replacements:
        stdout_log = stdout_log.replace(sensitive_path, replacement)
        stderr_log = stderr_log.replace(sensitive_path, replacement)

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "status": "success" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "input": {
            "filename": input_path.name,
            "duration_seconds": source_info["duration_seconds"],
            "sample_rate_hz": source_info["sample_rate_hz"],
            "channels": source_info["channels"],
            "crop_start_seconds": args.start,
            "crop_duration_seconds": crop_duration,
        },
        "configuration": {
            "model": args.model,
            "targets": args.targets,
            "residual": args.residual,
            "niter": args.niter,
            "device": args.device,
        },
        "timing": {
            "wall_seconds": wall_seconds,
            "audio_seconds": crop_duration,
            "wall_rtf": wall_seconds / crop_duration,
            "scope": (
                "subprocess total: Python startup, model loading, inference, "
                "output writing, and any checkpoint download"
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "torch_cache": str(model_cache_dir.relative_to(PROJECT_DIR)),
            "packages": {
                "torch": package_version("torch"),
                "torchaudio": package_version("torchaudio"),
                "openunmix": package_version("openunmix"),
                "numpy": package_version("numpy"),
            },
        },
        "output_directory": str(output_dir.relative_to(PROJECT_DIR)),
        "output_files": output_files,
        "process_stdout": stdout_log.strip(),
        "process_stderr": stderr_log.strip(),
    }
    report_path = report_dir / f"{run_id}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    print(f"\nTotal wall time: {wall_seconds:.3f} s")
    print(f"Wall RTF: {wall_seconds / crop_duration:.4f}")
    print(f"Run report: {report_path}")
    if completed.returncode != 0:
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
