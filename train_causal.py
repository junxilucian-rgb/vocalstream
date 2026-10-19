"""Train the project's small causal GRU baseline on MUSDB18-HQ train stems.

Example (PowerShell, from the project directory):
  python train_causal.py --root data/musdb18-hq --steps 200 --output results/causal_pilot

This script deliberately reads root/train only.  Full benchmark evaluation must
use held-out root/test recordings through the separate evaluation command.
Training a pilot on a few tracks is a learning exercise, not a paper result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

# cuBLAS requires this before the first CUDA context is initialized when
# deterministic algorithms are enabled. Preserve an explicit user setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from vocalstream.causal import CausalConfig, CausalMaskGRU, analysis_frames
from vocalstream.data import dataset_info


@dataclass(frozen=True)
class TrainTrack:
    name: str
    mixture: Path
    vocals: Path
    frames: int
    sample_rate: int
    channels: int


def discover_train_tracks(root: Path, config: CausalConfig) -> list[TrainTrack]:
    train_root = root.resolve() / "train"
    if not train_root.is_dir():
        raise ValueError(f"HQ training folder not found: {train_root}")
    tracks = []
    for folder in sorted(train_root.iterdir(), key=lambda item: item.name):
        if not folder.is_dir():
            continue
        mixture, vocals = folder / "mixture.wav", folder / "vocals.wav"
        if not mixture.is_file() or not vocals.is_file():
            continue
        mix_info, vocal_info = sf.info(mixture), sf.info(vocals)
        if (mix_info.frames, mix_info.samplerate, mix_info.channels) != (vocal_info.frames, vocal_info.samplerate, vocal_info.channels):
            raise ValueError(f"mixture/vocals are not time-aligned with identical format: {folder.name}")
        if mix_info.samplerate != config.sample_rate or mix_info.channels != config.channels:
            raise ValueError(f"{folder.name}: expected {config.sample_rate} Hz/{config.channels} channels, got {mix_info.samplerate}/{mix_info.channels}")
        if mix_info.frames <= 0:
            raise ValueError(f"empty recording: {folder.name}")
        tracks.append(TrainTrack(folder.name, mixture, vocals, mix_info.frames, mix_info.samplerate, mix_info.channels))
    if len(tracks) < 2:
        raise ValueError("need at least two HQ train tracks for a song-level training/validation split")
    return tracks


def _official_validation_names() -> list[str]:
    """Read package metadata without importing musdb/stempeg/FFmpeg."""
    try:
        package = importlib.metadata.distribution("musdb")
        config_path = Path(package.locate_file("musdb/configs/mus.yaml"))
        import yaml

        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        names = value.get("validation_tracks", [])
        return [str(name) for name in names]
    except (importlib.metadata.PackageNotFoundError, ImportError, OSError, AttributeError):
        return []


def split_tracks(
    tracks: list[TrainTrack], validation_names: list[str] | None, seed: int
) -> tuple[list[TrainTrack], list[TrainTrack], str]:
    available = {track.name for track in tracks}
    if validation_names is not None:
        missing = sorted(set(validation_names) - available)
        if missing:
            raise ValueError(f"requested validation tracks are not in root/train: {missing}")
        selected = set(validation_names)
        source = "explicit_song_names"
    else:
        selected = set(_official_validation_names()) & available
        source = "musdb_official_validation_tracks_present"
        if not selected or selected == available:
            names = sorted(available)
            random.Random(seed).shuffle(names)
            count = max(1, min(len(names) - 1, round(len(names) * 0.14)))
            selected = set(names[:count])
            source = "seeded_song_level_holdout_for_partial_dataset"
    train = [track for track in tracks if track.name not in selected]
    validation = [track for track in tracks if track.name in selected]
    if not train or not validation:
        raise ValueError("training and validation must both contain at least one different song")
    return train, validation, source


def read_crop(track: TrainTrack, start: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    stop = min(start + samples, track.frames)
    mixture, _ = sf.read(track.mixture, start=start, stop=stop, dtype="float32", always_2d=True)
    vocals, _ = sf.read(track.vocals, start=start, stop=stop, dtype="float32", always_2d=True)
    if not np.isfinite(mixture).all() or not np.isfinite(vocals).all():
        raise ValueError(f"non-finite audio in {track.name}")
    mix = torch.from_numpy(mixture.T.copy())
    reference = torch.from_numpy(vocals.T.copy())
    padding = samples - mix.shape[-1]
    if padding:
        mix = F.pad(mix, (0, padding))
        reference = F.pad(reference, (0, padding))
    return mix, reference


def spectral_loss(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    # Phase is inherited from the mixture.  This magnitude objective cannot
    # independently correct phase or completely undo cancellation in the mix.
    return F.l1_loss(torch.log1p(prediction.abs()), torch.log1p(reference.abs()))


@torch.inference_mode()
def validate(model: CausalMaskGRU, tracks: list[TrainTrack], crop_samples: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for track in tracks:
        # Fixed central crop: every validation uses the same unseen song segment.
        start = max(0, (track.frames - crop_samples) // 2)
        mix, target = read_crop(track, start, crop_samples)
        mix_spectrum = analysis_frames(mix.to(device), model.config)
        target_spectrum = analysis_frames(target.to(device), model.config)
        prediction, _ = model(mix_spectrum)
        losses.append(float(spectral_loss(prediction, target_spectrum).cpu()))
    return sum(losses) / len(losses)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0 or args.batch_size <= 0 or args.crop_seconds <= 0 or args.validate_every <= 0:
        raise ValueError("steps, batch-size, crop-seconds and validate-every must be positive")
    if args.learning_rate <= 0 or args.threads <= 0:
        raise ValueError("learning-rate and threads must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this PyTorch environment; use --device cpu")
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    config = CausalConfig(sample_rate=args.sample_rate, n_fft=args.n_fft, hop_length=args.hop_length, hidden_size=args.hidden_size)
    tracks = discover_train_tracks(Path(args.root), config)
    training_tracks, validation_tracks, split_source = split_tracks(tracks, args.validation_tracks, args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best.pt", "last.pt", "training_log.json")):
        raise ValueError("output already contains a training run; choose a new --output directory")
    crop_samples = max(config.n_fft, round(args.crop_seconds * config.sample_rate))
    model = CausalMaskGRU(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)
    started = time.perf_counter()
    history: list[dict[str, int | float]] = []
    initial_validation = validate(model, validation_tracks, crop_samples, device)
    best_validation = float("inf")
    best_step = 0
    training_info: dict[str, Any] = {
        "dataset_metadata": dataset_info(args.root),
        "seed": args.seed,
        "optimizer_steps": 0,
        "root": str(Path(args.root).resolve()),
        "synthetic_engineering_fixture": (Path(args.root).resolve() / "SYNTHETIC_ONLY.json").is_file(),
        "training_tracks": [track.name for track in training_tracks],
        "validation_tracks": [track.name for track in validation_tracks],
        "split_source": split_source,
        "test_tracks_accessed": False,
        "crop_samples": crop_samples,
        "crop_seconds": crop_samples / config.sample_rate,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss": "mean_absolute_difference_of_log1p_STFT_magnitudes",
        "device": str(device),
        "torch_threads": args.threads,
        "torch_version": str(torch.__version__),
        "python_version": sys.version.split()[0],
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "validation_crop_rule": "fixed central crop per validation song",
        "recurrent_training_state_rule": "GRU state resets for each randomly sampled crop; streaming inference carries state continuously",
        "context_training_limitation": "crop-reset training is an educational baseline; continuous-state or context-warmed training is a future ablation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "initial_validation_loss": initial_validation,
        "educational_baseline_not_full_paper_reproduction": True,
    }
    print(f"Train songs: {len(training_tracks)}; validation songs: {len(validation_tracks)}; parameters: {model.parameter_count:,}", flush=True)
    print(f"Delay convention: {config.delay_samples} samples; first original sample ready after {config.n_fft} collected samples.", flush=True)

    def save_checkpoint(filename: str) -> None:
        weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        state = {
            "format_version": 1,
            "model_name": "educational_causal_stft_gru",
            "config": config.to_dict(),
            "state_dict": weights,
            "training": dict(training_info),
        }
        temporary = output / (filename + ".tmp")
        torch.save(state, temporary)
        temporary.replace(output / filename)

    for step in range(1, args.steps + 1):
        model.train()
        inputs, references = [], []
        for _ in range(args.batch_size):
            track = rng.choice(training_tracks)
            start = rng.randrange(max(0, track.frames - crop_samples) + 1)
            mix, reference = read_crop(track, start, crop_samples)
            inputs.append(mix)
            references.append(reference)
        mix_spectrum = analysis_frames(torch.stack(inputs).to(device), config)
        reference_spectrum = analysis_frames(torch.stack(references).to(device), config)
        prediction, _ = model(mix_spectrum)
        loss = spectral_loss(prediction, reference_spectrum)
        if not torch.isfinite(loss):
            raise ValueError("training loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        model.trained_optimizer_steps = step
        training_info["optimizer_steps"] = step
        row: dict[str, int | float] = {"step": step, "train_spectral_loss": float(loss.detach().cpu())}
        if step % args.validate_every == 0 or step == args.steps:
            score = validate(model, validation_tracks, crop_samples, device)
            row["validation_spectral_loss"] = score
            training_info["last_validation_loss"] = score
            if score < best_validation:
                best_validation, best_step = score, step
                training_info["best_validation_loss"] = best_validation
                training_info["best_step"] = best_step
                save_checkpoint("best.pt")
            print(f"step {step}/{args.steps}: train loss {row['train_spectral_loss']:.6f}; validation loss {score:.6f}", flush=True)
        history.append(row)
        if step % args.validate_every == 0 or step == args.steps:
            training_info["elapsed_seconds"] = time.perf_counter() - started
            _atomic_json(output / "training_log.json", {"config": config.to_dict(), "training": dict(training_info), "history": history})
    training_info["elapsed_seconds"] = time.perf_counter() - started
    save_checkpoint("last.pt")
    report = {
        "config": config.to_dict(),
        "training": training_info,
        "history": history,
        "best_checkpoint": str(output / "best.pt"),
        "last_checkpoint": str(output / "last.pt"),
        "latency": config.latency_info(),
    }
    _atomic_json(output / "training_log.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an educational causal STFT-GRU vocal separator on MUSDB18-HQ train stems.")
    parser.add_argument("--root", required=True, help="MUSDB18-HQ root containing train/<song>/mixture.wav and vocals.wav")
    parser.add_argument("--output", default="results/causal_pilot", help="new directory for checkpoint and complete training log")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-seconds", type=float, default=2.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="fixed CPU threads; one often helps short-hop inference/training")
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--validation-tracks", nargs="+", help="exact song names; default uses available official validation names, otherwise a seeded song-level holdout")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = train(args)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(f"Saved: {report['best_checkpoint']}")
    print("This pilot does not establish music separation quality or ultra-low-latency deployment performance.")


if __name__ == "__main__":
    main()
