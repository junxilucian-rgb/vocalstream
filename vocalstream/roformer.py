"""Quality reference: pinned Mel-Band RoFormer, with offline overlap-add.

This adapter deliberately makes no streaming or algorithmic-latency claim.
The model consumes future samples in each eight-second input segment. Audio
I/O and checkpoint loading are excluded from ``separate`` timing; padding,
device transfers, all forward passes, overlap-add and optional TTA are reported
as total processing time. The returned seconds measure model forwards alone,
including the model's own STFT/ISTFT, to match the Open-Unmix model timer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


CHECKPOINT_SHA256 = "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"
CHECKPOINT_REVISION = "ac9b0614ab3cd7f77219e18ba494dfd93956c348"
CHECKPOINT_REPOSITORY = "KimberleyJSN/melbandroformer"
CONFIG_SHA256 = "87aabb300193b019159269b69d6fe5f313aae4201a22b4af6268bc2846ab2fe1"
CHUNK_SAMPLES = 352800
SAMPLE_RATE = 44100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _read_config(path: Path) -> dict:
    # The author config has a !!python/tuple sequence. Supporting only this
    # harmless tag keeps object instantiation / python execution disabled.
    class ConfigLoader(yaml.SafeLoader):
        pass

    ConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node, deep=True)),
    )
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=ConfigLoader)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Mel-Band RoFormer config must contain a model mapping")
    return config


def _vendor_model(project: Path):
    """Use the MIT architecture release matched to the checkpoint publisher."""
    version = importlib.metadata.version("bs-roformer")
    if version != "0.3.9":
        raise RuntimeError(f"Expected bs-roformer==0.3.9; found {version}")
    from bs_roformer import MelBandRoformer
    return MelBandRoformer, Path(inspect.getfile(MelBandRoformer))


def _state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a PyTorch state dictionary")
    if checkpoint and all(isinstance(key, str) and torch.is_tensor(value)
                          for key, value in checkpoint.items()):
        return checkpoint, "raw_state_dict"
    matches = [name for name in ("state_dict", "model_state_dict", "state")
               if name in checkpoint and isinstance(checkpoint[name], Mapping)]
    if len(matches) != 1:
        raise ValueError("Unrecognized or ambiguous checkpoint state dictionary")
    state = checkpoint[matches[0]]
    if not state or not all(isinstance(key, str) and torch.is_tensor(value)
                            for key, value in state.items()):
        raise ValueError("Checkpoint state dictionary contains unexpected values")
    return state, matches[0]


class MelRoFormerBaseline:
    """Pinned pretrained stereo vocal separator for offline quality comparison.

    Args:
        device: Explicit ``cpu`` or ``cuda`` device; unavailable CUDA raises.
        overlap: Number of eight-second chunks overlapping, default eight.
        tta: Average original, swapped-channel and inverted-polarity runs.
        precision: ``fp32`` or explicit CUDA float16 autocast (``amp``).
        model_dir: Directory containing ``MelBandRoformer.ckpt`` / ``config.yaml``.
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, device="cuda", overlap=8, tta=False, precision="fp32",
                 model_dir=None):
        if isinstance(overlap, bool) or not isinstance(overlap, int) or not 1 <= overlap <= 32:
            raise ValueError("overlap must be an integer between 1 and 32")
        if precision not in ("fp32", "amp"):
            raise ValueError("precision must be 'fp32' or 'amp'")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("This pinned baseline supports only cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment; select cpu explicitly")
        if precision == "amp" and self.device.type != "cuda":
            raise ValueError("amp means CUDA float16 autocast; choose fp32 on CPU")
        self.overlap = overlap
        self.tta = bool(tta)
        self.precision = precision
        self.chunk_samples = CHUNK_SAMPLES
        self.step_samples = self.chunk_samples // overlap
        self.fade_samples = self.chunk_samples // 10 if overlap > 1 else 0
        self.border_samples = self.chunk_samples - self.step_samples
        project = Path(__file__).resolve().parents[1]
        self.model_dir = Path(model_dir) if model_dir is not None else project / "model_cache" / "mel_roformer"
        checkpoint_path = self.model_dir / "MelBandRoformer.ckpt"
        config_path = self.model_dir / "config.yaml"
        if not checkpoint_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Missing pinned checkpoint or config in {self.model_dir}")
        begin = time.perf_counter()
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        if self.checkpoint_sha256 != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint SHA-256 does not match the pinned KimberleyJensen model; refusing to load")
        self.config_sha256 = _sha256(config_path)
        if self.config_sha256 != CONFIG_SHA256:
            raise ValueError("Configuration SHA-256 does not match the pinned checkpoint configuration")
        config = _read_config(config_path)
        model_config = dict(config["model"])
        if model_config.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE:
            raise ValueError("Pinned model must use 44100 Hz")
        if not model_config.get("stereo", True) or model_config.get("num_stems", 1) != 1:
            raise ValueError("Pinned model must be the single-stem stereo vocal model")
        configured_chunk = config.get("inference", {}).get("chunk_size", CHUNK_SAMPLES)
        if configured_chunk != CHUNK_SAMPLES:
            raise ValueError(f"Pinned model must use an eight-second {CHUNK_SAMPLES}-sample chunk")
        target = config.get("training", {}).get("target_instrument")
        if target not in (None, "vocals"):
            raise ValueError(f"Expected vocal target, found {target!r}")
        model_class, source = _vendor_model(project)
        self.vendor_source = str(source)
        self.vendor_sha256 = _sha256(source)
        self.model = model_class(**model_config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state, self.checkpoint_state_key = _state_dict(checkpoint)
        # A mismatch is a fatal configuration error. Do not skip, resize or
        # initialize unmatched layers, as that silently changes the model.
        self.model.load_state_dict(state, strict=True)
        del state, checkpoint
        self.model.eval().requires_grad_(False).to(self.device)
        self._sync()
        self.load_seconds = time.perf_counter() - begin
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.last_run = None

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _vocal_prediction(self, prediction):
        if not torch.is_tensor(prediction):
            raise ValueError("RoFormer returned a non-tensor prediction")
        if prediction.ndim == 4 and prediction.shape[1] == 1:
            prediction = prediction[:, 0]
        if tuple(prediction.shape) != (1, 2, self.chunk_samples):
            raise ValueError(f"Unexpected model output shape {tuple(prediction.shape)}")
        return prediction

    def _demix(self, audio: torch.Tensor):
        """MSST generic demix windowing, batch size one, preserving gain."""
        original_length = audio.shape[-1]
        reflected = original_length > 2 * self.border_samples and self.border_samples > 0
        if reflected:
            audio = F.pad(audio, (self.border_samples, self.border_samples), mode="reflect")
        length = audio.shape[-1]
        # Accumulate on CPU so long songs cannot fill GPU memory. Individual
        # segments still run on GPU, and transfers are included in the timer.
        result = torch.zeros((2, length), dtype=torch.float32)
        counter = torch.zeros(length, dtype=torch.float32)
        base_window = torch.ones(self.chunk_samples, dtype=torch.float32)
        if self.fade_samples:
            base_window[:self.fade_samples] = torch.linspace(0, 1, self.fade_samples)
            base_window[-self.fade_samples:] = torch.linspace(1, 0, self.fade_samples)
        forwards = 0
        network_seconds = 0.0
        for start in range(0, length, self.step_samples):
            part = audio[:, start:start + self.chunk_samples]
            segment_length = part.shape[-1]
            missing = self.chunk_samples - segment_length
            if missing:
                # Reflection requires padding < input length. This condition is
                # the author's generic demix safeguard for tail fragments.
                mode = "reflect" if segment_length > self.chunk_samples // 2 else "constant"
                part = F.pad(part, (0, missing), mode=mode)
            batch = part.unsqueeze(0).to(self.device)
            self._sync()
            network_begin = time.perf_counter()
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.precision == "amp"):
                prediction = self.model(batch)
            self._sync()
            network_seconds += time.perf_counter() - network_begin
            forwards += 1
            prediction = self._vocal_prediction(prediction)
            prediction = prediction[0, :, :segment_length].to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(prediction).all():
                raise ValueError("Non-finite model output; retry explicit fp32 if using amp")
            window = base_window.clone()
            if self.fade_samples and start == 0:
                window[:self.fade_samples] = 1
            elif self.fade_samples and start + self.step_samples >= length:
                window[-self.fade_samples:] = 1
            weights = window[:segment_length]
            result[:, start:start + segment_length] += prediction * weights
            counter[start:start + segment_length] += weights
        if not torch.all(counter > 0):
            raise RuntimeError("Overlap-add produced an uncovered output sample")
        result /= counter
        if reflected:
            result = result[:, self.border_samples:-self.border_samples]
        if result.shape != (2, original_length):
            raise RuntimeError("Internal overlap-add alignment error")
        return result, forwards, reflected, network_seconds

    def separate(self, audio):
        """Return aligned vocals and summed model-forward seconds, excluding transfers."""
        audio = np.asarray(audio)
        if audio.ndim != 2 or audio.shape[1] != 2 or audio.shape[0] == 0:
            raise ValueError("Input must be a nonempty stereo [samples, 2] array")
        if np.iscomplexobj(audio):
            raise ValueError("Input must be a real-valued audio waveform")
        if not np.isfinite(audio).all():
            raise ValueError("Input contains NaN or infinity")
        self._sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        begin = time.perf_counter()
        mixture = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))
        with torch.inference_mode():
            result, forwards, reflected, network_seconds = self._demix(mixture)
            if self.tta:
                swapped, count, _, elapsed = self._demix(mixture.flip(0))
                result += swapped.flip(0)
                forwards += count
                network_seconds += elapsed
                inverted, count, _, elapsed = self._demix(-mixture)
                result -= inverted
                forwards += count
                network_seconds += elapsed
                result /= 3
        vocals = result.numpy().T.copy()
        self._sync()
        seconds = time.perf_counter() - begin
        if not np.isfinite(vocals).all():
            raise ValueError("Non-finite separated output")
        self.last_run = {
            "samples": len(vocals), "duration_seconds": len(vocals) / self.sample_rate,
            "processing_seconds": seconds, "processing_rtf": seconds / (len(vocals) / self.sample_rate),
            "network_inference_seconds": network_seconds,
            "network_inference_rtf": network_seconds / (len(vocals) / self.sample_rate),
            "model_forward_passes": forwards, "tta_passes": 3 if self.tta else 1,
            "global_reflect_border_applied": reflected,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
            "output_peak": float(np.max(np.abs(vocals))),
            "gain_normalization_applied": False, "clipping_applied": False,
        }
        return vocals, network_seconds

    def metadata(self):
        return {
            "name": "Mel-Band RoFormer KimberleyJensen pretrained vocal quality reference",
            "architecture": "MelBandRoformer", "parameters": self.parameters,
            "sample_rate": self.sample_rate, "channels": 2, "target": "vocals",
            "device": str(self.device), "precision": self.precision,
            "chunk_samples": self.chunk_samples, "chunk_seconds": self.chunk_samples / self.sample_rate,
            "overlap": self.overlap, "step_samples": self.step_samples,
            "fade_samples": self.fade_samples, "reflect_border_samples": self.border_samples,
            "chunk_batch_size": 1, "tta": self.tta,
            "tta_augmentation": ["original", "channel_swap", "polarity_inversion"] if self.tta else ["original"],
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_state_key": self.checkpoint_state_key,
            "config_sha256": self.config_sha256,
            "vendor_source": self.vendor_source, "vendor_sha256": self.vendor_sha256,
            "architecture_package": "bs-roformer==0.3.9",
            "architecture_commit": "93a07dda7867d4acd8f5bd49ae3f33e1fcbfd8cf",
            "state_dict_loading": "weights_only=True; strict=True",
            "load_seconds": self.load_seconds, "warmup_applied": False,
            "streaming": False, "causal": False, "algorithmic_latency_ms": None,
            "latency_reason": "Offline bidirectional temporal attention uses future samples in an eight-second chunk",
            "timing_scope": "sum of synchronized model forwards across all chunks and TTA, including in-model STFT + neural mask + ISTFT; excludes loading, file I/O, padding, transfers and overlap-add",
            "processing_timing_scope": "all chunks and TTA, CPU preparation, padding, device transfers, neural inference and overlap-add; excludes checkpoint loading and file I/O",
            "last_run": self.last_run,
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32 if self.device.type == "cuda" else None,
        }
