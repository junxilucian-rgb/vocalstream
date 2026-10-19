"""Load an offline baseline once. File I/O and loading are timed separately."""

import os
import time
from pathlib import Path

import numpy as np
import torch


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class OpenUnmixBaseline:
    def __init__(self, model="umxhq", niter=0, device="cpu", warmup=True):
        from openunmix import utils
        if model not in ("umx", "umxhq", "umxl"):
            raise ValueError("Unknown Open-Unmix model")
        if niter < 0:
            raise ValueError("niter must be >= 0")
        self.device = torch.device(device)
        self.model_name = model
        self.niter = niter
        project = Path(__file__).resolve().parents[1]
        os.environ["TORCH_HOME"] = str(project / "model_cache" / "torch")
        begin = time.perf_counter()
        self.separator = utils.load_separator(model, targets=["vocals"], residual=True,
                                              niter=niter, device=self.device)
        self.separator.freeze()
        self.separator.to(self.device)
        synchronize(self.device)
        self.load_seconds = time.perf_counter() - begin
        self.sample_rate = int(self.separator.sample_rate.item())
        self.parameters = sum(p.numel() for p in self.separator.parameters())
        self.warmup_seconds = 0.0
        if warmup:
            # Warmup is explicit and not added to measured track times.
            audio = np.random.default_rng(0).normal(0, 0.001, (self.sample_rate, 2)).astype(np.float32)
            _, self.warmup_seconds = self.separate(audio)

    def separate(self, audio):
        # CPU->device transfer is outside model timer and separately documented.
        tensor = torch.from_numpy(np.ascontiguousarray(audio.T))[None].to(self.device)
        synchronize(self.device)
        begin = time.perf_counter()
        with torch.inference_mode():
            estimates = self.separator(tensor)
        synchronize(self.device)
        seconds = time.perf_counter() - begin
        outputs = self.separator.to_dict(estimates)
        vocals = outputs["vocals"][0].cpu().numpy().T.copy()
        return vocals, seconds

    def metadata(self):
        return {"name": self.model_name, "niter": self.niter,
                "parameters": self.parameters, "sample_rate": self.sample_rate,
                "load_seconds": self.load_seconds, "warmup_seconds": self.warmup_seconds,
                "streaming": False, "algorithmic_latency_ms": None,
                "latency_reason": "Offline bidirectional LSTM uses future frames; no causal streaming latency claim",
                "timing_scope": "STFT + separator + Wiener + ISTFT; excludes load, file I/O and device transfers"}
