"""Optional official museval v4 metric module, without its audio-format facade.

HQ is read by soundfile. museval.__init__ imports musdb/stempeg which demand
FFmpeg even for WAV; loading the installed pure metrics module avoids that
unrelated dependency. The upstream algorithm is not modified.
"""

import importlib.metadata
import importlib.util

import numpy as np


def evaluate_vocal_accompaniment(vocals, accompaniment, estimate, mixture, sample_rate):
    dist = importlib.metadata.distribution("museval")
    path = dist.locate_file("museval/metrics.py")
    spec = importlib.util.spec_from_file_location("vocalstream_museval_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    refs = np.stack([vocals, accompaniment]).astype(np.float64)
    ests = np.stack([estimate, mixture - estimate]).astype(np.float64)
    if any(np.sum(x * x) == 0 for x in refs) or any(np.sum(x * x) == 0 for x in ests):
        return {"status": "undefined", "reason": "A full reference/estimate source is silent"}
    scores = module.bss_eval(refs, ests, window=sample_rate, hop=sample_rate,
                            compute_permutation=False, framewise_filters=False,
                            bsseval_sources_version=False)
    out = {"status": "ok", "implementation": f"museval {dist.version}",
           "protocol": "BSSEval v4 images, vocals/accompaniment, 1s window/hop, track-constant 512-tap filters",
           "residual_estimate": "input mixture minus estimated vocals"}
    for name, values in zip(("sdr", "isr", "sir", "sar"), scores[:4]):
        vocal = values[0]
        usable = vocal[~np.isnan(vocal)]
        median = float(np.median(usable)) if usable.size else None
        out[f"vocals_{name}_median_db"] = median if median is not None and np.isfinite(median) else None
        out[f"vocals_{name}_frames_db"] = [float(v) if np.isfinite(v) else None for v in vocal]
        out[f"vocals_{name}_nonfinite_frames"] = int(np.sum(~np.isfinite(vocal)))
    return out
