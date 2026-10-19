# Methodology and interpretation

## Tasks

The main task estimates vocals from a stereo music mixture. MUSDB18-HQ supplies time-aligned mixture and vocal reference files. Standard separation uses the official `mixture.wav`, rather than rebuilding it from stems.

Controlled bleed uses `vocals + gain × (drums + bass + other)` and keeps the original vocal stem as its target. The −30/−20/−10 dB settings are accompaniment amplitude gains, not fixed input SNRs. Actual reference/interference ratios vary by song. Original vocal stems can contain production effects; they are not guaranteed dry close-microphone recordings.

Noise, artificial room response and educational causal-model utilities are available as research extensions. The recorded main comparisons are the published excerpt protocols in [RESULTS.md](RESULTS.md).

## Model provenance and head selection

The active reference is the ZFTurbo MUSDB18-HQ community **BS-RoFormer** checkpoint, with 11 s trained input context, FFT 2048, hop 441, and original output order `drums, bass, other, vocals`. Source and asset hashes are pinned. It is a different checkpoint from the original paper, and a different model family from the optional Mel-Band RoFormer adapter.

The four-output checkpoint is loaded with `weights_only=True` and `strict=True`. Vocal-only inference retains the original mask estimator at index 3 after loading. Shared band splitting, Transformers, final normalization, vocal weights and waveform alignment are unchanged. The other three independent mask estimators and their inverse STFT reconstructions are skipped. The on-disk checkpoint and its initial strict-loading step remain unchanged.

Numerical equality is measured rather than inferred from the architecture. Different ISTFT batch shapes could affect implementations on other platforms. Same-precision tolerances were declared before the head study: waveform NRMSE ≤ `1e-5` and absolute per-song SI-SDR change ≤ `0.001 dB`. Cross-precision AMP comparisons are reported separately and do not inherit these thresholds.

## Data partition and scope

The 14 official MUSDB validation names are stored in `project_config.json`. Development scripts select them only from the dataset's `train` folder. The recorded head/quality studies use the fixed 60–90 s excerpt of every song. The stream study uses three fixed songs at 60–64 s. Repeated timing or additional settings do not create additional independent song samples.

The repository contains a separate whole-song validation → freeze → final-test workflow. Those full-song stages have not produced the published results. In addition, the external checkpoint's training-song list is unknown; freezing this project's choices cannot retroactively establish pretraining independence.

## Quality metrics

For SI-SDR, first subtract each channel's whole-segment mean, then flatten both channels. With target `s` and estimate `y`:

```text
alpha = dot(y, s) / dot(s, s)
SI-SDR = 10 log10(||alpha*s||² / ||y-alpha*s||²)
SI-SDRi = SI-SDR(output, target) - SI-SDR(input, target)
```

There is one shared projection gain across stereo channels. There is no time-shift search or reference-filter fitting. Silent references/estimates produce an undefined value with a reason. Extreme numerical ratios are capped at ±120 dB. This metric is not the same protocol as museval BSSEval SDR.

Waveform preservation uses the unscaled ratio `||estimate-reference|| / ||reference||`. For head consistency, the reference in this expression is the original same-precision separator output. For vocal-preservation experiments, it is the aligned vocal stem. Neither comparison fits a gain, searches alignment, clips peaks or normalizes the waveform.

Song-level scores are summarized by their median. The median paired score change can differ from the difference between two score medians. Three-song common-interval streaming scores are exploratory; their short duration is explicitly reported.

## Timing and memory

Processing-call timing runs from an in-memory audio array to a returned CPU waveform. It includes padding, transfers, neural inference and output assembly; excludes model loading, external warmup, audio-file I/O, scoring and audio-device buffers. Summed synchronized model-forward timing is also recorded with its narrower scope.

`RTF = processing_seconds / audio_seconds`. RTF below 1 means faster-than-playback average computation for that workload. It does not establish timely delivery of every block or actual device latency.

Head-study timing uses three independent processes per setting, each loading and warming up its model. Setting order changes between rounds. GPU temperatures/power/clocks are retained as context, not used to assert identical hardware conditions at every instant.

Memory reports use PyTorch peak allocated and reserved bytes. Allocated bytes describe tensors managed by the allocator; reserved bytes additionally include cache. Neither is the complete system-wide memory figure shown by `nvidia-smi`.

## Bounded-future stream scheduling

For output interval `[s, s+E)`, the input window is `[s-P, s+E+L)`. `P` is cached past, `E` the output block, and `L` explicit lookahead after the block. Total context is `P+E+L`. The input boundary needed for regular emission is `s+E+L`.

Thus availability relative to the first output boundary is `E+L`, and relative to the block end is `L`. Cached past is not future waiting. Zero lookahead still permits later samples within the output block to affect earlier samples in it.

The adapter accepts only the next small input packet. It pads missing past at startup and missing future only after an explicit EOF. Each emitted block recomputes the offline separator; the implementation has no neural-state cache or crossfade between output crops. All source dependencies and padding are recorded per emission.

The stream study uses the identical unpadded target interval shared by every setting, then separately reports whole-clip scores and full-real-context timing. A virtual serial worker uses `finish = max(input_available, previous_finish) + measured_compute`. Queue and deadline values are calculations on this virtual clock, not actual soundcard observations.

## Publication copy

The sanitized JSON export preserves numerical values, original experiment/report fingerprints and model provenance. Local filesystem paths are replaced with logical identifiers. Audio, model files, raw console logs and correspondence remain outside the public repository.

Publishing adds portable dataset-root and installation handling. The full model studies have not been rerun after those changes. Original research source hashes identify the recorded runs; new executions create their own run manifests.
