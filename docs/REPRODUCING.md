# Reproducing the project

Run commands from the cloned repository root, using one Python environment for an experiment. Python 3.12 is the declared supported version. [README.md](../README.md) gives the installation and single-file demonstration.

## 1. Environment and assets

```shell
conda create -n vocalstream python=3.12 pip -y
conda activate vocalstream
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python download_bs_roformer_assets.py
```

For a CPU installation, install the matching Torch/Torchaudio CPU wheels instead and use FP32. The fixed four-way AMP study requires CUDA.

Optional components:

```shell
python -m pip install -e ".[metrics]"
python -m pip install -e ".[reports]"
```

`metrics` adds musdb/museval for the extended BSSEval or educational training workflows; `reports` adds plotting/PDF dependencies. For an exact historical CUDA environment, inspect `requirements-roformer-lock.txt`; it records the experiment environment and may require the indicated platform-specific CUDA index. Main installation metadata lists the directly required packages.

`setup_roformer_runtime.py` is an alternative setup helper. Its explicit `freeze` phase writes a local ignored runtime lock; normal installation does not replace the committed historical lock.

## 2. Dataset location

Obtain MUSDB18-HQ from its [official page](https://sigsep.github.io/datasets/musdb.html). Keep its original `train` and `test` folder layout. The main experiment scripts resolve their dataset root in this order:

1. Explicit `--root` argument.
2. `MUSDB_ROOT` environment variable.
3. Configuration value, relative to the repository when applicable.
4. Repository-local `data/musdb18-hq` default.

PowerShell example:

```powershell
$env:MUSDB_ROOT = 'D:\datasets\musdb18-hq'
```

Bash example:

```shell
export MUSDB_ROOT=/path/to/musdb18-hq
```

The examples below use an explicit root so their input is easy to audit. The optional inventory utility `python prepare_data.py path/to/musdb18-hq` reads folder/file metadata and a short alignment segment from both train and test. It does not run model inference or choose parameters from test scores. The development-study runners select only the reserved train-folder songs.

## 3. Vocal-head and precision study

```shell
python run_head_optimization.py --root path/to/musdb18-hq --out results/head_study_01
python build_head_optimization_report.py --run results/head_study_01
```

Protocol: 14 official development songs, fixed 60–90 s excerpts, 11 s model context, overlap 2, no TTA. Four quality jobs compare all/vocal heads × FP32/AMP. Each setting additionally receives three independent fresh-process timing jobs on the same three songs. Total: 16 jobs.

Each quality estimate is saved as an aligned FLOAT WAV. The report generator compares same-precision outputs without shift/gain fitting and checks the declared waveform/SI-SDR tolerances. It records cross-precision errors separately. Loading, warmup and file operations are outside reported call timing.

Optional plots, after installing `[reports]`:

```shell
python analysis_tools/plot_head_optimization.py --run results/head_study_01
```

## 4. Bounded-future file-stream study

```shell
python run_streaming_study.py --root path/to/musdb18-hq --out results/stream_study_01
python build_streaming_study_report.py --run results/stream_study_01
```

Protocol: three development songs, 60–64 s, vocal-only AMP, 2 s context. Compare `E/L = 20/0, 20/40, 20/160, 80/40 ms`. There are 12 stream jobs and one three-song offline reference job. The exact common unpadded target interval is derived from emission records; the recorded run used `[1.980, 3.840) s` relative to clip start.

This is one short timing observation per song/setting. Report computation over the output period, virtual queue/deadline misses, and SI-SDR separately. Audio-device latency is not measured. The recorded implementation exceeded all four compute budgets; a new run need not match its exact timing.

## 5. Baselines, bleed and preservation

```shell
python run_refinement.py --root path/to/musdb18-hq --out results/refinement_01
python build_refinement_report.py --run results/refinement_01
```

This larger study uses the same 14 development excerpts for Open-Unmix and BS-RoFormer quality/context comparisons, original-vocal input, three accompaniment-gain conditions and independent repeated timing. It produces 18 declared jobs. Gain-scaled accompaniment is a controlled proxy rather than a physical microphone/room simulation.

The Open-Unmix pretrained loader can download its model on first use. Its own upstream model conditions apply. Any locally generated listening examples belong in ignored result folders and retain their audio owners' terms.

## 6. Whole-song validation and frozen test workflow

The following long-running stages are available, but the published results are development excerpts rather than completed full-song benchmarks.

```shell
python run_reference_project.py run --config reference_optimized_experiments.json --root path/to/musdb18-hq --stage validation --out results/whole_validation_01
```

Select a setting based on validation results and record the reason before accessing final-test scores:

```shell
python run_reference_project.py freeze --run results/whole_validation_01 --experiment bs_hq_vocal_amp_o2 --reason "Replace with the reason supported by your validation results"
python run_reference_project.py run --config reference_optimized_experiments.json --root path/to/musdb18-hq --stage test --frozen results/whole_validation_01/frozen_reference.json --out results/final_test_01
```

Install `[metrics]` for final BSSEval scoring. Code, runtime, dataset root and configuration must agree with the frozen manifest. Keep code stable during validation through final test. Freezing this project's choices does not establish that an external pretrained model never saw a test song.

## 7. Results, resuming and sharing

- Run GPU experiments sequentially. Concurrent workloads change timing conditions.
- Choose a new output directory for a new protocol or code/runtime version. Main runners use OS locks and protect completed report hashes.
- Repeat the same command to resume a compatible interrupted run. Failed jobs get a new attempt directory; preserved completed jobs are checked before reuse.
- Local outputs include model metadata, input/reference fingerprints, commands, console logs and FLOAT audio. Treat raw reports as machine-local records because they contain paths.
- Public sanitized evidence is under [reports/](../reports/README.md). Its index records original and public-export hashes. The export script takes an explicit historical results root; it does not copy audio or checkpoints.
- This publishing pass did not run the test suite or repeat the full ML experiments. Existing engineering checks are retained under `tests/`; the recorded experiment claims are supported by their exported evidence.

See [METHODOLOGY.md](METHODOLOGY.md) for exact metric definitions and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for source and asset provenance.
