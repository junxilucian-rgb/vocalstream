# Public experiment evidence

Read the [results explanation](../docs/RESULTS.md) or [中文说明](../docs/RESULTS.zh-CN.md) before interpreting these numbers.

| File | Completed jobs | Scope |
|---|---:|---|
| [head_optimization.json](head_optimization.json) | 16 | Four precision/head settings on 14 fixed development excerpts, plus 3 fresh-process timing runs per setting |
| [streaming_study.json](streaming_study.json) | 13 | Four file-streaming settings × 3 short development excerpts, plus one offline reference job |
| [refinement.json](refinement.json) | 18 | Earlier Open-Unmix/BS-RoFormer, context, controlled-bleed and timing experiments |
| [window_comparison.json](window_comparison.json) | 25 | Matched nested-OLA/single-forward policies × four schedules × three short development clips, plus one offline reference |

Total: **72 per-job reports**. The window comparison was measured on 2026-10-22, with one timing observation per song/schedule/policy. It is not an independent repeated-timing benchmark. Its earlier OLA counterpart remains in the historical streaming study; fresh timing ratios use the newly rerun OLA baseline.

Each main JSON contains `analysis`, `protocol`, `execution` and `jobs`. `jobs[*].public_report` is relative to this directory and identifies a distributed per-job report. Streaming worker reports retain individual block timings; head optimization analysis retains per-track waveform errors. [index.json](index.json) records main-file SHA-256 hashes; job manifests record original and sanitized report hashes separately.

Original absolute paths have been replaced with logical identifiers. Historical `results/...` audio/log paths and `MUSDB18-HQ/...` dataset paths are provenance identifiers, not distributed files. No audio, weights, console logs or correspondence are included. Original-file hashes identify evidence retained outside this publication; they will not match sanitized-file bytes. Measurements were not changed.

The original historical code SHA-256 values are retained even when publication cleanup changes current code. Reproduction creates a new run and need not reproduce exact timings or waveforms on different software/hardware. These development excerpts do not establish held-out generalization or ultra-low-latency device operation.

The stdlib-only [export utility](../scripts/export_public_results.py) can rebuild these files from an existing complete evidence directory. It verifies original report hashes and performs no model inference:

```bash
python scripts/export_public_results.py --source-project /path/to/original-project
```

Append one completed 25-job local window study, preserving the existing index:

```shell
python scripts/export_public_results.py --run results/window_compare_01 --study-name window_comparison
```
