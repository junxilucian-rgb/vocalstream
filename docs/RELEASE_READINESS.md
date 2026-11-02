# Release verification — 0.2.0

Checked on **2026-11-02**. This release prepares the completed experiments for review and interview discussion. It does not add another GPU experiment or claim real-time vocal separation.

## Checks completed

| Check | Observed result |
| --- | --- |
| Full regression suite, `python check_project.py` | **67 tests passed** in the existing Python 3.12 environment. |
| Public evidence, `python -S scripts/verify_public_evidence.py` | **4 studies, 72 completed jobs, 77 JSON files**; declared hashes, counts, completion states and public paths passed. |
| Original-to-public evidence audit | All 72 original job reports matched their recorded fingerprints; sanitized reports retained their numerical results. Main study fingerprints and README claims were also checked. |
| Dependency consistency, `python -m pip check` | No broken requirements in the existing environment. |
| Distribution package | Built `vocalstream_research-0.2.0-py3-none-any.whl` without downloading dependencies. Verified version, single-window adapter, vendor license, and absence of audio/model weights. |
| Installed package smoke check | Installed the wheel into a separate temporary target, then imported version 0.2.0 and the single-window adapter from outside the repository using the existing runtime. |
| Documentation | Checked local Markdown file targets, CV claims and interview explanations against the published evidence. |

The 67 tests include **16 standard-library evidence-verifier tests** and **14 new streaming/window/report regression tests**. They cover future-input boundaries, packet and EOF alignment, virtual queue arithmetic, actual network call counts, head mapping, invalid model output, and evidence corruption. These checks exercise implementation behavior; they are not new separation-quality or GPU-performance measurements.

The evidence-only GitHub Actions workflow runs the verifier and its 16 fixture tests on pushes and pull requests, without downloading datasets, models or Python ML packages. The full 67-test suite was run locally. A configured workflow alone is not evidence of a successful remote run; consult the repository's Actions history for its status.

## Run the checks

From the repository root, after installing the project dependencies:

```powershell
python check_project.py
python -S scripts/verify_public_evidence.py
python -m pip check
```

The public-evidence check alone requires only Python 3.12 and the checkout:

```powershell
python -S -m unittest discover -s tests -p test_public_evidence.py -v
```

In the restricted Windows environment used for this review, temporary-directory permissions initially blocked some tests and the package build. Using a writable repository-local temporary directory resolved the issue:

```powershell
New-Item -ItemType Directory -Force .tmp\checks | Out-Null
$env:TEMP = (Resolve-Path .tmp\checks).Path
$env:TMP = $env:TEMP
```

The dependency check and wheel smoke check used the existing environment. A full dependency installation on a clean machine was **not** performed.

## What the evidence supports

- Pretrained Open-Unmix/BS-RoFormer reproduction and controlled vocal-preservation/bleed evaluation on fixed development excerpts.
- Vocal-head selection with **56.1% fewer active parameters** and same-precision output equality on the 14 evaluated excerpts. Combined with AMP, the separate repeated offline timing study measured **2.35× throughput** relative to four-head FP32.
- A matched single-window experiment with **25 jobs**, on three 4-second development clips, reduced typical block computation from approximately **183 ms to 91 ms** by eliminating a redundant inner inference pass. This changes predictions and has one timing run per song/setting/policy.
- Explicit reporting that the tested **20/80 ms compute budgets were still missed**. The virtual queue results are not soundcard end-to-end latency measurements.

The two speed ratios come from different workloads and must not be multiplied. The checkpoint's full training-song list is unavailable, so these development scores are not a held-out generalization benchmark. No new separation architecture, live feedback-control system, or full-song test-set result is claimed.

## Source and artifact history

Commit [`215bfa9`](https://github.com/junxilucian-rgb/vocalstream/commit/215bfa9) preserves the inference implementation used for the completed single-window study. The subsequent 0.2.0 release adds tests, verification, documentation and package-version metadata. Historical report fingerprints and numerical values remain unchanged. A source hash from an old run can therefore differ from the final release checkout; do not rewrite old reports to make it match. Use a new output directory for new experiments.

For the interview, start with [the CV wording](CV.md) and [the six-session learning guide](INTERVIEW_GUIDE.zh-CN.md), then trace each claim through [the results](RESULTS.md) and [window-inference study](WINDOW_INFERENCE.md).
