# Change history

## 0.2.0 — 2026-11-02

- Add single-window BS-RoFormer inference with one forward per supplied context, retaining the original nested overlap-add policy as a comparison.
- Include the completed 25-job matched window study from 2026-10-22, with common-target quality scores and per-emission timing. Total public evidence: 72 job reports. All observed 20/80 ms compute budgets still failed.
- Add regression coverage for future-input boundaries, EOF alignment, virtual scheduling, output-head mapping, call counts and report guards.
- Add public-evidence verification, an evidence-only CI workflow, and Chinese/English CV and interview learning materials.
- Preserve the measured window-study source version in commit `215bfa9`; the release changes package metadata and documentation after that experiment. Use new output folders for new experiments. Earlier reports keep their original fingerprints and numerical values.

See [release checks and scope](docs/RELEASE_READINESS.md). This release adds engineering verification; it does not add another GPU timing experiment or a full-song test benchmark.

## 0.1.0 — 2026-10-19

- Publish the initial pretrained baseline, controlled bleed, output-head/precision and bounded-future streaming studies.
- Include 47 sanitized job reports, reproducibility guides and original source/model attribution.
