# Contributing

Please open an issue describing the problem and a small reproducible command before substantial architecture or protocol changes.

Useful issue details:

- Operating system, Python/PyTorch versions, CPU/GPU, thread count, and precision.
- Exact command, expected behavior, and observed behavior.
- Public checkpoint/config hashes and the relevant metric definitions.
- A sanitized log or minimal synthetic example.

Keep recordings, model weights, private paths, account tokens, and personal correspondence out of issues and pull requests. Dataset access remains the contributor's responsibility.

For experiment changes, use a new output directory and preserve the previous result. Separate model quality, processing throughput, algorithmic input dependencies, and measured device latency. A speed claim should identify the full timing scope and workload. Report negative results and undefined metrics.

Changes to vendored architecture files require attribution and updated provenance. Prefer adapters outside the pinned source. Contributions are under the project's MIT license; third-party notices must be retained.
