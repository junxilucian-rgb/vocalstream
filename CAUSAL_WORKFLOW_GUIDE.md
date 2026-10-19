# Optional educational causal branch

The main published experiments use pretrained BS-RoFormer and Open-Unmix. The additional STFT-GRU modules support educational training and streaming experiments; they are not a reproduction of HS-TasNet or RT-STT and have no published real-music quality claim here.

Install the optional metrics dependencies and obtain MUSDB18-HQ separately. From the repository root:

```shell
python run_project.py prepare --root path/to/musdb18-hq --profile development --out results/causal_plan_01
python run_project.py run --plan results/causal_plan_01/plan.json
```

Inspect the plan budget and hardware settings before running. The standard four-candidate development budget is substantially larger than a short inference demo. Preserve the train/validation split and freeze choices before final-test evaluation.

Start with [README.md](README.md) and [methodology](docs/METHODOLOGY.md) for the current measured project.
