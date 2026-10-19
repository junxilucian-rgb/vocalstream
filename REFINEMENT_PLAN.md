# Refinement study

The recorded protocol has 18 declared jobs: four expanded quality/context settings on 14 fixed development excerpts, four preservation/bleed conditions for two models, and three independent timing repetitions for two BS settings.

Run: python run_refinement.py --root path/to/musdb18-hq --out results/refinement_new

Then: python build_refinement_report.py --run results/refinement_new

See [the reproduction guide](docs/REPRODUCING.md) and [public results](docs/RESULTS.md). Original-vocal stems may include production effects; scaled accompaniment is not a real microphone simulation.
