# Third-party notices and asset provenance

## Vendored BS-RoFormer implementation

`vocalstream/vendor/zfturbo_bsroformer/` contains source from
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
commit `aef04b2e52fb3beaf25e333199f5a7236e628e7b` (the source associated with release `v1.0.12`).

- Copyright (c) 2024 Roman Solovyev (ZFTurbo).
- License: MIT; the full upstream notice is preserved in [the vendored LICENSE](vocalstream/vendor/zfturbo_bsroformer/LICENSE).
- Local adaptation: the `Attend` import in `bs_roformer.py` uses `.attend` instead of the upstream `models.bs_roformer.attend` namespace.
- Vocal-head selection is implemented in the project adapter outside the vendored architecture. Shared layers and the selected pretrained head retain their weights.

The architecture also builds on [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer),
Copyright (c) 2023 Phil Wang, MIT licensed. The installed `BS-RoFormer==0.3.9` dependency and optional Mel-Band RoFormer implementation retain their own notices. See the [upstream license](https://github.com/lucidrains/BS-RoFormer/blob/93a07dda7867d4acd8f5bd49ae3f33e1fcbfd8cf/LICENSE).

## Main checkpoint

The BS-RoFormer weights are obtained separately from the [ZFTurbo v1.0.12 release](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.12):

| Asset | SHA-256 |
| --- | --- |
| `model_bs_roformer_ep_17_sdr_9.6568.ckpt` | `3e9daecd70aaed5b5a0d1f861cc4d77eaa45afb3fc6301b1cf32c1be0f5868fb` |
| `config_bs_roformer_384_8_2_485100.yaml` | `d8afb980318d0c08b9c2e24a7adc00d4f3150320c127a7e4de861800d1321939` |

This is a community checkpoint, not the original BS-RoFormer paper's checkpoint. The publisher describes MUSDB18-HQ training but does not provide the complete training/validation song list. The release does not provide a separate explicit weight license in the recorded provenance. The project's code license does not grant rights to those weights. They are excluded from this repository.

## Audio dataset

MUSDB18-HQ is obtained directly from its [official dataset page](https://sigsep.github.io/datasets/musdb.html) / [dataset record](https://doi.org/10.5281/zenodo.3338373). Its recordings and stems are governed by the dataset and recording owners' terms. No mixture, stem, separated waveform, commercial-song excerpt, or internal listening pack is redistributed here.

Public reports contain scores, benchmark track names, hashes, and computational measurements. They do not contain audio samples.

## Other dependencies

PyTorch, torchaudio, Open-Unmix, NumPy, SciPy, SoundFile, musdb, museval, PyYAML, einops, and plotting libraries are installed separately. Their licenses remain applicable. Dependency installation does not transfer their ownership or extend this project's MIT license to external assets.
