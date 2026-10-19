"""BS-RoFormer HQ community checkpoint using its publisher's pinned code."""
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from .roformer import MelRoFormerBaseline, _sha256, _read_config, _state_dict

CHECKPOINT_SHA256 = '3e9daecd70aaed5b5a0d1f861cc4d77eaa45afb3fc6301b1cf32c1be0f5868fb'
CONFIG_SHA256 = 'd8afb980318d0c08b9c2e24a7adc00d4f3150320c127a7e4de861800d1321939'


class BSRoFormerBaseline(MelRoFormerBaseline):
    """Pinned network with optional inference-only selection of its vocal head.

    The full checkpoint always loads strictly before head selection. The
    publisher's shared trunk and selected trained mask estimator are unchanged.
    ``heads='vocals'`` removes three independent mask heads and their ISTFTs;
    it does not make temporal attention causal or retrain any weights.
    """
    def __init__(self, device='cuda', overlap=2, precision='fp32', tta=False,
                 chunk_seconds=None, warmup=True, heads='all'):
        if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap not in (2, 4, 8):
            raise ValueError('overlap must be 2, 4 or 8')
        if precision not in ('fp32', 'amp'):
            raise ValueError('precision must be fp32 or amp')
        if heads not in ('all', 'vocals'):
            raise ValueError('heads must be all or vocals')
        self.heads = heads
        self.device = torch.device(device)
        if self.device.type not in ('cuda', 'cpu') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Choose an available cpu/cuda device')
        if precision == 'amp' and self.device.type != 'cuda':
            raise ValueError('amp requires CUDA')
        self.precision, self.overlap, self.tta = precision, overlap, bool(tta)
        project = Path(__file__).resolve().parents[1]
        self.model_dir = project / 'model_cache' / 'bs_roformer_hq'
        checkpoint_path, config_path = self.model_dir/'model.ckpt', self.model_dir/'config.yaml'
        start = time.perf_counter()
        self.checkpoint_sha256, self.config_sha256 = _sha256(checkpoint_path), _sha256(config_path)
        if self.checkpoint_sha256 != CHECKPOINT_SHA256 or self.config_sha256 != CONFIG_SHA256:
            raise ValueError('BS-RoFormer checkpoint/config failed pinned SHA256 verification')
        self.provenance = json.loads((self.model_dir/'provenance.json').read_text(encoding='utf-8'))
        vendor_root = project / 'vocalstream' / 'vendor' / 'zfturbo_bsroformer'
        for name, source_info in self.provenance['source_files'].items():
            if _sha256(vendor_root/name) != source_info['local_sha256']:
                raise ValueError(f'Pinned publisher source changed: {name}')
        from .vendor.zfturbo_bsroformer import BSRoformer
        self.vendor_source = str(Path(inspect.getfile(BSRoformer)))
        self.vendor_sha256 = _sha256(Path(self.vendor_source))
        config = _read_config(config_path)
        self.sample_rate = int(config['audio']['sample_rate'])
        self.stem_order = list(config['training']['instruments'])
        if self.sample_rate != 44100 or self.stem_order != ['drums', 'bass', 'other', 'vocals']:
            raise ValueError('Unexpected sample rate or target stem order')
        self.vocal_index = self.stem_order.index('vocals')
        self.trained_chunk_samples = int(config['audio']['chunk_size'])
        self.chunk_samples = self.trained_chunk_samples
        if chunk_seconds is not None:
            if not np.isfinite(chunk_seconds) or not .25 <= chunk_seconds <= 11:
                raise ValueError('chunk_seconds must be finite in [0.25, 11]')
            hop = int(config['model']['stft_hop_length'])
            self.chunk_samples = round(chunk_seconds * self.sample_rate / hop) * hop
        self.step_samples = self.chunk_samples // overlap
        self.fade_samples = self.chunk_samples // 10
        self.border_samples = self.chunk_samples - self.step_samples
        self.model = BSRoformer(**config['model'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state, self.checkpoint_state_key = _state_dict(checkpoint)
        self.model.load_state_dict(state, strict=True)
        del checkpoint, state
        self.full_parameters = sum(p.numel() for p in self.model.parameters())
        self.original_head_parameters = [sum(p.numel() for p in head.parameters())
                                         for head in self.model.mask_estimators]
        if len(self.model.mask_estimators) != len(self.stem_order):
            raise ValueError('Mask estimator count does not match pinned stem order')
        if heads == 'vocals':
            # Independent heads run after the common final_norm; no mixing or
            # cross-head normalization follows them. Select ONLY after strict
            # four-stem loading, without modifying the pinned vendor source.
            self.model.mask_estimators = torch.nn.ModuleList([
                self.model.mask_estimators[self.vocal_index]])
            self.model.num_stems = 1
        self.output_stem_order = ['vocals'] if heads == 'vocals' else self.stem_order.copy()
        self.model.eval().requires_grad_(False).to(self.device)
        self._sync()
        self.load_seconds = time.perf_counter() - start
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.last_run = None
        self.warmup_seconds = 0.0
        self.warmup_applied = bool(warmup)
        if warmup:
            begin = time.perf_counter()
            self.separate(np.zeros((self.chunk_samples, 2), dtype=np.float32))
            self.warmup_seconds = time.perf_counter()-begin
            self.last_run = None

    def _vocal_prediction(self, prediction):
        if self.heads == 'vocals':
            expected = (1, 2, self.chunk_samples)
            if not torch.is_tensor(prediction) or tuple(prediction.shape) != expected:
                raise ValueError(f'Expected {expected} vocal-only prediction; got {getattr(prediction, "shape", None)}')
            return prediction
        expected = (1, len(self.stem_order), 2, self.chunk_samples)
        if not torch.is_tensor(prediction) or tuple(prediction.shape) != expected:
            raise ValueError(f'Expected {expected} four-stem prediction; got {getattr(prediction, "shape", None)}')
        return prediction[:, self.vocal_index]

    def metadata(self):
        result = super().metadata()
        result.update({
            'name': 'BS-RoFormer ZFTurbo MUSDB18-HQ pretrained',
            'architecture': 'BSRoformer', 'checkpoint_repository': 'ZFTurbo/Music-Source-Separation-Training',
            'checkpoint_revision': 'v1.0.12', 'architecture_package': 'pinned publisher source (local namespace)',
            'architecture_commit': self.provenance['source_commit'], 'source_provenance': self.provenance,
            'stem_order': self.stem_order, 'vocal_stem_index': self.vocal_index,
            'network_outputs_all_four_stems': self.heads == 'all', 'returned_audio_target': 'vocals',
            'inference_heads': self.heads, 'output_stem_order': self.output_stem_order,
            'full_checkpoint_parameters': self.full_parameters,
            'removed_parameters': self.full_parameters - self.parameters,
            'original_mask_head_parameters': self.original_head_parameters,
            'head_selection_after_strict_load': True,
            'head_selection_scope': 'inference only; pretrained vocal head and shared trunk unchanged; source checkpoint remains unmodified',
            'trained_chunk_seconds': self.trained_chunk_samples/self.sample_rate,
            'input_context_reduced_without_retraining': self.chunk_samples != self.trained_chunk_samples,
            'warmup_applied': self.warmup_applied, 'warmup_seconds': self.warmup_seconds,
            'latency_reason': 'Offline bidirectional attention and overlap-add use future samples; context length is not a measured causal latency',
            'training_split_verified': False, 'exact_original_paper_checkpoint': False,
        })
        return result
