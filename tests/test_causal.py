"""Synthetic DSP/state checks, not a music separation benchmark."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from vocalstream.causal import (
    CausalConfig,
    CausalMaskGRU,
    CausalStreamProcessor,
    analysis_frames,
    benchmark_stream,
    load_checkpoint,
    synthesize_frames,
)


def run_identity(waveform: torch.Tensor, config: CausalConfig) -> torch.Tensor:
    processor = CausalStreamProcessor(config=config, identity_mask=True)
    pieces = []
    for start in range(0, waveform.shape[-1], config.hop_length):
        block = waveform[:, start : start + config.hop_length]
        block = F.pad(block, (0, config.hop_length - block.shape[-1]))
        pieces.append(processor.push(block))
    pieces.append(processor.flush())
    return torch.cat(pieces, dim=-1)


class CausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_invalid_reconstruction_configurations_are_rejected(self) -> None:
        for fft, hop in ((512, 200), (512, 512), (511, 128)):
            with self.subTest(fft=fft, hop=hop), self.assertRaises(ValueError):
                CausalConfig(n_fft=fft, hop_length=hop)

    def test_identity_reconstructs_edges_short_inputs_and_tail(self) -> None:
        for fft, hop in ((512, 128), (1024, 256), (2048, 512)):
            config = CausalConfig(n_fft=fft, hop_length=hop)
            for length in (1, 123, 256, 513, 2607):
                with self.subTest(fft=fft, length=length):
                    waveform = torch.randn(2, length) * 0.1
                    raw = run_identity(waveform, config)
                    aligned = raw[:, config.delay_samples : config.delay_samples + length]
                    torch.testing.assert_close(aligned, waveform, atol=2e-7, rtol=2e-5)
                    offline = synthesize_frames(analysis_frames(waveform, config), config, length)[0]
                    torch.testing.assert_close(aligned, offline, atol=2e-7, rtol=2e-5)

    def test_impulse_stream_displacement_and_availability(self) -> None:
        config = CausalConfig()
        waveform = torch.zeros(2, 1700)
        waveform[:, 0] = 1
        raw = run_identity(waveform, config)
        self.assertEqual(int(raw[0].argmax()), config.delay_samples)
        self.assertEqual(float(raw[0, config.delay_samples]), 1.0)
        # Original sample zero resides in output hop floor(D/H), which is
        # returned after that input hop has been received (N samples collected).
        self.assertEqual((config.delay_samples // config.hop_length + 1) * config.hop_length, config.n_fft)

    def test_random_weights_cannot_be_presented_as_separation(self) -> None:
        model = CausalMaskGRU(CausalConfig(n_fft=512, hop_length=128, hidden_size=32))
        with self.assertRaisesRegex(ValueError, "trained checkpoint"):
            CausalStreamProcessor(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "random.pt"
            torch.save({"format_version": 1, "config": model.config.to_dict(), "state_dict": model.state_dict(), "training": {"optimizer_steps": 0}}, path)
            with self.assertRaisesRegex(ValueError, "optimizer steps"):
                load_checkpoint(path)

    def test_no_future_frames_and_incremental_gru_state(self) -> None:
        config = CausalConfig(n_fft=512, hop_length=128, hidden_size=32)
        model = CausalMaskGRU(config).eval()
        spectrum = analysis_frames(torch.randn(2, 1200) * 0.1, config)
        altered = spectrum.clone()
        altered[:, 4:] *= 3.7
        with torch.inference_mode():
            result, _ = model(spectrum)
            changed, _ = model(altered)
            torch.testing.assert_close(result[:, :4], changed[:, :4], atol=1e-6, rtol=1e-5)
            chunks, state = [], None
            for frame in spectrum.unbind(1):
                output, state = model(frame[:, None], state)
                chunks.append(output)
            torch.testing.assert_close(result, torch.cat(chunks, dim=1), atol=2e-5, rtol=2e-5)

    def test_trained_checkpoint_stream_matches_frame_inference(self) -> None:
        config = CausalConfig(n_fft=512, hop_length=128, hidden_size=32)
        model = CausalMaskGRU(config)
        waveform = torch.randn(2, 1201) * 0.1
        spectrum = analysis_frames(waveform, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        prediction, _ = model(spectrum)
        # This one synthetic step exists only to exercise trained-checkpoint IO;
        # it is not advertised as learning to separate music.
        loss = (prediction.abs() - 0.5 * spectrum.abs()).square().mean()
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            torch.save({"format_version": 1, "config": config.to_dict(), "state_dict": model.state_dict(), "training": {"optimizer_steps": 1, "training_tracks": ["synthetic fixture"]}}, path)
            loaded, loaded_config, metadata = load_checkpoint(path)
            self.assertEqual(config, loaded_config)
            self.assertEqual(metadata["training"]["optimizer_steps"], 1)
            streamed, timing = benchmark_stream(waveform, loaded, warmup_hops=2)
            with torch.inference_mode():
                predicted, _ = loaded(spectrum)
                offline = synthesize_frames(predicted, config, waveform.shape[-1])[0]
            torch.testing.assert_close(streamed, offline, atol=3e-7, rtol=2e-5)
            self.assertEqual(streamed.shape, waveform.shape)
            self.assertEqual(timing["input_blocks"], 10)
            self.assertEqual(timing["flush_blocks"], 3)
            self.assertFalse(timing["end_to_end_latency_measured"])

    def test_flush_is_terminal_and_idempotent(self) -> None:
        config = CausalConfig(n_fft=512, hop_length=128)
        processor = CausalStreamProcessor(config=config, identity_mask=True)
        processor.push(torch.zeros(2, 128))
        self.assertEqual(processor.flush().shape, (2, 384))
        self.assertEqual(processor.flush().shape, (2, 0))
        with self.assertRaises(RuntimeError):
            processor.push(torch.zeros(2, 128))


if __name__ == "__main__":
    unittest.main()
