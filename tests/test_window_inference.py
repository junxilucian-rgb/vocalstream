"""One-call inference contract, without a checkpoint, downloads or GPU."""
import unittest

import numpy as np
import torch

from vocalstream.bsroformer import BSRoFormerBaseline
from vocalstream.window_inference import BSRoFormerWindowSeparator


class RecordingModel(torch.nn.Module):
    def __init__(self, heads="vocals", invalid=None):
        super().__init__()
        self.heads, self.invalid, self.calls = heads, invalid, []

    def forward(self, waveform):
        self.calls.append(waveform.detach().clone())
        vocals = waveform * .25
        if self.invalid == "nonfinite":
            vocals[..., 0] = float("nan")
        if self.invalid == "length":
            return vocals[..., :-1]
        if self.heads == "all":
            return torch.stack([waveform+1, waveform+2, waveform+3, vocals], dim=1)
        return vocals


def make_separator(kind=BSRoFormerWindowSeparator, heads="vocals", invalid=None):
    separator = object.__new__(kind)
    separator.device = torch.device("cpu")
    separator.precision, separator.tta, separator.heads = "fp32", False, heads
    separator.sample_rate = 44100
    separator.chunk_samples = 4410
    separator.overlap, separator.step_samples = 2, 2205
    separator.fade_samples, separator.border_samples = 441, 2205
    separator.stem_order = ["drums", "bass", "other", "vocals"]
    separator.vocal_index = 3
    separator.model = RecordingModel(heads, invalid)
    return separator


class WindowInferenceTests(unittest.TestCase):
    def test_single_forward_preserves_alignment_gain_and_vocal_target(self):
        audio = np.random.default_rng(10).normal(size=(4410, 2)).astype(np.float32)
        original = audio.copy()
        for heads in ("all", "vocals"):
            with self.subTest(heads=heads):
                separator = make_separator(heads=heads)
                result, forward_seconds = separator.separate(audio)
                np.testing.assert_array_equal(result, audio*.25)
                np.testing.assert_array_equal(audio, original)
                np.testing.assert_array_equal(separator.model.calls[0].numpy()[0].T, audio)
                self.assertEqual(len(separator.model.calls), 1)
                self.assertEqual(separator.last_run["model_forward_passes"], 1)
                self.assertFalse(separator.last_run["global_reflect_border_applied"])
                self.assertFalse(separator.last_run["gain_normalization_applied"])
                self.assertFalse(separator.last_run["clipping_applied"])
                self.assertGreaterEqual(forward_seconds, 0)

    def test_old_adapter_has_two_calls_for_same_exact_context(self):
        audio = np.ones((4410, 2), dtype=np.float32)
        old = make_separator(BSRoFormerBaseline)
        new = make_separator()
        old.separate(audio)
        new.separate(audio)
        self.assertEqual(len(old.model.calls), 2)
        self.assertEqual(len(new.model.calls), 1)
        np.testing.assert_array_equal(old.model.calls[1].numpy()[0, :, :2205], audio.T[:, 2205:])
        np.testing.assert_array_equal(old.model.calls[1].numpy()[0, :, 2205:], 0)

    def test_wrong_context_length_rejected_before_network(self):
        for count in (1, 4409, 4411, 8820):
            with self.subTest(count=count):
                separator = make_separator()
                with self.assertRaisesRegex(ValueError, "exactly"):
                    separator.separate(np.zeros((count, 2), dtype=np.float32))
                self.assertEqual(separator.model.calls, [])

    def test_invalid_predictions_cannot_be_returned_as_audio(self):
        for invalid in ("nonfinite", "length"):
            with self.subTest(invalid=invalid):
                separator = make_separator(invalid=invalid)
                with self.assertRaises(ValueError):
                    separator.separate(np.zeros((4410, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
