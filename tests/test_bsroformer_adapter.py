"""Four-stem target mapping and lossless vocal overlap-add edge coverage."""
import unittest
import numpy as np
import torch

from vocalstream.bsroformer import BSRoFormerBaseline


class IdentityFourStem(torch.nn.Module):
    def forward(self, waveform):
        zero = torch.zeros_like(waveform)
        return torch.stack([zero, zero, zero, waveform], dim=1)


class AdapterTests(unittest.TestCase):
    def dummy(self, overlap):
        separator = object.__new__(BSRoFormerBaseline)
        separator.device = torch.device('cpu')
        separator.precision, separator.tta = 'fp32', False
        separator.heads = 'all'
        separator.overlap = overlap
        separator.chunk_samples = 4410
        separator.step_samples = 4410//overlap
        separator.fade_samples = 441
        separator.border_samples = 4410-separator.step_samples
        separator.stem_order = ['drums', 'bass', 'other', 'vocals']
        separator.vocal_index = 3
        separator.model = IdentityFourStem()
        return separator

    def test_vocal_is_fourth_head_and_bad_layout_is_rejected(self):
        separator = self.dummy(2)
        tensor = torch.stack([torch.full((1,2,4410), float(i)) for i in range(4)], dim=1)
        self.assertTrue(torch.all(separator._vocal_prediction(tensor) == 3))
        with self.assertRaises(ValueError):
            separator._vocal_prediction(tensor[:, :3])

    def test_all_overlaps_reconstruct_short_tail_and_reflected_input(self):
        rng = np.random.default_rng(30)
        for overlap in (2,4,8):
            separator = self.dummy(overlap)
            for samples in (100,2205,4410,9307,15001):
                with self.subTest(overlap=overlap,samples=samples):
                    audio = rng.normal(0,.1,(samples,2)).astype('float32')
                    result, _ = separator.separate(audio)
                    self.assertEqual(result.shape,audio.shape)
                    np.testing.assert_allclose(result,audio,atol=1e-7,rtol=1e-6)


if __name__=='__main__':
    unittest.main()
