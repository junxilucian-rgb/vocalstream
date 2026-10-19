import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from vocalstream.data import discover_hq, load_segment


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.song = self.root / "test" / "synthetic"
        self.song.mkdir(parents=True)
        rng = np.random.default_rng(0)
        self.stems = {name: rng.normal(0, 0.01, (4410, 2)).astype('float32')
                      for name in ("vocals", "drums", "bass", "other")}
        self.stems["mixture"] = sum(self.stems.values())
        for name, value in self.stems.items():
            sf.write(self.song / (name + '.wav'), value, 44100, subtype='FLOAT')

    def tearDown(self):
        self.tmp.cleanup()

    def test_alignment_and_segment(self):
        track = discover_hq(self.root)[0]
        arrays, info = load_segment(track, start=0.02, duration=0.03)
        self.assertEqual(arrays['vocals'].shape, (1323, 2))
        self.assertLess(info['mixture_sum_max_abs_difference'], 1e-7)
        np.testing.assert_allclose(arrays['vocals'], self.stems['vocals'][882:2205])

    def test_missing_or_misaligned_stem_rejected(self):
        sf.write(self.song / 'vocals.wav', self.stems['vocals'][:-1], 44100, subtype='FLOAT')
        with self.assertRaises(ValueError):
            discover_hq(self.root)

    def test_invalid_start(self):
        track = discover_hq(self.root)[0]
        for start in (-1, 0.1, float('nan')):
            with self.assertRaises(ValueError):
                load_segment(track, start)


if __name__ == '__main__':
    unittest.main()
