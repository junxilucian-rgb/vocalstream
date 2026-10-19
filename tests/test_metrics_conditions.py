"""Deterministic numerical checks; no training data or model downloads needed."""

from __future__ import annotations

import json
import unittest

import numpy as np

from vocalstream.conditions import make_condition, synthetic_rir
from vocalstream.metrics import global_residual_sdr, separation_metrics, si_sdr, timing_summary


class MetricsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(40)
        self.reference = rng.standard_normal((4096, 2))
        noise = rng.standard_normal((4096, 2))
        # Orthogonalize centered noise to obtain a known projected signal/error.
        self.reference -= self.reference.mean(axis=0, keepdims=True)
        noise -= noise.mean(axis=0, keepdims=True)
        noise -= np.sum(noise * self.reference) / np.sum(self.reference**2) * self.reference
        self.noise = noise / np.sqrt(np.sum(noise**2) / np.sum(self.reference**2))

    def test_si_sdr_known_ratio_and_scale_invariance(self):
        estimate = self.reference + self.noise * 0.1
        score, reason = si_sdr(self.reference, estimate)
        self.assertIsNone(reason)
        self.assertAlmostEqual(score, 20.0, places=9)
        scaled, _ = si_sdr(self.reference, estimate * 3.0)
        self.assertAlmostEqual(score, scaled, places=9)

    def test_per_channel_dc_removed_and_joint_gain_preserved(self):
        estimate = self.reference + self.noise * 0.1
        score, _ = si_sdr(self.reference, estimate)
        shifted, _ = si_sdr(self.reference + [3.0, -2.0], estimate + [-5.0, 4.0])
        self.assertAlmostEqual(score, shifted, places=9)
        perfect, _ = si_sdr(self.reference, self.reference)
        unequal_channels, _ = si_sdr(self.reference, self.reference * [1.0, 0.5])
        self.assertGreater(perfect, unequal_channels)

    def test_silence_is_none_with_reason(self):
        score, reason = si_sdr(np.zeros((16, 2)), np.ones((16, 2)))
        self.assertIsNone(score)
        self.assertIn("reference_silent", reason)
        score, reason = si_sdr(self.reference, np.zeros_like(self.reference))
        self.assertIsNone(score)
        self.assertIn("estimate_silent", reason)

    def test_si_sdri_and_json_safety(self):
        mixture = self.reference + self.noise
        scores = separation_metrics(self.reference, self.reference + 0.1 * self.noise, mixture)
        self.assertAlmostEqual(scores["si_sdri_db"], 20.0, places=9)
        self.assertFalse(scores["protocol"]["museval_bsseval_sdr"])
        json.dumps(scores, allow_nan=False)
        json.dumps(separation_metrics(self.reference, self.reference, mixture), allow_nan=False)

    def test_global_residual_sdr_penalizes_gain(self):
        score, reason = global_residual_sdr(self.reference, self.reference * 2.0)
        self.assertIsNone(reason)
        self.assertAlmostEqual(score, 0.0, places=9)

    def test_metrics_do_not_modify_input(self):
        original = self.reference.copy()
        separation_metrics(self.reference, self.reference * 0.7, self.reference + self.noise)
        np.testing.assert_array_equal(self.reference, original)

    def test_bad_shapes_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            si_sdr(self.reference, self.reference[:-1])
        with self.assertRaises(ValueError):
            si_sdr(self.reference, self.reference * np.nan)

    def test_weighted_rtr_and_deadline_misses(self):
        report = timing_summary([0.02, 0.1, 0.3], [0.01, 0.2, 0.2])
        self.assertAlmostEqual(report["mean_rtr"], (2.0 + 0.5 + 1.5) / 3.0)
        self.assertAlmostEqual(report["total_rtr"], 0.42 / 0.41)
        self.assertEqual(report["deadline_miss_count"], 2)
        self.assertEqual(report["max_rtr"], 2.0)
        self.assertFalse(report["latency_inferred_from_rtr"])
        json.dumps(report, allow_nan=False)

    def test_timing_invalid_values(self):
        for processing, duration in [([], []), ([1.0], []), ([-0.1], [1.0]), ([0.1], [0.0]), ([np.inf], [1.0])]:
            with self.assertRaises(ValueError):
                timing_summary(processing, duration)


class ConditionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.vocals = (rng.standard_normal((16000, 2)) * 0.1).astype(np.float32)
        self.backing = (rng.standard_normal((16000, 2)) * 0.15).astype(np.float32)
        self.sample_rate = 16000

    def test_mixture_and_clean_tasks_are_explicit(self):
        mix, ref, meta = make_condition(self.vocals, self.backing, self.sample_rate)
        np.testing.assert_allclose(mix, self.vocals + self.backing, atol=3e-8)
        np.testing.assert_array_equal(ref, self.vocals)
        self.assertEqual(meta["task"], "music_vocal_source_separation")
        clean, _, clean_meta = make_condition(self.vocals, None, self.sample_rate, mode="clean")
        np.testing.assert_array_equal(clean, self.vocals)
        self.assertEqual(clean_meta["task"], "clean_vocal_identity_sanity_check")

    def test_noise_snr_and_determinism(self):
        first, reference, meta = make_condition(self.vocals, self.backing, self.sample_rate, mode="noise", snr_db=7.0, seed=3)
        second, _, _ = make_condition(self.vocals, self.backing, self.sample_rate, mode="noise", snr_db=7.0, seed=3)
        changed, _, _ = make_condition(self.vocals, self.backing, self.sample_rate, mode="noise", snr_db=7.0, seed=4)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, changed))
        actual = 10.0 * np.log10(np.sum(reference.astype(np.float64)**2) / np.sum((first.astype(np.float64) - reference)**2))
        self.assertAlmostEqual(actual, 7.0, places=5)
        self.assertAlmostEqual(meta["actual_component_snr_db"], 7.0, places=9)
        self.assertEqual(meta["comparison_group"], "dry_vocal_restoration")

    def test_bleed_gain_and_reported_ratio(self):
        audio, reference, meta = make_condition(self.vocals, self.backing, self.sample_rate, mode="bleed", bleed_db=-20.0)
        np.testing.assert_allclose(audio, reference + self.backing * 0.1, atol=3e-8)
        self.assertAlmostEqual(meta["bleed_amplitude_gain"], 0.1)
        mixture, _, mixture_meta = make_condition(self.vocals, self.backing, self.sample_rate)
        self.assertAlmostEqual(meta["input_target_to_interference_db"] - mixture_meta["input_target_to_interference_db"], 20.0, places=9)
        self.assertEqual(mixture.shape, audio.shape)

    def test_rir_tail_energy_ratio_and_reverb_alignment(self):
        rir = synthetic_rir(self.sample_rate, rt60_seconds=0.1, seed=4, direct_to_reverb_db=6.0)
        self.assertEqual(rir[0], 1.0)
        self.assertAlmostEqual(10.0 * np.log10(rir[0]**2 / np.sum(rir[1:]**2)), 6.0, places=9)
        first, reference, meta = make_condition(self.vocals, None, self.sample_rate, mode="reverb", rt60_seconds=0.1, seed=4)
        second, _, _ = make_condition(self.vocals, None, self.sample_rate, mode="reverb", rt60_seconds=0.1, seed=4)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(reference, self.vocals)
        self.assertEqual(first.shape, self.vocals.shape)
        self.assertFalse(meta["measured_or_physical_room_rir"])
        self.assertFalse(meta["closed_loop_acoustic_feedback_simulated"])

    def test_inputs_and_reference_are_independent(self):
        vocal_original = self.vocals.copy()
        backing_original = self.backing.copy()
        for mode in ["mixture", "clean", "noise", "reverb", "bleed"]:
            result, reference, _ = make_condition(self.vocals, self.backing, self.sample_rate, mode=mode)
            result[:] = 0
            reference[:] = 0
            np.testing.assert_array_equal(self.vocals, vocal_original)
            np.testing.assert_array_equal(self.backing, backing_original)

    def test_no_clipping_or_normalization(self):
        audio, reference, meta = make_condition(np.full((8, 2), 0.9), np.full((8, 2), 0.8), self.sample_rate)
        self.assertGreater(float(audio.max()), 1.0)
        self.assertTrue(meta["peak_exceeds_pcm_unit_range"])
        self.assertFalse(meta["normalization_applied"])
        self.assertAlmostEqual(float(reference.max()), 0.9, places=6)

    def test_invalid_conditions(self):
        for args in [
            {"mode": "mixture", "accompaniment": None},
            {"mode": "noise", "vocals": np.zeros_like(self.vocals)},
            {"mode": "noise", "snr_db": np.nan},
            {"mode": "reverb", "rt60_seconds": 0.0},
            {"mode": "unknown"},
        ]:
            kwargs = {"vocals": self.vocals, "accompaniment": self.backing, "sample_rate": self.sample_rate}
            kwargs.update(args)
            with self.assertRaises(ValueError):
                make_condition(**kwargs)


if __name__ == "__main__":
    unittest.main()
