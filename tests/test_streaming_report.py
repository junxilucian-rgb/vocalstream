"""Reject mismatched dependencies and inference policies before report aggregation."""
import copy
from pathlib import Path
import unittest

import numpy as np

from build_streaming_study_report import inside, verify_emissions
from run_streaming_study import study_settings
from vocalstream.streaming import BoundedFutureSeparator


class IdentityWindow:
    sample_rate = 44100
    window_inference_mode = "single"

    def separate(self, window):
        self.last_run = {"model_forward_passes": 1}
        return window.copy(), 0.0


class StreamingReportTests(unittest.TestCase):
    def setUp(self):
        self.setting = study_settings("single")[0]
        stream = BoundedFutureSeparator(IdentityWindow(), context_seconds=2,
                                        emit_seconds=.02, lookahead_seconds=0)
        stream.push(np.zeros((882, 2), dtype=np.float32))
        stream.push(np.zeros((882, 2), dtype=np.float32))
        stream.finish()
        self.report = {"stream": stream.summary(), "emissions": stream.records,
                       "window_inference_mode": "single",
                       "steady_state_quality_intervals_clip_relative_samples": []}

    def test_valid_short_clip_has_explicitly_empty_steady_interval(self):
        self.assertEqual(verify_emissions(self.report, self.setting, 1764), [])

    def test_changed_count_or_dependency_is_rejected(self):
        for key, replacement in (("model_forward_passes", 2),
                                 ("required_real_input_sample_boundary", 883),
                                 ("output_start_sample", 1)):
            with self.subTest(key=key):
                altered = copy.deepcopy(self.report)
                altered["emissions"][0][key] = replacement
                with self.assertRaises(ValueError):
                    verify_emissions(altered, self.setting, 1764)

    def test_mislabeled_policy_cannot_be_scored_as_single(self):
        altered = copy.deepcopy(self.report)
        altered["window_inference_mode"] = "ola"
        with self.assertRaises(ValueError):
            verify_emissions(altered, self.setting, 1764)

    def test_artifact_paths_cannot_escape_declared_run(self):
        root = Path.cwd() / "results" / "report_guard_example"
        self.assertEqual(inside(root / "worker" / "report.json", root),
                         (root / "worker" / "report.json").resolve())
        with self.assertRaises(ValueError):
            inside(root / ".." / "another_run" / "report.json", root)


if __name__ == "__main__":
    unittest.main()
