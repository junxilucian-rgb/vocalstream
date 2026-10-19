"""Protocol checks: isolation, failed-job recovery and artifact preservation."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_project as workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.scratch = workflow.PROJECT / "results" / "workflow_checks"
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.scratch)
        self.root = Path(self.temporary.name)
        self.config = workflow.read_json(workflow.PROJECT / "project_config.json")

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.scratch.resolve()))
        self.temporary.cleanup()

    def test_prepare_without_data_or_site_packages(self):
        # -S disables third-party site packages: preparing a plan needs no ML runtime.
        destination = self.root / "plan"
        subprocess.run([sys.executable, "-S", str(workflow.PROJECT / "run_project.py"), "prepare",
                        "--root", str(self.root / "not_downloaded"), "--out", str(destination)], check=True,
                       capture_output=True, text=True)
        plan = workflow.read_json(destination / "plan.json")
        self.assertEqual(plan["profile"], "pilot")
        self.assertFalse((self.root / "not_downloaded").exists())
        self.assertFalse((destination / "run.json").exists())

    def test_empty_validation_selection_cannot_evaluate_training_songs(self):
        self.config["profiles"]["pilot"]["validation_track_limit"] = 0
        with self.assertRaises(ValueError):
            workflow.validate_config(self.config)
        with self.assertRaises(ValueError):
            workflow.evaluate(None, self.root, {}, {"conditions": []}, [], tracks=[])

    def test_pilot_split_reads_train_only_and_keeps_validation_out(self):
        import numpy as np
        import soundfile as sf
        from vocalstream.data import discover_hq
        validation_name = self.config["validation_tracks"][0]
        for name in (validation_name, "Non-validation training song"):
            folder = self.root / "train" / name
            folder.mkdir(parents=True)
            for stem in ("mixture", "vocals", "drums", "bass", "other"):
                sf.write(folder / f"{stem}.wav", np.zeros((100, 2)), 44100, subtype="FLOAT")
        workflow.write_json(self.root / "_DATASET_INFO.json", {"kind": "MUSDB18-7-compressed-preview", "formal_benchmark": False})
        subsets = []
        def record_discovery(root, subset):
            subsets.append(subset)
            return discover_hq(root, subset)
        with patch("vocalstream.data.discover_hq", side_effect=record_discovery):
            split = workflow.resolve_data(self.root, self.config, "pilot")
        self.assertEqual(subsets, ["train"])
        self.assertEqual(split["validation_tracks"], [validation_name])
        self.assertEqual(split["training_tracks"], ["Non-validation training song"])
        self.assertFalse(split["test_set_accessed"])
        with self.assertRaises(ValueError):
            workflow.resolve_data(self.root, self.config, "development")

    def test_failed_job_retries_without_overwriting_and_completed_job_skips(self):
        runner = workflow.JobRunner(self.root / "run", "fixed-identity", {"fixed": True})
        with self.assertRaises(RuntimeError):
            runner.job("demo", "evaluation", lambda out: [sys.executable, "-c", "raise SystemExit(4)"])
        self.assertEqual(runner.state["jobs"]["demo"]["status"], "failed")
        first_log = self.root / "run" / "demo" / "attempt_01" / "console.log"
        def succeed(out):
            code = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'report.json').write_text('{\"status\": \"complete\"}')"
            return [sys.executable, "-c", code, str(out)]
        artifact = runner.job("demo", "evaluation", succeed)
        self.assertEqual(runner.state["jobs"]["demo"]["attempt"], 2)
        self.assertTrue(first_log.is_file())
        def never_called(out):
            self.fail("A complete unchanged result must not execute again")
        self.assertEqual(runner.job("demo", "evaluation", never_called), artifact)
        artifact.write_text('{"status": "changed"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            runner.job("demo", "evaluation", never_called)

    def test_resume_rejects_changed_context(self):
        workflow.JobRunner(self.root, "identity", {"source": "first"})
        with self.assertRaises(ValueError):
            workflow.JobRunner(self.root, "identity", {"source": "modified"})

    def test_preview_cannot_be_frozen_for_formal_test(self):
        workflow.write_json(self.root / "run.json", {"status": "complete", "context": {
            "plan": {"profile": "pilot"}, "data": {"metadata": {"kind": "MUSDB18-7-compressed-preview"}}}})
        with self.assertRaises(ValueError):
            workflow.freeze(argparse.Namespace(run=self.root, candidate="fft512_h32", reason="pilot result"))
        self.assertFalse((self.root / "frozen_selection.json").exists())

    def test_concurrent_run_lock_is_released_after_exit(self):
        with workflow.exclusive_run(self.root):
            with self.assertRaises(RuntimeError):
                with workflow.exclusive_run(self.root):
                    self.fail("A second writer obtained the lock")
        with workflow.exclusive_run(self.root):
            pass


if __name__ == "__main__":
    unittest.main()
