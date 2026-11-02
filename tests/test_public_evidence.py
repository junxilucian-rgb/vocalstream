"""Portable publication-integrity checks; no model packages or dataset needed."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_public_evidence import EvidenceError, main, verify


class PublicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reports = self.root / "reports"
        (self.reports / "example/jobs").mkdir(parents=True)
        self.report = {"score": 12.5, "input_path": "MUSDB18-HQ/train/song/mixture.wav",
                       "paper": "https://arxiv.org/abs/2309.02612"}
        self.study = {
            "publication_schema_version": 1, "study_id": "example_2026",
            "execution": {"status": "complete", "job_count": 1},
            "protocol": {"configuration": {"expected_jobs": 1}},
            "jobs": {"example_job": {
                "status": "complete", "artifact": "results/example/output/report.json",
                "log": "results/example/console.log", "public_report": "example/jobs/job.json",
            }},
        }
        self.entry = {"path": "example.json", "job_count": 1,
                      "original_study_id": "example_2026"}
        self.write_fixture()

    def write(self, relative, data):
        raw = json.dumps(data).encode("utf-8")
        (self.reports / relative).write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def write_fixture(self):
        digest = self.write("example/jobs/job.json", self.report)
        self.study["jobs"]["example_job"]["public_report_sha256"] = digest
        self.entry["sha256"] = self.write("example.json", self.study)
        self.write("index.json", {"schema_version": 1, "studies": {"example": self.entry}})

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(EvidenceError, pattern):
            verify(self.root)

    def test_complete_fixture_needs_no_original_audio_or_logs(self):
        result = verify(self.root)
        self.assertEqual((result.studies, result.jobs, result.json_files), (1, 1, 3))
        self.assertFalse((self.root / "results").exists())

    def test_corrupt_main_hash_is_rejected(self):
        (self.reports / "example.json").write_bytes(b"{}")
        self.assert_rejected("SHA-256 mismatch")

    def test_corrupt_job_hash_is_rejected(self):
        (self.reports / "example/jobs/job.json").write_bytes(b"{}")
        self.assert_rejected("SHA-256 mismatch")

    def test_missing_job_is_rejected(self):
        (self.reports / "example/jobs/job.json").unlink()
        self.assert_rejected("cannot read evidence file")

    def test_declared_counts_must_all_match(self):
        targets = [self.entry, self.study["execution"], self.study["protocol"]["configuration"]]
        for target in targets:
            key = "expected_jobs" if "expected_jobs" in target else "job_count"
            with self.subTest(key=key, target=repr(target)):
                target[key] = 2
                self.write_fixture()
                self.assert_rejected("declared job count")
                target[key] = 1

    def test_incomplete_execution_or_job_is_rejected(self):
        for record in [self.study["execution"], self.study["jobs"]["example_job"]]:
            with self.subTest(record=repr(record)):
                record["status"] = "running"
                self.write_fixture()
                self.assert_rejected("not complete")
                record["status"] = "complete"

    def test_traversal_is_rejected_after_rehash(self):
        job = self.study["jobs"]["example_job"]
        for key in ["artifact", "log", "public_report"]:
            original = job[key]
            with self.subTest(key=key):
                job[key] = "../outside.json"
                self.write_fixture()
                self.assert_rejected("path must stay")
                job[key] = original

    def test_machine_paths_are_rejected_even_after_rehash(self):
        for value in [r"C:\Users\private\song.wav", "/home/private/song.wav",
                      r"\\server\private\song.wav", "//server/private/song.wav",
                      "input=/Users/private/song.wav", "file:///home/private/song.wav"]:
            with self.subTest(value=value):
                self.report["note"] = value
                self.write_fixture()
                self.assert_rejected("absolute machine path")

    def test_urls_are_not_misclassified_as_machine_paths(self):
        self.report["note"] = "See https://example.org/data/a:b/c and https://arxiv.org/abs/2309.02612"
        self.write_fixture()
        self.assertEqual(verify(self.root).jobs, 1)

    def test_url_does_not_hide_a_separate_machine_path(self):
        self.report["note"] = r"https://example.org/path followed by C:\Users\private"
        self.write_fixture()
        self.assert_rejected("absolute machine path")

    def test_study_identity_must_match_index(self):
        self.entry["original_study_id"] = "different_study"
        self.write_fixture()
        self.assert_rejected("study identity mismatch")

    def test_missing_hash_is_not_treated_as_optional(self):
        del self.entry["sha256"]
        self.write("index.json", {"schema_version": 1, "studies": {"example": self.entry}})
        self.assert_rejected("missing SHA-256")

    def test_null_or_malformed_hash_cannot_disable_verification(self):
        for digest in [None, "", "not-a-sha256", 7]:
            with self.subTest(digest=digest):
                self.entry["sha256"] = digest
                self.write("index.json", {"schema_version": 1, "studies": {"example": self.entry}})
                self.assert_rejected("invalid SHA-256")
        self.write_fixture()
        self.study["jobs"]["example_job"]["public_report_sha256"] = None
        self.entry["sha256"] = self.write("example.json", self.study)
        self.write("index.json", {"schema_version": 1, "studies": {"example": self.entry}})
        self.assert_rejected("invalid SHA-256")

    def test_duplicate_public_reference_cannot_inflate_job_total(self):
        self.study["jobs"]["duplicate"] = dict(self.study["jobs"]["example_job"])
        self.entry["job_count"] = 2
        self.study["execution"]["job_count"] = 2
        self.study["protocol"]["configuration"]["expected_jobs"] = 2
        self.write_fixture()
        self.assert_rejected("duplicate evidence file")

    def test_malformed_json_is_rejected_after_rehash(self):
        raw = b"{not-json}"
        (self.reports / "example.json").write_bytes(raw)
        self.entry["sha256"] = hashlib.sha256(raw).hexdigest()
        self.write("index.json", {"schema_version": 1, "studies": {"example": self.entry}})
        self.assert_rejected("invalid JSON")

    def test_cli_has_nonzero_exit_and_no_traceback_on_corruption(self):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            self.assertEqual(main(["--root", str(self.root)]), 0)
            (self.reports / "example.json").write_bytes(b"{}")
            self.assertEqual(main(["--root", str(self.root)]), 1)
        self.assertIn("1 studies, 1 complete jobs, 3 JSON files", output.getvalue())
        self.assertIn("SHA-256 mismatch", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
