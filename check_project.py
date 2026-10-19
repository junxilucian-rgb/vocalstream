"""Run deterministic engineering checks; these are not a music benchmark."""
import sys
import tempfile
import unittest
from pathlib import Path

project = Path(__file__).resolve().parent
temp_folder = project / "results" / "check_tmp"
temp_folder.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(temp_folder)
suite = unittest.defaultTestLoader.discover(str(project / "tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
