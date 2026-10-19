"""Protect the pretrained-only project's formal test boundary."""
import argparse
from pathlib import Path
import unittest
from unittest.mock import patch

import run_reference_project as workflow


class ReferenceWorkflowTests(unittest.TestCase):
    def test_test_audio_is_not_inspected_without_frozen_selection(self):
        args = argparse.Namespace(config=workflow.PROJECT/'reference_experiments.json', root=None,
                                  stage='test', frozen=None, out=Path('not_created'), reuse=None)
        with patch.object(workflow, 'dataset') as data:
            with self.assertRaisesRegex(ValueError, 'requires --frozen'):
                workflow.run(args)
            data.assert_not_called()

    def test_pilot_cannot_be_used_as_full_validation_selection(self):
        state={'status':'complete','context':{'stage':'pilot'}}
        args=argparse.Namespace(run=Path('nonexistent_run.json'),reason='pilot-only observation',experiment='bs_hq_amp_o2')
        with patch.object(workflow,'read_json',return_value=state):
            with self.assertRaisesRegex(ValueError,'full 14-song validation'):
                workflow.freeze(args)


if __name__=='__main__':
    unittest.main()
