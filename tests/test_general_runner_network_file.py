"""The general ensemble runner can train on a stored network (--network-file), like the
targeted runner (ADR 0002, issue #2).

Run: python -m unittest tests.test_general_runner_network_file
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from base.stored_network import NETWORK_A_FILE, NETWORK_A_FINGERPRINT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'runners'))
import ensemble_runner as R  # noqa: E402


def trainer(network_file):
    return R.GeneralAuxeticTrainer(
        task_id=0, realization_id=0, realization_seed=1000014, gradient_method='jax',
        network_type='jammed', learning_rate=0.003, n_steps=3, results_dir=None, verbose=False,
        network_file=network_file)


class TestNetworkFile(unittest.TestCase):
    def test_prepare_loads_the_stored_network_without_packing(self):
        t = trainer(NETWORK_A_FILE)
        record = mock.Mock()
        record.resume_state.return_value = None
        with mock.patch.object(R, 'create_auxetic_network',
                               side_effect=AssertionError('packing must not run')):
            t.prepare(record)
        self.assertEqual(len(t.network.edges), 194)
        self.assertEqual(t.extra_hparams(None), {'network_fingerprint': NETWORK_A_FINGERPRINT})

    def test_without_a_network_file_the_meta_is_unchanged(self):
        self.assertEqual(trainer(None).extra_hparams(None), {})

    def test_cli_has_the_option(self):
        with mock.patch.object(sys, 'argv', ['ensemble_runner.py', '--help']), \
                mock.patch('sys.stdout') as out, self.assertRaises(SystemExit):
            R.main()
        printed = ''.join(c.args[0] for c in out.write.call_args_list)
        self.assertIn('--network-file', printed)


if __name__ == '__main__':
    unittest.main()
