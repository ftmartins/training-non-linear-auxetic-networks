"""get_complete_jobs / get_incomplete_jobs read the results directory through the record.

Run: python -m unittest tests.test_campaign_status
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_realization_record_auxetic import write_auxetic_run
from training.src.checkpoint_manager import get_complete_jobs, get_incomplete_jobs


class TestCampaignStatus(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_dir(self, task, real):
        return self.root / f'task_{task:02d}' / f'realization_{real:02d}'

    def test_complete_by_marker_or_loss_reduction_the_rest_incomplete(self):
        write_auxetic_run(self.run_dir(0, 0))                                   # loss falls 10x only
        write_auxetic_run(self.run_dir(0, 1))
        (self.run_dir(0, 1) / 'training_complete_small_loss_jammed.txt').write_text('done\n')
        write_auxetic_run(self.run_dir(1, 0))
        np.save(self.run_dir(1, 0) / 'loss_trajectory_jammed.npy', np.geomspace(1.0, 1e-8, 5))
        kwargs = dict(n_tasks=2, n_realizations=2, results_dir=self.root, network_type='jammed')
        self.assertEqual(get_complete_jobs(**kwargs), [(0, 1), (1, 0)])
        self.assertEqual(get_incomplete_jobs(**kwargs), [(0, 0), (1, 1)])

    def test_missing_directory_is_incomplete_and_outside_the_grid_is_ignored(self):
        write_auxetic_run(self.run_dir(0, 0))
        (self.run_dir(0, 0) / 'training_complete_small_loss_jammed.txt').write_text('done\n')
        write_auxetic_run(self.run_dir(5, 0))
        (self.run_dir(5, 0) / 'training_complete_small_loss_jammed.txt').write_text('done\n')
        kwargs = dict(n_tasks=2, n_realizations=2, results_dir=self.root, network_type='jammed')
        self.assertEqual(get_complete_jobs(**kwargs), [(0, 0)])
        self.assertEqual(get_incomplete_jobs(**kwargs), [(0, 1), (1, 0), (1, 1)])
        self.assertEqual(get_incomplete_jobs(n_tasks=1, n_realizations=1,
                                             results_dir=self.root / 'nowhere'), [(0, 0)])


if __name__ == '__main__':
    unittest.main()
