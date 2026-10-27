"""base.initial_condition reproduces the pre-refactor draws bit for bit."""
import unittest

import numpy as np

from base import initial_condition as ic
from base.config import STIFFNESS_LOG_MIN, STIFFNESS_LOG_MAX


class TestInitialCondition(unittest.TestCase):
    def test_auxetic_gradient_log_uniform_below_task_20(self):
        want = np.exp(np.random.RandomState(1234).uniform(
            STIFFNESS_LOG_MIN, STIFFNESS_LOG_MAX, size=50))
        np.testing.assert_array_equal(ic.auxetic_gradient(19, 1234, 50), want)

    def test_auxetic_gradient_linear_uniform_from_task_20(self):
        want = np.random.RandomState(1234).uniform(
            np.exp(STIFFNESS_LOG_MIN), np.exp(STIFFNESS_LOG_MAX), size=50)
        np.testing.assert_array_equal(ic.auxetic_gradient(20, 1234, 50), want)

    def test_contrastive_offsets_seed(self):
        want = np.random.RandomState(3_000_000 + 77).uniform(0.5, 4.0, size=30)
        np.testing.assert_array_equal(ic.contrastive(77, 30, 0.5, 4.0), want)

    def test_task_generator_delegates(self):
        from training.src.task_generator import generate_realization_stiffnesses as g
        np.testing.assert_array_equal(g(3, 9, 20), ic.auxetic_gradient(3, 9, 20))


if __name__ == '__main__':
    unittest.main()
