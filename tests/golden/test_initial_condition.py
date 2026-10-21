"""
Initial condition: today's draw of the initial stiffnesses from (task kind, task,
recorded realization seed, edge count) reproduces what the runs on disk started from.

Auxetic: the draw must equal stiffness_trajectory[0], the pre-update stiffnesses of
step 0 (to ~1 ulp). Allosteric: no local run stores step 0 (every stiffnesses_traj starts at
step 49 or later), so the trainer's draw is checked against the regeneration the
figure code uses instead (the read golden pins its digest).

Run: GOLDEN=1 python -m unittest tests.golden.test_initial_condition
"""
import json
import unittest

import numpy as np

from tests.golden import skip_unless_golden
from tests.golden import sample


@skip_unless_golden
class TestAllostericInitialCondition(unittest.TestCase):
    def test_trainer_draw_matches_figure_regeneration(self):
        from tests.golden.readers import A, _value, allosteric_trainer as T
        for geometry, task, real, why in sample.ALLOSTERIC:
            with self.subTest(sample.allosteric_name(geometry, task, real), why=why):
                d = _value(A.load_allosteric, geometry, task, real)
                meta = json.loads((A._allo_geom_dir(geometry) / f'task_{task}'
                                   / f'realization_{real}' / 'training_meta.json').read_text())
                self.assertEqual((meta['k_min'], meta['k_max']), (T.K_MIN, T.K_MAX))
                drawn = T.realization_rng(int(meta['realization_seed'])).uniform(
                    T.K_MIN, T.K_MAX, size=len(d['edges']))
                np.testing.assert_array_equal(drawn, A.allo_initial_stiffnesses(d))


@skip_unless_golden
class TestAuxeticInitialCondition(unittest.TestCase):
    def test_draw_equals_stored_step_zero(self):
        if not sample.AUXETIC:
            self.skipTest('no auxetic sample chosen yet')
        from tests.golden.readers import A, _value
        from training.src.task_generator import generate_realization_stiffnesses
        for family, task, real, why in sample.AUXETIC:
            aa = _value(A.load_auxetic, task, real, family)
            if aa is None or aa['stiff_traj'] is None:
                continue
            with self.subTest(sample.auxetic_name(family, task, real), why=why):
                seed = A.aux_realization_seed(task, real, family)
                drawn = generate_realization_stiffnesses(task, seed, len(aa['network'].edges))
                # Stored step 0 differs from the raw draw by float rounding (~1 ulp, seen
                # on every sample), so equality is to 1e-13, not bitwise.
                np.testing.assert_allclose(drawn, np.asarray(aa['stiff_traj'][0]),
                                           rtol=1e-13, atol=0)


if __name__ == '__main__':
    unittest.main()
