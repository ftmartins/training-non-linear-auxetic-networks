"""RealizationRecord.iterate: the realizations stored under a data root, in tmp dirs.

Run: python -m unittest tests.test_realization_record_iterate
"""
import tempfile
import unittest
from pathlib import Path

from base.realization_record import RealizationRecord
from tests.test_realization_record import write_allosteric_run
from tests.test_realization_record_auxetic import write_auxetic_run


def ids(records):
    return [(r.cell, r.realization_id) for r in records]


class IterateTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def allo(self, geometry, task, real, **kw):
        g = 'geometry_targeted' if geometry == 'targeted' else f'geometry_{geometry}'
        return write_allosteric_run(self.root / g / f'task_{task}' / f'realization_{real}', **kw)

    def aux(self, task, real, padded=True, **kw):
        t = f'task_{task:02d}' if padded else f'task_{task}'
        return write_auxetic_run(self.root / t / f'realization_{real:02d}', **kw)


class TestAllosteric(IterateTestCase):
    def test_yields_every_realization_targeted_first_then_numeric(self):
        for g, t, r in [(1, 0, 0), ('targeted', 10, 0), (0, 2, 10), (0, 2, 2), ('targeted', 2, 1),
                        (0, 10, 0)]:
            self.allo(g, t, r)
        got = ids(RealizationRecord.iterate(self.root, 'allosteric'))
        self.assertEqual(got, [(('targeted', 2), 1), (('targeted', 10), 0), ((0, 2), 2),
                               ((0, 2), 10), ((0, 10), 0), ((1, 0), 0)])

    def test_records_are_usable_and_know_their_kind(self):
        d = self.allo(0, 1, 4)
        (rec,) = RealizationRecord.iterate(self.root, 'allosteric')
        self.assertEqual((rec.kind, rec.run_dir), ('allosteric', d))
        self.assertEqual(rec.realization_seed, 1000032)

    def test_cell_restricts_to_one_cell(self):
        for g, t, r in [(0, 1, 0), (0, 1, 1), (0, 2, 0), ('targeted', 1, 0)]:
            self.allo(g, t, r)
        self.assertEqual(ids(RealizationRecord.iterate(self.root, 'allosteric', cell=(0, 1))),
                         [((0, 1), 0), ((0, 1), 1)])
        self.assertEqual(ids(RealizationRecord.iterate(self.root, 'allosteric',
                                                       cell=('targeted', 1))), [(('targeted', 1), 0)])
        self.assertEqual(ids(RealizationRecord.iterate(self.root, 'allosteric', cell=(3, 3))), [])

    def test_a_realization_without_files_is_still_yielded(self):
        (self.root / 'geometry_0' / 'task_0' / 'realization_0').mkdir(parents=True)
        (rec,) = RealizationRecord.iterate(self.root, 'allosteric')
        self.assertFalse(rec.has('loss'))

    def test_other_names_are_ignored(self):
        self.allo(0, 0, 0)
        (self.root / 'geometry_0' / 'task_0' / 'realization_notes.txt').write_text('x')
        (self.root / 'geometry_0' / 'task_0' / 'realization_9').write_text('a file, not a dir')
        (self.root / 'geometry_0' / 'task_extra').mkdir()
        (self.root / 'geometry_x' / 'task_0' / 'realization_0').mkdir(parents=True)
        (self.root / 'summary').mkdir()
        self.assertEqual(ids(RealizationRecord.iterate(self.root, 'allosteric')), [((0, 0), 0)])

    def test_missing_root_yields_nothing(self):
        self.assertEqual(list(RealizationRecord.iterate(self.root / 'nope', 'allosteric')), [])


class TestAuxetic(IterateTestCase):
    def test_yields_every_realization_in_numeric_order(self):
        for t, r in [(10, 0), (2, 11), (2, 3), (0, 1)]:
            self.aux(t, r)
        self.assertEqual(ids(RealizationRecord.iterate(self.root, 'auxetic')),
                         [(0, 1), (2, 3), (2, 11), (10, 0)])

    def test_padded_and_unpadded_task_dirs(self):
        self.aux(1, 0)
        self.aux(7, 2, padded=False)
        got = list(RealizationRecord.iterate(self.root, 'auxetic'))
        self.assertEqual(ids(got), [(1, 0), (7, 2)])
        self.assertEqual(got[1].run_dir, self.root / 'task_7' / 'realization_02')

    def test_records_read_with_the_requested_layout(self):
        self.aux(0, 0, suffix='')
        (rec,) = RealizationRecord.iterate(self.root, 'auxetic')
        self.assertEqual(rec.realization_seed, 1010001)           # jammed falls back to plain
        (rec,) = RealizationRecord.iterate(self.root, 'auxetic', network_type='lattice')
        self.assertFalse(rec.has('loss'))

    def test_cell_restricts_to_one_task(self):
        for t, r in [(1, 0), (1, 1), (2, 0)]:
            self.aux(t, r)
        self.assertEqual(ids(RealizationRecord.iterate(self.root, 'auxetic', cell=1)),
                         [(1, 0), (1, 1)])

    def test_other_names_are_ignored(self):
        self.aux(0, 0)
        (self.root / 'task_00' / 'realization_summary.json').write_text('{}')
        (self.root / 'task_final').mkdir()
        (self.root / 'task_01.log').write_text('x')
        self.assertEqual(ids(RealizationRecord.iterate(self.root, 'auxetic')), [(0, 0)])

    def test_missing_root_yields_nothing(self):
        self.assertEqual(list(RealizationRecord.iterate(self.root / 'nope', 'auxetic')), [])


class TestLocateKnowsItsCell(IterateTestCase):
    def test_located_records_carry_cell_and_realization_id(self):
        self.allo(2, 3, 4)
        self.aux(5, 6)
        rec = RealizationRecord.locate(self.root, 'allosteric', (2, 3), 4)
        self.assertEqual((rec.cell, rec.realization_id), ((2, 3), 4))
        rec = RealizationRecord.locate(self.root, 'auxetic', 5, 6)
        self.assertEqual((rec.cell, rec.realization_id), (5, 6))

    def test_iterate_and_locate_agree_on_the_run_dir(self):
        self.allo('targeted', 0, 3)
        self.aux(7, 2, padded=False)
        for kind, cell, real in (('allosteric', ('targeted', 0), 3), ('auxetic', 7, 2)):
            (it,) = RealizationRecord.iterate(self.root, kind)
            self.assertEqual(it.run_dir, RealizationRecord.locate(self.root, kind, cell, real).run_dir)


if __name__ == '__main__':
    unittest.main()
