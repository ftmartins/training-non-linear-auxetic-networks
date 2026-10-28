"""
Realization record: everything stored on disk for one realization, read through one
task-independent interface (ADR 0003). This module is the only code that knows run-directory
file names and layouts; callers ask for domain parts (network, task, loss, stiffness
trajectory, resume state, ...) and never open files themselves.

    rec = RealizationRecord.open(run_dir)                  # task kind detected from its files
    rec = RealizationRecord.locate(root, 'allosteric', ('targeted', 0), 3)
    rec.loss.training, rec.network.edges, rec.resume_state()
    for rec in RealizationRecord.iterate(root, 'auxetic'):  # rec.cell, rec.realization_id
        ...

Parts are read on first access and cached. A part the run did not store raises
MissingRecordPart (a FileNotFoundError), never returns None; rec.has(part) asks first.
Arrays come back exactly as stored (no abs, no casts), so callers that transform them keep
doing so themselves.

Best-loss stiffnesses (K*, ADR 0004) are read only here, as rec.best_loss_stiffnesses and
rec.best_loss_step. They raise BestLossUndetermined rather than fall back to anything else,
and a sweep whose after cost Hessian is not at K* raises StaleSweep.

Trainers write through the record too (write_network, write_task, save_checkpoint,
save_final, wipe), so the layout lives only here.

Each on-disk layout is a private reader class behind the record. Layouts so far:
  * allosteric (allosteric_trainer, the _aug tree): nodes / incidence_matrix / eq_lengths,
    tasks.txt (geometry_seed, strain_output2, strain_output), mse1 / mse2 per step,
    stiffnesses.npy (current), stiffnesses_ckpt.npy + ckpt_step.txt (last NaN-free state),
    stiffnesses_traj(_steps).npy, best_stiffnesses.npy + best_combined_mse.txt,
    training_meta.json, timestep_sweep.npz.
  * auxetic (targeted and general ensemble runners, checkpoint_manager): task_NN /
    realization_NN directories (unpadded task_N too), file names suffixed by network type
    (<stem>_jammed.<ext>, <stem>_lattice.<ext>) or plain (<stem>.<ext>, the older jammed
    runs): final_network.pkl, boundary_nodes.json, task_config.json, loss_trajectory.npy
    (training loss only), stiffness_trajectory.npy (one row per step), history.pkl
    (per-step positions), checkpoint.pkl, training_meta.json, timestep_sweep.npz.
    Open with network_type='jammed' (default; falls back to plain names), 'lattice', or
    None (plain names only).
  * uniform (ADR 0006, new realizations of either task kind): record.json marks it; plain
    names, no network-type suffix: network.npz, task.json, loss.npy, stiffness.npz,
    current.npy, best.npz, resume.npz, positions.npy, training_meta.json, sweep files.
    A fresh run directory gets this layout when NMN_RECORD_LAYOUT=uniform (default: legacy,
    i.e. the layouts above); a directory that already holds a run keeps the layout it
    started in.
"""
import json
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np

KINDS = ('allosteric', 'auxetic')
PARTS = ('config', 'realization_seed', 'network', 'boundary', 'task', 'loss',
         'stiffness_trajectory', 'positions_trajectory', 'current_stiffnesses', 'best_so_far',
         'best_loss_stiffnesses', 'best_loss_step', 'sweep')

# The allosteric trainer saved its best stiffnesses post-update until commit 3c24f68
# (2026-08-10). Runs first started before the end of that day cannot give K* (ADR 0004).
BEST_SO_FAR_PRE_UPDATE_SINCE = '2026-08-11'


class NotARealization(FileNotFoundError):
    """The directory does not exist or holds no known realization layout."""


class MissingRecordPart(FileNotFoundError):
    """The realization did not store the requested part."""


class BestLossUndetermined(ValueError):
    """The stored data do not determine the best-loss stiffnesses exactly (ADR 0004)."""


class StaleSweep(ValueError):
    """The post-training sweep's after cost Hessian was not evaluated at K* (sweeps made
    before 2026-09-10); it must be recomputed, not read (ADR 0004)."""


@dataclass(frozen=True)
class Network:
    nodes: np.ndarray             # (n_nodes, d)
    incidence_matrix: np.ndarray  # (n_edges, n_nodes), signed
    eq_lengths: np.ndarray        # (n_edges,)

    @cached_property
    def edges(self):
        """(n_edges, 2) endpoint indices, read off the incidence matrix."""
        return incidence_to_edges(self.incidence_matrix)


@dataclass(frozen=True)
class AuxeticNetwork:
    positions: np.ndarray      # (n_nodes, 2)
    edges: np.ndarray          # (n_edges, 2)
    rest_lengths: np.ndarray   # (n_edges,)


@dataclass(frozen=True)
class AuxeticTask:
    config: dict   # as stored (task_seed, packing_seed, n_nodes, compression_strains, ...)

    @property
    def compression_strains(self):
        return self.config['compression_strains']

    @property
    def target_poisson_ratios(self):
        return self.config['target_poisson_ratios']


@dataclass(frozen=True)
class AllostericTask:
    geometry_seed: float
    strain_output: float    # subtask 1 target
    strain_output2: float   # subtask 2 target


class Loss:
    """Recorded loss history. per_subtask is (n_steps, n_subtasks) in stored task order; the
    training loss is its mean over subtasks, for every task kind. Runs that stored only the
    training loss (existing auxetic runs) give Loss(training=...), whose per_subtask raises
    MissingRecordPart."""

    def __init__(self, per_subtask=None, *, training=None):
        if (per_subtask is None) == (training is None):
            raise ValueError('give per_subtask or training, not both')
        self._per_subtask = per_subtask
        self._training = training

    @property
    def per_subtask(self):
        if self._per_subtask is None:
            raise MissingRecordPart('per-subtask loss was not recorded')
        return self._per_subtask

    @property
    def training(self):
        if self._per_subtask is None:
            return self._training
        return self._per_subtask.mean(axis=1)

    @property
    def n_steps(self):
        return len(self._training if self._per_subtask is None else self._per_subtask)


@dataclass(frozen=True)
class StiffnessTrajectory:
    steps: np.ndarray         # loss-step index each snapshot pairs with
    stiffnesses: np.ndarray   # (n_snapshots, n_edges)


@dataclass(frozen=True)
class BestSoFar:
    """The trainer's running lowest-loss state, carried across resumes. Not K*: K* is the
    finished realization's best-loss stiffnesses, defined and validated per ADR 0004."""
    stiffnesses: np.ndarray
    training_loss: float


@dataclass(frozen=True)
class ResumeState:
    """Step reached, stiffnesses and loss history to continue training from.
    source: 'current' (latest saved stiffnesses) or 'checkpoint' (rolled back to the last
    NaN-free checkpoint because the latest stiffnesses hold NaN)."""
    step: int
    stiffnesses: np.ndarray
    loss: Loss
    source: str
    trainer_state: dict = None   # anything else the trainer saved to resume (auxetic: the
                                 # checkpoint's network, history and task_config)


def incidence_to_edges(incidence_matrix):
    """(n_edges, 2) endpoint indices from a signed incidence matrix."""
    incidence_matrix = np.asarray(incidence_matrix)
    edges = np.zeros((incidence_matrix.shape[0], 2), dtype=int)
    for e, row in enumerate(incidence_matrix):
        cols = np.where(np.abs(row) > 0.5)[0]
        edges[e] = cols[0], cols[1]
    return edges


class RealizationRecord:
    """One realization's record. Build with open() or locate(), not the constructor."""

    def __init__(self, run_dir, reader, cell=None, realization_id=None):
        self.run_dir = Path(run_dir)
        self.kind = reader.kind
        self._reader = reader
        self.cell = cell                      # None for a record opened from a bare directory
        self.realization_id = realization_id

    @classmethod
    def open(cls, run_dir, kind=None, **layout):
        """Record of the realization stored in run_dir. With kind=None the task kind is
        detected from the files present; passing kind skips detection, so a directory a
        trainer is about to fill (nothing written yet) can be opened too. layout: auxetic
        network_type (see the module docstring)."""
        run_dir = Path(run_dir)
        if kind is None:
            if not run_dir.is_dir():
                raise NotARealization(f'{run_dir}: no such directory')
            kind = _detect_kind(run_dir)
        elif kind not in _READERS:
            raise ValueError(f'unknown task kind {kind!r}; known: {", ".join(KINDS)}')
        return cls(run_dir, _reader_for(kind, run_dir, layout))

    @classmethod
    def locate(cls, root, kind, cell, realization_id, must_exist=True, **layout):
        """Record of realization `realization_id` in `cell` under data root `root`.
        Allosteric cell = (geometry_id, task_id); geometry_id may be 'targeted'.
        Auxetic cell = task_id."""
        if kind not in _READERS:
            raise ValueError(f'unknown task kind {kind!r}; known: {", ".join(KINDS)}')
        run_dir = _READERS[kind].run_dir(Path(root), cell, realization_id)
        if must_exist and not run_dir.is_dir():
            raise NotARealization(f'{run_dir}: no such directory')
        return cls(run_dir, _reader_for(kind, run_dir, layout), cell, realization_id)

    @classmethod
    def iterate(cls, root, kind, cell=None, **layout):
        """Records of every realization stored under data root `root`, cell by cell in
        numeric order (allosteric: geometry 'targeted' first, then numbered; then task, then
        realization). Only realizations present on disk are yielded, whatever files they hold:
        ask rec.has(part) before reading. rec.cell and rec.realization_id say where each sits.
        cell restricts to one cell (see locate()); a root that does not exist yields nothing."""
        if kind not in _READERS:
            raise ValueError(f'unknown task kind {kind!r}; known: {", ".join(KINDS)}')
        for found_cell, realization_id, run_dir in _READERS[kind].iterate(Path(root), cell):
            yield cls(run_dir, _reader_for(kind, run_dir, layout), found_cell, realization_id)

    def has(self, part):
        """Whether the realization stored `part` (one of PARTS)."""
        if part not in PARTS:
            raise ValueError(f'unknown record part {part!r}; known: {", ".join(PARTS)}')
        try:
            getattr(self, part)
        except MissingRecordPart:
            return False
        return True

    @cached_property
    def config(self):
        """The recorded training configuration (dict), e.g. solver, k_min, k_max."""
        return self._reader.config()

    @cached_property
    def realization_seed(self):
        """The recorded realization seed (authoritative; seed tables are not)."""
        cfg = self.config
        if 'realization_seed' not in cfg:
            raise MissingRecordPart(f'{self.run_dir}: no realization_seed recorded')
        return int(cfg['realization_seed'])

    @cached_property
    def network(self):
        return self._reader.network()

    @cached_property
    def boundary(self):
        """Boundary node indices fixed when the network was built (auxetic):
        dict of 'top' / 'bottom' / 'left' / 'right' -> int array."""
        return self._reader.boundary()

    @cached_property
    def task(self):
        return self._reader.task()

    @cached_property
    def loss(self):
        return self._reader.loss()

    @cached_property
    def stiffness_trajectory(self):
        return self._reader.stiffness_trajectory()

    @cached_property
    def positions_trajectory(self):
        """Per-step relaxed positions the loss was measured at (auxetic history), a list
        of (n_nodes, 2) arrays."""
        return self._reader.positions_trajectory()

    @cached_property
    def current_stiffnesses(self):
        """Latest saved stiffnesses (may hold NaN if the run diverged after its last save)."""
        return self._reader.current_stiffnesses()

    @cached_property
    def best_so_far(self):
        return self._reader.best_so_far()

    @property
    def best_loss_stiffnesses(self):
        """K*: the pre-update stiffnesses at the step with the lowest training loss, earliest
        on ties, NaN losses excluded (ADR 0004)."""
        return self._best_loss[1]

    @property
    def best_loss_step(self):
        """Loss-step index K* was measured at."""
        return self._best_loss[0]

    @cached_property
    def _best_loss(self):
        def undetermined(why):
            return BestLossUndetermined(f'{self.run_dir}: {why}')

        loss = self.loss.training
        if loss.size == 0 or np.all(np.isnan(loss)):
            raise undetermined('no finite training loss')
        step = int(np.nanargmin(loss))
        return step, self._reader.stiffnesses_at_best_loss(self, step, undetermined)

    @cached_property
    def sweep(self):
        """Post-training sweep arrays (dict of name -> array). Raises StaleSweep when its
        after cost Hessian was not evaluated at K*, and BestLossUndetermined when K* itself
        is undetermined or missing."""
        sweep = self._reader.sweep()
        try:
            at_k_star = self._reader.sweep_after_hessian_at_k_star(self, sweep)
        except MissingRecordPart as e:
            raise BestLossUndetermined(f'{self.run_dir}: has a sweep but no K* ({e})') from e
        if not at_k_star:
            raise StaleSweep(f'{self.run_dir}: sweep after cost Hessian is not at K*; '
                             f'recompute the sweep')
        return sweep

    def sweep_files(self):
        """(sweep arrays path, sweep meta path) this layout stores a post-training sweep in;
        the sweep scripts write them, rec.sweep reads the first."""
        npz, meta = self._reader.sweep_file_names()
        return self.run_dir / npz, self.run_dir / meta

    def resume_state(self):
        """ResumeState to continue training from, or None when there is nothing usable to
        resume (no saved stiffnesses, or NaN stiffnesses and no clean checkpoint)."""
        return self._reader.resume_state()

    # ── Write side: every write goes through these, in the realization's own layout ──

    def start(self):
        """Mark the realization's directory as started (creates it). The lifecycle calls this
        before anything else is written, so a run killed early still has its layout."""
        self._reader.start()

    def refresh(self):
        """Forget parts read so far, so the next access sees what is on disk now."""
        for name in [*PARTS, '_best_loss']:
            self.__dict__.pop(name, None)

    def wipe(self):
        """Delete everything stored for this realization (auxetic: only this network type's
        files; jammed and lattice runs share a directory)."""
        self._reader.wipe()
        self.refresh()

    def meta_file_name(self, stem):
        """File name this layout gives a run_provenance record ('training_meta',
        'run_provenance'), for training.src.run_provenance to write."""
        return self._reader.meta_file_name(stem)

    # Auxetic writes (the ensemble runners and their trainers).

    def save_results(self, history, network, task_config, boundary=None):
        """Trajectories, final network, boundary nodes and task config from a trainer
        history and network."""
        self._reader.save_results(history, network, task_config, boundary)
        self.refresh()

    def save_training_checkpoint(self, step, history, network, task_config, task_id,
                                 realization_id):
        """The auxetic resume state: everything the trainer needs to continue at `step`."""
        self._reader.save_training_checkpoint(step, history, network, task_config, task_id,
                                              realization_id)
        self.refresh()

    def drop_checkpoint(self):
        self._reader.drop_checkpoint()
        self.refresh()

    def has_marker(self, name):
        """Whether the marker file `name` (e.g. 'training_complete_small_loss') exists."""
        return self._reader.path(name, 'txt').exists()

    def mark_complete(self):
        self._reader.mark_complete()

    def mark_complete_small_loss(self, reduction_ratio=None):
        self._reader.mark_complete_small_loss(reduction_ratio)

    def write_network(self, network):
        self._reader.write_network(network)
        self.refresh()

    def write_task(self, task):
        self._reader.write_task(task)
        self.refresh()

    def save_checkpoint(self, step, current_stiffnesses, loss, measured_stiffnesses, best=None):
        """Save training progress at `step` (steps completed). current_stiffnesses is the
        post-update state training continues from; measured_stiffnesses the pre-update
        state the last loss entry was measured at, appended to the stiffness trajectory;
        best (a BestSoFar) only when it changed since the last checkpoint."""
        self._reader.save_checkpoint(step, current_stiffnesses, loss, measured_stiffnesses, best)
        self.refresh()

    def save_final(self, current_stiffnesses, loss, best):
        """Save the state an invocation ends with. best is a BestSoFar, or None when the run
        never had one."""
        self._reader.save_final(current_stiffnesses, loss, best)
        self.refresh()


def _fits(traj, n_steps):
    """Whether a stiffness trajectory pairs with an n_steps loss trajectory: one snapshot per
    step index, indices in range and increasing. A step saved twice in a row (a checkpoint
    and an early stop in the same iteration, or a resume) is fine if both snapshots agree;
    steps going backwards mean two jobs appended to one run directory."""
    steps, k = traj.steps, traj.stiffnesses
    if len(steps) != len(k):
        return False
    if len(steps) == 0:
        return True
    if steps[0] < 0 or steps[-1] >= n_steps:
        return False
    gaps = np.diff(steps)
    if np.any(gaps < 0):
        return False
    return all(np.array_equal(k[i], k[i + 1]) for i in np.nonzero(gaps == 0)[0])


def _detect_kind(run_dir):
    marker = run_dir / _UniformReader.MARKER
    if marker.exists():
        return json.loads(marker.read_text())['kind']
    for kind in KINDS:
        if _READERS[kind].recognises(run_dir):
            return kind
    raise NotARealization(f'{run_dir}: no known realization layout')


def _numbered(parent, prefix):
    """(number, directory) of the subdirectories of `parent` named <prefix>_<integer>, in
    numeric order; nothing when `parent` is not a directory."""
    if not parent.is_dir():
        return []
    pattern = re.compile(rf'{prefix}_(\d+)')
    found = [(int(m.group(1)), p) for p in parent.iterdir()
             if p.is_dir() and (m := pattern.fullmatch(p.name))]
    return sorted(found, key=lambda item: item[0])


def _existing(path):
    if not path.exists():
        raise MissingRecordPart(str(path))
    return path


def _load(path):
    return np.load(_existing(path))


class _AllostericReader:
    kind = 'allosteric'

    def __init__(self, run_dir):
        self.d = Path(run_dir)

    @staticmethod
    def run_dir(root, cell, realization_id):
        geometry_id, task_id = cell
        return root / f'geometry_{geometry_id}' / f'task_{task_id}' / f'realization_{realization_id}'

    @staticmethod
    def iterate(root, cell=None):
        geometries = [('targeted', root / 'geometry_targeted')] if (root / 'geometry_targeted').is_dir() else []
        geometries += _numbered(root, 'geometry')
        for geometry_id, geometry_dir in geometries:
            for task_id, task_dir in _numbered(geometry_dir, 'task'):
                if cell is not None and cell != (geometry_id, task_id):
                    continue
                for realization_id, run_dir in _numbered(task_dir, 'realization'):
                    yield (geometry_id, task_id), realization_id, run_dir

    @staticmethod
    def recognises(run_dir):
        return (run_dir / 'tasks.txt').exists() or (run_dir / 'mse1.npy').exists()

    def start(self):
        self.d.mkdir(parents=True, exist_ok=True)

    def config(self):
        return json.loads(_existing(self.d / 'training_meta.json').read_text())

    def boundary(self):
        raise MissingRecordPart(f'{self.d}: allosteric runs store no boundary nodes')

    def positions_trajectory(self):
        raise MissingRecordPart(f'{self.d}: allosteric runs store no per-step positions')

    def network(self):
        return Network(nodes=_load(self.d / 'nodes.npy'),
                       incidence_matrix=_load(self.d / 'incidence_matrix.npy'),
                       eq_lengths=_load(self.d / 'eq_lengths.npy'))

    def task(self):
        geometry_seed, strain_output2, strain_output = (
            float(v) for v in np.loadtxt(_existing(self.d / 'tasks.txt')))
        return AllostericTask(geometry_seed, strain_output, strain_output2)

    def loss(self):
        return self._loss(_load(self.d / 'mse1.npy'), _load(self.d / 'mse2.npy'))

    def _loss(self, mse1, mse2):
        if len(mse1) != len(mse2):
            raise ValueError(f'{self.d}: mse1 has {len(mse1)} steps, mse2 has {len(mse2)}')
        return Loss(np.column_stack([mse1, mse2]))

    def stiffness_trajectory(self):
        return StiffnessTrajectory(steps=_load(self.d / 'stiffnesses_traj_steps.npy'),
                                   stiffnesses=_load(self.d / 'stiffnesses_traj.npy'))

    def current_stiffnesses(self):
        return _load(self.d / 'stiffnesses.npy')

    def best_so_far(self):
        return BestSoFar(_load(self.d / 'best_stiffnesses.npy'),
                         float(np.loadtxt(_existing(self.d / 'best_combined_mse.txt'))))

    def sweep(self):
        with np.load(_existing(self.d / 'timestep_sweep.npz'), allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    @staticmethod
    def sweep_file_names():
        return 'timestep_sweep.npz', 'timestep_sweep_meta.json'

    @staticmethod
    def sweep_after_hessian_at_k_star(record, sweep):
        """2026-09-10+ sweeps record the stiffnesses their after Hessian was taken at."""
        at = sweep.get('cost_hessian_after_stiffness')
        return at is not None and np.array_equal(
            np.asarray(at, float), np.asarray(record.best_loss_stiffnesses, float))

    def stiffnesses_at_best_loss(self, record, step, undetermined):
        """The trainer's stored best state, accepted only if the stored data pin it to `step`."""
        loss = record.loss.training
        best = record.best_so_far
        if best.training_loss != loss[step]:
            raise undetermined(f'stored best loss {best.training_loss!r} is not the lowest '
                               f'training loss {loss[step]!r} (step {step})')
        traj = record.stiffness_trajectory
        if not _fits(traj, len(loss)):
            raise undetermined('stiffness trajectory does not fit the loss trajectory')
        snapshot = np.nonzero(traj.steps == step)[0]
        if len(snapshot) and not np.array_equal(traj.stiffnesses[snapshot[0]], best.stiffnesses):
            raise undetermined(f'stored best stiffnesses differ from the snapshot at step {step}')
        started = self.first_started()
        if started < BEST_SO_FAR_PRE_UPDATE_SINCE:
            raise undetermined(f'first started {started}, before best stiffnesses were saved '
                               f'pre-update')
        return best.stiffnesses

    def first_started(self):
        """ISO timestamp of the earliest recorded invocation (run_provenance.json)."""
        path = self.d / 'run_provenance.json'
        if not path.exists():
            raise BestLossUndetermined(f'{self.d}: no run_provenance.json, so the run start '
                                       f'date is unknown')
        return min(entry['timestamp'] for entry in json.loads(path.read_text()))

    def resume_state(self):
        stiff_path = self.d / 'stiffnesses.npy'
        if not stiff_path.exists():
            return None
        # A run killed before its first loss save resumes from step 0 with no history.
        mse1 = np.load(self.d / 'mse1.npy') if (self.d / 'mse1.npy').exists() else np.array([])
        mse2 = np.load(self.d / 'mse2.npy') if (self.d / 'mse2.npy').exists() else np.array([])
        stiff = np.load(stiff_path)
        if not np.any(np.isnan(stiff)):
            return ResumeState(len(mse1), stiff, self._loss(mse1, mse2), 'current')
        ckpt_path, step_path = self.d / 'stiffnesses_ckpt.npy', self.d / 'ckpt_step.txt'
        if ckpt_path.exists() and step_path.exists():
            step = int(np.loadtxt(step_path))
            return ResumeState(step, np.load(ckpt_path),
                               self._loss(mse1[:step], mse2[:step]), 'checkpoint')
        return None

    # Writes, byte-for-byte what allosteric_trainer wrote before the record existed.

    def wipe(self):
        shutil.rmtree(self.d)
        self.d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def meta_file_name(stem):
        """File name of a run_provenance record (training_meta, run_provenance)."""
        return f'{stem}.json'

    def write_network(self, network):
        np.save(self.d / 'nodes.npy', network.nodes)
        np.save(self.d / 'incidence_matrix.npy', network.incidence_matrix)
        np.save(self.d / 'eq_lengths.npy', network.eq_lengths)

    def write_task(self, task):
        np.savetxt(self.d / 'tasks.txt',
                   [task.geometry_seed, task.strain_output2, task.strain_output])

    def _save_loss(self, loss):
        np.save(self.d / 'mse1.npy', np.ascontiguousarray(loss.per_subtask[:, 0]))
        np.save(self.d / 'mse2.npy', np.ascontiguousarray(loss.per_subtask[:, 1]))

    def _save_best(self, best):
        np.save(self.d / 'best_stiffnesses.npy', best.stiffnesses)
        np.savetxt(self.d / 'best_combined_mse.txt', [best.training_loss])

    def save_checkpoint(self, step, current_stiffnesses, loss, measured_stiffnesses, best):
        np.save(self.d / 'stiffnesses.npy', current_stiffnesses)
        self._save_loss(loss)
        if not np.any(np.isnan(current_stiffnesses)):
            np.save(self.d / 'stiffnesses_ckpt.npy', current_stiffnesses)
            np.savetxt(self.d / 'ckpt_step.txt', [step], fmt='%d')
        if best is not None:
            self._save_best(best)
        # The snapshot pairs with its own loss index (the entry just measured), not `step`.
        traj_path, steps_path = self.d / 'stiffnesses_traj.npy', self.d / 'stiffnesses_traj_steps.npy'
        if traj_path.exists() and steps_path.exists():
            traj = np.vstack([np.load(traj_path), measured_stiffnesses])
            steps = np.append(np.load(steps_path), loss.n_steps - 1)
        else:
            traj = measured_stiffnesses[np.newaxis, :]
            steps = np.array([loss.n_steps - 1])
        np.save(traj_path, traj)
        np.save(steps_path, steps)

    def save_final(self, current_stiffnesses, loss, best):
        np.save(self.d / 'stiffnesses.npy', current_stiffnesses)
        if best is None:
            # As before the record: a run that never had a best state stores None and inf.
            best = BestSoFar(None, np.inf)
        np.save(self.d / 'best_stiffnesses.npy', best.stiffnesses)
        self._save_loss(loss)
        np.savetxt(self.d / 'best_combined_mse.txt', [best.training_loss])


class _AuxeticReader:
    kind = 'auxetic'
    _MARKERS = ('final_network', 'checkpoint', 'loss_trajectory', 'training_meta',
                'training_complete', 'training_complete_small_loss')

    def __init__(self, run_dir, network_type='jammed'):
        self.d = Path(run_dir)
        self.network_type = network_type

    @staticmethod
    def run_dir(root, cell, realization_id):
        padded = root / f'task_{cell:02d}' / f'realization_{realization_id:02d}'
        unpadded = root / f'task_{cell}' / f'realization_{realization_id:02d}'
        return unpadded if not padded.exists() and unpadded.exists() else padded

    @staticmethod
    def iterate(root, cell=None):
        for task_id, task_dir in _numbered(root, 'task'):
            if cell is not None and cell != task_id:
                continue
            for realization_id, run_dir in _numbered(task_dir, 'realization'):
                yield task_id, realization_id, run_dir

    @classmethod
    def recognises(cls, run_dir):
        return any(any(run_dir.glob(f'{m}*')) for m in cls._MARKERS)

    def start(self):
        self.d.mkdir(parents=True, exist_ok=True)

    def path(self, stem, ext):
        """<stem>_<network_type>.<ext>; plain <stem>.<ext> for network_type None, and as the
        fallback for 'jammed' (plain names are the older jammed runs)."""
        plain = self.d / f'{stem}.{ext}'
        if self.network_type is None:
            return plain
        suffixed = self.d / f'{stem}_{self.network_type}.{ext}'
        if self.network_type == 'jammed' and not suffixed.exists() and plain.exists():
            return plain
        return suffixed

    def _pickle(self, stem):
        with open(_existing(self.path(stem, 'pkl')), 'rb') as f:
            return pickle.load(f)

    def config(self):
        return json.loads(_existing(self.path('training_meta', 'json')).read_text())

    def _final_network(self):
        return self._pickle('final_network')

    def network(self):
        net = self._final_network()
        return AuxeticNetwork(net['positions'], net['edges'], net['rest_lengths'])

    def boundary(self):
        nodes = json.loads(_existing(self.path('boundary_nodes', 'json')).read_text())
        return {k: np.asarray(v, dtype=int) for k, v in nodes.items()}

    def task(self):
        config = json.loads(_existing(self.path('task_config', 'json')).read_text())
        if config is None:   # some general-family files hold a bare `null`
            raise MissingRecordPart(f'{self.d}: task_config is null')
        return AuxeticTask(config)

    def loss(self):
        return Loss(training=_load(self.path('loss_trajectory', 'npy')))

    def stiffness_trajectory(self):
        k = _load(self.path('stiffness_trajectory', 'npy'))
        return StiffnessTrajectory(steps=np.arange(len(k)), stiffnesses=k)

    def positions_trajectory(self):
        return self._pickle('history')['positions']

    def current_stiffnesses(self):
        return self._final_network()['stiffnesses']

    def best_so_far(self):
        raise MissingRecordPart(f'{self.d}: auxetic runners store no best-so-far state')

    def sweep(self):
        with np.load(_existing(self.path('timestep_sweep', 'npz')), allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    def sweep_file_names(self):
        return (self._write_path('timestep_sweep', 'npz').name,
                self._write_path('timestep_sweep_meta', 'json').name)

    def resume_state(self):
        """The checkpoint, or None when there is none or it cannot be unpickled."""
        path = self.path('checkpoint', 'pkl')
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                ckpt = pickle.load(f)
        except Exception as e:  # noqa: BLE001 - a truncated checkpoint means a fresh start
            print(f'Warning: Failed to load checkpoint {path}: {e}')
            return None
        return ResumeState(ckpt['current_step'], ckpt['network']['stiffnesses'],
                           Loss(training=np.asarray(ckpt['history'].get('loss', []), float)),
                           'checkpoint', trainer_state=ckpt)

    @staticmethod
    def sweep_after_hessian_at_k_star(record, sweep):
        """The sweep takes its after Hessian at the last selected step, t_indices[-1]."""
        return int(np.asarray(sweep['t_indices'])[-1]) == record.best_loss_step

    def stiffnesses_at_best_loss(self, record, step, undetermined):
        """Row `step` of the stiffness trajectory, which stores the stiffnesses each step's
        loss was measured at, one row per step."""
        traj = record.stiffness_trajectory
        if len(traj.stiffnesses) != record.loss.n_steps:
            raise undetermined(f'stiffness trajectory has {len(traj.stiffnesses)} rows for '
                               f'{record.loss.n_steps} loss steps')
        return traj.stiffnesses[step]


    # Writes, byte-for-byte what checkpoint_manager wrote before the record existed: always
    # the network-type-suffixed name (never the plain fallback reads accept).

    def _write_path(self, stem, ext):
        if self.network_type is None:
            return self.d / f'{stem}.{ext}'
        return self.d / f'{stem}_{self.network_type}.{ext}'

    def meta_file_name(self, stem):
        return self._write_path(stem, 'json').name

    def wipe(self):
        if not self.d.exists():
            return
        suffix = f'_{self.network_type}'
        for f in self.d.iterdir():
            if f.is_file() and suffix in f.stem:
                f.unlink()

    @staticmethod
    def _network_dict(network):
        return {'positions': network.positions, 'edges': network.edges,
                'stiffnesses': network.stiffnesses, 'rest_lengths': network.rest_lengths}

    def save_results(self, history, network, task_config, boundary=None):
        """history.pkl, boundary_nodes.json, loss / stiffness trajectories, edges,
        final_network.pkl and task_config.json, from a trainer history and network."""
        self.d.mkdir(parents=True, exist_ok=True)
        stiffness_array = np.array(history.get('stiffnesses', []))
        loss_array = np.array(history.get('loss', []))
        boundary_to_save = None
        if boundary is not None:
            boundary_to_save = {k: np.asarray(v).tolist() for k, v in boundary.items()}
        with open(self._write_path('history', 'pkl'), 'wb') as f:
            pickle.dump({'stiffnesses': stiffness_array, 'loss': loss_array,
                         'positions': history.get('positions', []),
                         'freetraj': history.get('freetraj', []),
                         'boundary': boundary_to_save}, f)
        if boundary_to_save is not None:
            with open(self._write_path('boundary_nodes', 'json'), 'w') as f:
                json.dump(boundary_to_save, f, indent=2)
        np.save(self._write_path('loss_trajectory', 'npy'), loss_array)
        np.save(self._write_path('stiffness_trajectory', 'npy'), stiffness_array)
        np.save(self._write_path('edges', 'npy'), np.array(network.edges))
        with open(self._write_path('final_network', 'pkl'), 'wb') as f:
            pickle.dump(self._network_dict(network), f)
        self._save_task_config(task_config)

    def _save_task_config(self, task_config):
        with open(self._write_path('task_config', 'json'), 'w') as f:
            json.dump(task_config, f, indent=2)

    def save_training_checkpoint(self, step, history, network, task_config, task_id,
                                 realization_id):
        self.d.mkdir(parents=True, exist_ok=True)
        with open(self._write_path('checkpoint', 'pkl'), 'wb') as f:
            pickle.dump({'task_seed': task_id, 'realization_seed': realization_id,
                         'current_step': step, 'history': history,
                         'network': self._network_dict(network), 'task_config': task_config}, f)
        self._save_task_config(task_config)

    def drop_checkpoint(self):
        path = self._write_path('checkpoint', 'pkl')
        if path.exists():
            path.unlink()

    def mark_complete(self):
        self.d.mkdir(parents=True, exist_ok=True)
        with open(self._write_path('training_complete', 'txt'), 'w') as f:
            f.write(f"Completed at {datetime.now().isoformat()}\n")

    def mark_complete_small_loss(self, reduction_ratio=None):
        self.d.mkdir(parents=True, exist_ok=True)
        with open(self._write_path('training_complete_small_loss', 'txt'), 'w') as f:
            f.write(f"Completed (loss reduction criterion) at {datetime.now().isoformat()}\n")
            if reduction_ratio is not None:
                f.write(f"Loss reduction ratio: {reduction_ratio:.2e} (initial/min)\n")
                f.write(f"Orders of magnitude: {np.log10(reduction_ratio):.2f}\n")


class _UniformReader:
    """The uniform layout of ADR 0006, for new realizations of either task kind. Plain file
    names (no network-type suffix); every file is written to a temporary name and renamed, so
    a killed job leaves the previous complete file. Kind-specific behaviour (what K* is,
    what the sweep's after Hessian is taken at, how a run resumes) follows the legacy reader
    of the same kind."""
    MARKER = 'record.json'
    SCHEMA_VERSION = 1

    def __init__(self, run_dir, kind, **_layout):   # layout (network_type) does not apply
        self.d = Path(run_dir)
        self.kind = kind
        self._legacy = _READERS[kind]

    # Where realizations sit under a data root is the same as for the legacy layouts.

    @staticmethod
    def stored_kind(run_dir):
        return json.loads((Path(run_dir) / _UniformReader.MARKER).read_text())['kind']

    def path(self, stem, ext):
        return self.d / f'{stem}.{ext}'

    # ── Reads ──

    def config(self):
        return json.loads(_existing(self.d / 'training_meta.json').read_text())

    def _network_file(self):
        with np.load(_existing(self.d / 'network.npz')) as z:
            return {k: z[k] for k in z.files}

    def network(self):
        z = self._network_file()
        if self.kind == 'allosteric':
            return Network(nodes=z['nodes'], incidence_matrix=z['incidence_matrix'],
                           eq_lengths=z['eq_lengths'])
        return AuxeticNetwork(z['positions'], z['edges'], z['rest_lengths'])

    def boundary(self):
        z = self._network_file()
        boundary = {k[len('boundary_'):]: z[k].astype(int) for k in z if k.startswith('boundary_')}
        if not boundary:
            raise MissingRecordPart(f'{self.d}: no boundary nodes stored')
        return boundary

    def task(self):
        data = json.loads(_existing(self.d / 'task.json').read_text())
        if self.kind == 'allosteric':
            return AllostericTask(data['geometry_seed'], data['strain_output'],
                                  data['strain_output2'])
        if data is None:
            raise MissingRecordPart(f'{self.d}: task is null')
        return AuxeticTask(data)

    def loss(self):
        loss = _load(self.d / 'loss.npy')
        return Loss(loss) if loss.ndim == 2 else Loss(training=loss)

    def stiffness_trajectory(self):
        with np.load(_existing(self.d / 'stiffness.npz')) as z:
            return StiffnessTrajectory(steps=z['steps'], stiffnesses=z['stiffnesses'])

    def positions_trajectory(self):
        return list(_load(self.d / 'positions.npy'))

    def current_stiffnesses(self):
        return _load(self.d / 'current.npy')

    def best_so_far(self):
        with np.load(_existing(self.d / 'best.npz')) as z:
            return BestSoFar(z['stiffnesses'], float(z['training_loss']))

    def sweep(self):
        with np.load(_existing(self.d / 'sweep.npz'), allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    @staticmethod
    def sweep_file_names():
        return 'sweep.npz', 'sweep_meta.json'

    def sweep_after_hessian_at_k_star(self, record, sweep):
        return self._legacy.sweep_after_hessian_at_k_star(record, sweep)

    def stiffnesses_at_best_loss(self, record, step, undetermined):
        if self.kind == 'auxetic':
            return _AuxeticReader.stiffnesses_at_best_loss(self, record, step, undetermined)
        # Allosteric: the trainer's stored best state, accepted only if the stored data pin it
        # to `step`. New runs always store it pre-update (ADR 0004), so no start-date check.
        loss = record.loss.training
        best = record.best_so_far
        if best.training_loss != loss[step]:
            raise undetermined(f'stored best loss {best.training_loss!r} is not the lowest '
                               f'training loss {loss[step]!r} (step {step})')
        traj = record.stiffness_trajectory
        if not _fits(traj, len(loss)):
            raise undetermined('stiffness trajectory does not fit the loss trajectory')
        snapshot = np.nonzero(traj.steps == step)[0]
        if len(snapshot) and not np.array_equal(traj.stiffnesses[snapshot[0]], best.stiffnesses):
            raise undetermined(f'stored best stiffnesses differ from the snapshot at step {step}')
        return best.stiffnesses

    def resume_state(self):
        if self.kind == 'allosteric':
            return self._resume_allosteric()
        return self._resume_auxetic()

    def _resume_allosteric(self):
        if not (self.d / 'current.npy').exists():
            return None
        # A run killed before its first loss save resumes from step 0 with no history.
        loss = np.load(self.d / 'loss.npy') if (self.d / 'loss.npy').exists() else np.empty((0, 2))
        current = np.load(self.d / 'current.npy')
        if not np.any(np.isnan(current)):
            return ResumeState(len(loss), current, Loss(loss), 'current')
        if (self.d / 'resume.npz').exists():
            with np.load(self.d / 'resume.npz') as z:
                step, stiffnesses = int(z['step']), z['stiffnesses']
            return ResumeState(step, stiffnesses, Loss(loss[:step]), 'checkpoint')
        return None

    def _resume_auxetic(self):
        """The checkpoint, or None when there is none or it cannot be read."""
        resume, blob = self.d / 'resume.npz', self.d / 'trainer_state.pkl'
        if not resume.exists() or not blob.exists():
            return None
        try:
            with np.load(resume) as z:
                step, stiffnesses, loss = int(z['step']), z['stiffnesses'], z['loss']
            with open(blob, 'rb') as f:
                trainer_state = pickle.load(f)
        except Exception as e:  # noqa: BLE001 - a truncated checkpoint means a fresh start
            print(f'Warning: Failed to load checkpoint {resume}: {e}')
            return None
        return ResumeState(step, stiffnesses, Loss(training=loss), 'checkpoint',
                           trainer_state=trainer_state)

    # ── Writes ──

    def start(self):
        self.d.mkdir(parents=True, exist_ok=True)
        marker = self.d / self.MARKER
        if not marker.exists():
            self._write_json(self.MARKER, {'schema_version': self.SCHEMA_VERSION,
                                           'kind': self.kind})

    def _atomic(self, name, write):
        self.start()
        tmp = self.d / f'.{name}.tmp'
        with open(tmp, 'wb') as f:
            write(f)
        os.replace(tmp, self.d / name)

    def _write_npy(self, name, array):
        self._atomic(name, lambda f: np.save(f, array))

    def _write_npz(self, name, **arrays):
        self._atomic(name, lambda f: np.savez(f, **arrays))

    def _write_json(self, name, data):
        tmp = self.d / f'.{name}.tmp'
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.d / name)

    def _write_text(self, name, text):
        self._atomic(name, lambda f: f.write(text.encode()))

    def meta_file_name(self, stem):
        return f'{stem}.json'

    def wipe(self):
        if self.d.exists():
            shutil.rmtree(self.d)
        self.start()

    def write_network(self, network):
        self._write_npz('network.npz', nodes=network.nodes,
                        incidence_matrix=network.incidence_matrix, eq_lengths=network.eq_lengths)

    def write_task(self, task):
        self._write_json('task.json', {'geometry_seed': task.geometry_seed,
                                       'strain_output': task.strain_output,
                                       'strain_output2': task.strain_output2})

    def save_checkpoint(self, step, current_stiffnesses, loss, measured_stiffnesses, best):
        self._write_npy('current.npy', current_stiffnesses)
        self._write_npy('loss.npy', np.ascontiguousarray(loss.per_subtask))
        if not np.any(np.isnan(current_stiffnesses)):
            self._write_npz('resume.npz', step=step, stiffnesses=current_stiffnesses)
        if best is not None:
            self._write_npz('best.npz', stiffnesses=best.stiffnesses,
                            training_loss=best.training_loss)
        # The snapshot pairs with its own loss index (the entry just measured), not `step`.
        if (self.d / 'stiffness.npz').exists():
            old = self.stiffness_trajectory()
            traj = np.vstack([old.stiffnesses, measured_stiffnesses])
            steps = np.append(old.steps, loss.n_steps - 1)
        else:
            traj = measured_stiffnesses[np.newaxis, :]
            steps = np.array([loss.n_steps - 1])
        self._write_npz('stiffness.npz', steps=steps, stiffnesses=traj)

    def save_final(self, current_stiffnesses, loss, best):
        self._write_npy('current.npy', current_stiffnesses)
        if best is not None:
            self._write_npz('best.npz', stiffnesses=best.stiffnesses,
                            training_loss=best.training_loss)
        self._write_npy('loss.npy', np.ascontiguousarray(loss.per_subtask))

    # Auxetic writes (the ensemble runners and their trainers).

    def save_results(self, history, network, task_config, boundary=None):
        stiffness_array = np.array(history.get('stiffnesses', []))
        loss_array = np.array(history.get('loss', []))
        arrays = {'positions': np.asarray(network.positions), 'edges': np.array(network.edges),
                  'rest_lengths': np.asarray(network.rest_lengths)}
        for k, v in (boundary or {}).items():
            arrays[f'boundary_{k}'] = np.asarray(v, dtype=int)
        self._write_npz('network.npz', **arrays)
        self._write_npy('current.npy', np.asarray(network.stiffnesses))
        self._write_npy('loss.npy', loss_array)
        self._write_npz('stiffness.npz', steps=np.arange(len(stiffness_array)),
                        stiffnesses=stiffness_array)
        self._write_npy('positions.npy', np.array(history.get('positions', [])))
        self._write_json('task.json', task_config)

    def save_training_checkpoint(self, step, history, network, task_config, task_id,
                                 realization_id):
        self._write_npz('resume.npz', step=step, stiffnesses=np.asarray(network.stiffnesses),
                        loss=np.asarray(history.get('loss', []), float))
        ckpt = {'task_seed': task_id, 'realization_seed': realization_id, 'current_step': step,
                'history': history, 'network': _AuxeticReader._network_dict(network),
                'task_config': task_config}
        self._atomic('trainer_state.pkl', lambda f: pickle.dump(ckpt, f))
        self._write_json('task.json', task_config)

    def drop_checkpoint(self):
        for name in ('resume.npz', 'trainer_state.pkl'):
            if (self.d / name).exists():
                (self.d / name).unlink()

    def mark_complete(self):
        self._write_text('training_complete.txt', f"Completed at {datetime.now().isoformat()}\n")

    def mark_complete_small_loss(self, reduction_ratio=None):
        text = f"Completed (loss reduction criterion) at {datetime.now().isoformat()}\n"
        if reduction_ratio is not None:
            text += (f"Loss reduction ratio: {reduction_ratio:.2e} (initial/min)\n"
                     f"Orders of magnitude: {np.log10(reduction_ratio):.2f}\n")
        self._write_text('training_complete_small_loss.txt', text)


_READERS = {'allosteric': _AllostericReader, 'auxetic': _AuxeticReader}

LAYOUT_ENV = 'NMN_RECORD_LAYOUT'   # 'uniform' | 'legacy' (default): layout of fresh run directories


def _reader_for(kind, run_dir, layout):
    """The reader of the layout `run_dir` is in; a directory holding no run yet gets the
    layout LAYOUT_ENV names (legacy unless 'uniform'). A run keeps the layout it started in."""
    run_dir = Path(run_dir)
    if (run_dir / _UniformReader.MARKER).exists():
        return _UniformReader(run_dir, kind, **layout)
    legacy = _READERS[kind]
    if run_dir.is_dir() and legacy.recognises(run_dir):
        return legacy(run_dir, **layout)
    if os.environ.get(LAYOUT_ENV, 'legacy') == 'uniform':
        return _UniformReader(run_dir, kind, **layout)
    return legacy(run_dir, **layout)
