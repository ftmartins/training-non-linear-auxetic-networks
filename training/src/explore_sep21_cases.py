"""
Case tables for the 2026-09-21 explorations (docs/newExplorationsSeptember21st.md).

Everything here writes under EXPLORE_ROOT, a tree separate from allosteric_nets_aug /
auxetic_networks so the existing ensemble is never touched.

Families
--------
aux_z        auxetic, coordination-number sweep.  Jammed packings generated with a
             varying `central` packing force (calibrated: central -> z=2E/N of the
             cut network, see Z_CENTRALS).  Targeted auxetic tasks AUX_Z_TASKS.
allo_z       allosteric, coordination-number sweep.  Same triangular-lattice geometry
             family as allosteric_trainer.create_network, randomly bond-diluted to a
             target z (degree >= 3, floppy-mode check).  Targeted allosteric tasks
             ALLO_Z_TASKS.
aux_stretch  auxetic tasks retrained with the strain SIGN flipped (stretch instead of
             compression) AND the Poisson-ratio sign flipped, on the same network and
             the same IC seeds as the original compression run.

Indexing (SLURM array index -> case) is done by `aux_z_cases()` etc. below.
"""
from pathlib import Path

EXPLORE_ROOT = Path('/data2/shared/felipetm/explore_sep21')

# ── coordination-number sweep ────────────────────────────────────────────────
N_TOPO = 5           # network topology realizations per z
N_IC = 5             # initial-stiffness seeds per (task, z, topology)

# Auxetic: packing `central` force -> measured z (2E/N, N=100 cut, 4 seeds, 2026-09-21):
#   5e-5:3.9  1e-4:4.25  2e-4:4.45  5e-4:4.75  1e-3:5.03  1.6e-3:5.2  2.2e-3:5.5  3e-3:6.25  5e-3:8.3
# 2026-09-26 revision: max z capped at 5.5 (was 6.3). z0/z1/z2 unchanged; z3/z4 retuned down
# from 2.5e-3/3.5e-3.
# 2026-09-29: the max-degree-6 pruning (cap_max_degree) is REMOVED -- it collapsed z3/z4 back
# onto z2 (all realized z~5.06). z3/z4 retrained uncapped. Also added z_idx 5 at the canonical
# packing force 5e-5 (z~3.9, the Fig 1-5 auxetic ensemble's own z), appended AFTER the original
# 250 cases (AUX_Z_EXTRA_CENTRALS) so existing case_ids / on-disk task_<case_id> dirs don't move.
Z_CENTRALS = [8e-5, 3e-4, 1e-3, 1.6e-3, 2.2e-3]     # ~4.1, ~4.6, ~5.0, ~5.2, ~5.5
AUX_Z_EXTRA_CENTRALS = [5e-5]                       # z_idx 5.. -> ~3.9 (canonical ensemble)
AUX_Z_PACKING_SEED_BASE = 100                       # topology t -> packing seed 100+t (shared across z)
AUX_Z_TASKS = [1, 2]                                # targeted auxetic tasks (see targeted_task_generator)

# Allosteric: target z = 2E/N for the diluted lattice
ALLO_Z_TARGETS = [4.1, 4.6, 5.1, 5.6, None]         # None -> native (undiluted) lattice
ALLO_Z_GEOM_SEED_BASE = 1_000_200                   # topology t -> geometry seed base+t
ALLO_Z_TASKS = [0, 2]                               # TARGETED_ENSEMBLE indices
ALLO_IC_SEED_BASE = 9_000                           # IC index i -> realization_rng(9000+i)
ALLO_TRAIN_STEPS = 5000

# ── stretch vs compress ──────────────────────────────────────────────────────
# (original targeted auxetic task id, flipped compression list, flipped nu list)
AUX_STRETCH_TASKS = {
    1: dict(strains=[0.2, 0.1], poissons=[0.8, 0.8]),     # orig -0.2,-0.1 / -0.8,-0.8
    2: dict(strains=[0.2, 0.1], poissons=[0.8, 1.0]),     # orig -0.2,-0.1 / -0.8,-1.0
    3: dict(strains=[0.2, 0.1], poissons=[0.8, 0.4]),     # orig -0.2,-0.1 / -0.8,-0.4
}
AUX_STRETCH_N_IC = 5      # screened-seed indices 0..4 of the original task (same as compression run)


def aux_z_cases():
    """List of dicts, one per SLURM array index (task-major, then z, topology, IC) for
    Z_CENTRALS (case_ids 0-249), followed by the same block for AUX_Z_EXTRA_CENTRALS
    (z_idx continuing from len(Z_CENTRALS); case_ids 250+)."""
    out = []
    for zlist, z0 in ((Z_CENTRALS, 0), (AUX_Z_EXTRA_CENTRALS, len(Z_CENTRALS))):
        for task in AUX_Z_TASKS:
            for zi, central in enumerate(zlist, start=z0):
                for topo in range(N_TOPO):
                    for ic in range(N_IC):
                        out.append(dict(family='aux_z', orig_task=task, z_idx=zi, central=central,
                                        topo=topo, ic=ic, packing_seed=AUX_Z_PACKING_SEED_BASE + topo))
    for cid, c in enumerate(out):
        c['case_id'] = cid
    return out


def allo_z_cases():
    out = []
    for task in ALLO_Z_TASKS:
        for zi, zt in enumerate(ALLO_Z_TARGETS):
            for topo in range(N_TOPO):
                for ic in range(N_IC):
                    out.append(dict(family='allo_z', task=task, z_idx=zi, z_target=zt, topo=topo,
                                    ic=ic, geom_seed=ALLO_Z_GEOM_SEED_BASE + topo,
                                    ic_seed=ALLO_IC_SEED_BASE + ic))
    for cid, c in enumerate(out):
        c['case_id'] = cid
    return out


def aux_stretch_cases():
    out = []
    for task, spec in AUX_STRETCH_TASKS.items():
        for ic in range(AUX_STRETCH_N_IC):
            out.append(dict(family='aux_stretch', orig_task=task, ic=ic,
                            strains=spec['strains'], poissons=spec['poissons']))
    for cid, c in enumerate(out):
        c['case_id'] = cid
    return out


ALLO_Z_EXTRA_BASE = 10_000    # case_id offset, disjoint from allo_z_cases()'s 0-249 -- on-disk paths
                               # are keyed by z_idx/topo/task/ic fields (allo_z_dir), never by case_id,
                               # so this offset is only to keep SLURM array indices from colliding.


def allo_z_extra_cases(cells, ic_start, ic_stop):
    """More IC seeds for specific (task, z_idx) cells that came up short of enough converged
    realizations in the first pass (2026-09-26 explorations follow-up). `cells` is an iterable
    of (task, z_idx) pairs; `ic` runs `range(ic_start, ic_stop)` across all N_TOPO topologies,
    same geometry/seed conventions as `allo_z_cases()`."""
    out = []
    for task, zi in cells:
        for topo in range(N_TOPO):
            for ic in range(ic_start, ic_stop):
                out.append(dict(family='allo_z', task=task, z_idx=zi, z_target=ALLO_Z_TARGETS[zi],
                                topo=topo, ic=ic, geom_seed=ALLO_Z_GEOM_SEED_BASE + topo,
                                ic_seed=ALLO_IC_SEED_BASE + ic))
    for i, c in enumerate(out):
        c['case_id'] = ALLO_Z_EXTRA_BASE + i
    return out


def aux_z_dir(root=EXPLORE_ROOT):
    return Path(root) / 'aux_z'


def aux_stretch_dir(root=EXPLORE_ROOT):
    return Path(root) / 'aux_stretch'


def allo_z_dir(case, root=EXPLORE_ROOT):
    """Trainer --output-dir for one allosteric case; the trainer appends
    geometry_targeted/task_<t>/realization_<r>. One dir per (z, topology) so the
    pre-placed geometry files are shared by that group's tasks/ICs."""
    return Path(root) / 'allo_z' / f"z{case['z_idx']}_topo{case['topo']}"
