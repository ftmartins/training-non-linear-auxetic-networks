"""
Fixed sample of real run directories the golden tests read, one entry per layout or
content variant they must keep reading the same way. Paths are relative to data/.
"""

def allosteric_name(geometry, task, real):
    return f'allo_g{geometry}_t{task}_r{real}'


def auxetic_name(family, task, real):
    return f'aux_{family}_t{task:02d}_r{real:02d}'


def legacy_name(data_dir, task, real, network_type):
    return f"legacy_{data_dir.replace('/', '_')}_t{task:02d}_r{real:02d}_{network_type or 'plain'}"


ALLOSTERIC = [
    # (geometry, task, realization, why)
    # "new sweep" = 2026-09-10+ sweep meta (cost_hessian_solver key: after Hessian at K*);
    # "old sweep" = earlier meta (after Hessian at the checkpoint grid).
    ('targeted', 0, 0, 'targeted, new sweep, converged, several invocations'),
    ('targeted', 0, 12, 'targeted, new sweep, converged, one invocation'),
    ('targeted', 0, 11, 'targeted, no sweep, not converged'),
    ('targeted', 0, 10, 'targeted, no sweep, several invocations'),
    ('targeted', 2, 8, 'targeted, old sweep, several invocations'),
    (0, 0, 2, 'general, old sweep, not converged'),
    (0, 1, 3, 'general, new sweep, one invocation'),
    (1, 0, 1, 'general, old sweep'),
    (1, 2, 7, 'general, no sweep, several invocations'),
    (2, 0, 5, 'sweep meta present but no timestep_sweep.npz'),
    (2, 0, 1, 'general, new sweep, several invocations'),
    (3, 0, 0, 'general, no sweep, not converged'),
    (4, 0, 3, 'general, new sweep, several invocations'),
    (4, 2, 1, 'stiffness trajectory starts at step 4349, not 49'),
    (4, 4, 2, 'stiffness trajectory starts at step 4449, not 49'),
]

AUXETIC = [
    # (family, task, realization, why); data/auxetic_nets_aug/<family>
    ('targeted', 0, 2, 'targeted Network A, sweep, boundary nodes, no checkpoint'),
    ('targeted', 0, 0, 'targeted Network A, sweep, other file set'),
    ('targeted', 1, 16, 'targeted, checkpoint present with sweep'),
    ('targeted', 2, 9, 'targeted, checkpoint present, no sweep'),
    ('targeted', 19, 2, 'targeted, no boundary_nodes'),
    ('targeted', 20, 1, 'targeted, sweep meta without npz'),
    ('targeted', 8, 0, 'targeted Network B'),
    ('targeted', 12, 5, 'targeted Network C'),
    ('general', 2, 4, 'general, no final_network (loaders return None)'),
    ('general', 2, 5, 'general, checkpoint only'),
    ('general', 26, 18, 'general, bare (marker only)'),
    ('general', 0, 3, 'general, checkpoint and final network'),
    ('general', 4, 5, 'general, checkpoint, final network and sweep'),
    ('general', 28, 8, 'general, no boundary nodes'),
    ('general', 29, 5, 'general, boundary nodes, final network, no sweep'),
]
LEGACY = [
    # (data_dir under data/, task, realization, network_type, why)
    ('auxetic_nets/targeted_results_sqr/jax', 0, 0, None, 'plain names, checkpoint'),
    ('auxetic_nets/targeted_results_sqr/newton', 0, 0, 'jammed', 'newton, jammed'),
    ('auxetic_nets/targeted_results_sqr/newton', 0, 0, 'lattice', 'newton, lattice'),
    ('auxetic_nets/targeted_results_sqr/newton', 10, 0, 'lattice', 'newton, lattice checkpoint'),
    ('auxetic_nets/targeted_results_sqr/old', 0, 0, None, 'small-loss marker only'),
    ('auxetic_nets/targeted_results/old_targeted', 0, 0, None, 'small-loss marker'),
]
