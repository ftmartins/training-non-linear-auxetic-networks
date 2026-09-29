#!/usr/bin/env python3
"""
Auxetic explorations (2026-09-21): coordination-number sweep and stretch-vs-compress.

Thin wrapper around targeted_ensemble_runner.run_single_training — no existing file is
modified. For one case (training/src/explore_sep21_cases) it monkeypatches, in-process:
  * get_targeted_task_config  -> the case's task (packing seed, strains, Poisson ratios)
  * get_realization_seed      -> the case's IC seed
  * generate_realization_stiffnesses -> always the log-uniform (task_seed<20) branch
  * PACKING_PARAMS['central'] -> the case's packing force (sets z); restored never (one case per process)
  * targeted_task_generator.TARGETED_RESULTS_DIR -> EXPLORE_ROOT/<family>
so results land in <EXPLORE_ROOT>/<family>/jax/task_<case_id>/realization_00/ and the
existing targeted_results_sqr_aug tree is untouched.

Families:
  aux_z        jammed packing with central force EC.Z_CENTRALS[z_idx] (or AUX_Z_EXTRA_CENTRALS),
               packing seed 100+topo, NO degree cap (cap_max_degree removed 2026-09-29),
               IC seed = `ic` (0..4), tasks = targeted auxetic tasks EC.AUX_Z_TASKS.
  aux_stretch  same network as the original targeted task (packing seed 42, central 5e-5),
               same screened IC seeds, strains/Poisson ratios sign-flipped.

Usage: python explore_sep21_aux.py --family aux_z|aux_stretch --case-index N
                                   [--steps S] [--learning-rate LR] [--overwrite]
       python explore_sep21_aux.py --family aux_z --list
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / 'training' / 'src'), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.src import explore_sep21_cases as EC


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=['aux_z', 'aux_stretch'], required=True)
    p.add_argument('--case-index', type=int)
    p.add_argument('--list', action='store_true')
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--root', type=str, default=str(EC.EXPLORE_ROOT))
    a = p.parse_args()

    cases = EC.aux_z_cases() if a.family == 'aux_z' else EC.aux_stretch_cases()
    if a.list:
        print(len(cases))
        return
    c = cases[a.case_index]
    cid = c['case_id']
    results_root = Path(a.root) / a.family

    import training.src.targeted_task_generator as ttg
    import targeted_ensemble_runner as R
    from training.src.good_realizations import get_realization_seed as _orig_seed
    from training.src.task_generator import generate_realization_stiffnesses as _orig_gen

    base_cfg = copy.deepcopy(ttg.get_targeted_task_config(c['orig_task']))
    if a.family == 'aux_z':
        cfg = dict(base_cfg, packing_seed=c['packing_seed'])
        central = c['central']
        ic_seed = c['ic']
    else:
        cfg = dict(base_cfg, compression_strains=list(c['strains']),
                   target_poisson_ratios=list(c['poissons']))
        central = R.PACKING_PARAMS['central']          # unchanged: same network as the original task
        ic_seed = _orig_seed('auxetic_targeted', c['orig_task'], c['ic'])
    cfg['task_seed'] = cid

    ttg.TARGETED_RESULTS_DIR = results_root
    R.get_targeted_task_config = lambda tid: copy.deepcopy(cfg)
    R.get_realization_seed = lambda kind, tid, rid: ic_seed
    R.generate_realization_stiffnesses = lambda tid, seed, n: _orig_gen(1, seed, n)
    R.PACKING_PARAMS['central'] = central

    print(f"[explore_sep21] {a.family} case {cid}: {c}\n  config: {cfg}\n  central={central} "
          f"ic_seed={ic_seed} -> {results_root}")
    ok = R.run_single_training(cid, 0, verbose=False, use_checkpoint=True, gradient_method='jax',
                               overwrite=a.overwrite, lr_override=a.learning_rate,
                               steps_override=a.steps)

    # Provenance for the notebook: case params + realized coordination number.
    try:
        from base.network_utils import create_auxetic_network
        net, _ = create_auxetic_network(n_nodes=cfg['n_nodes'], packing_seed=cfg['packing_seed'],
                                        central_force=central)
        z = 2 * len(net.edges) / len(net.positions)
        d = results_root / 'jax' / f'task_{cid:02d}' / 'realization_00'
        d.mkdir(parents=True, exist_ok=True)
        with open(d / 'explore_case.json', 'w') as f:
            json.dump({**c, 'central': central, 'ic_seed': ic_seed, 'z': z,
                       'n_nodes': len(net.positions), 'n_edges': len(net.edges),
                       'task_config': cfg}, f, indent=1)
    except Exception as e:                                # provenance is best-effort
        print(f"[explore_sep21] provenance write failed: {e}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
