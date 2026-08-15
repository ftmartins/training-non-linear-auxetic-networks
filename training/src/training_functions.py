"""
Training functions for spring networks.

Physics/simulation primitives (elastic_energy, FIRE minimizer, quasistatic
trajectories, JAX differentiable solver) live in base.simulate and are
re-exported here for convenience. Only training-specific logic (loss, gradient,
GD loops, checkpointing) is defined in this module.
"""

import sys
import copy
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from base.config import FORCE_TOL, NETWORK_TYPE
from base.simulate import (
    elastic_energy,
    fire_minimize_network,
    compute_quasistatic_trajectory_auxetic,
    compute_quasistatic_trajectory_full_cycle,
    compute_ift_gradient,
    make_compute_response_fire,
    crf,
    compute_quasistatic_trajectory_auxetic_jax,
    compute_poisson_ratio_single_jax,
)
from .checkpoint_manager import (
    save_training_results,
    save_checkpoint,
)
from . import lr_schedule



# ============================================================================
def poisson_loss_batch_jax(crf, stiffnesses, edges, rest_lengths, positions_flat,
                           top_nodes, bottom_nodes, left_nodes, right_nodes,
                           compression_strain_list, target_poisson_list, n_steps, d=2):
    """
    JAX-differentiable MSE loss across multiple compression-Poisson pairs.

    Returns:
        mse_loss: scalar = mean((computed_nu - target_nu)^2)
    """
    total_loss = 0.0
    n_pairs = len(compression_strain_list)
    for cs, target_nu in zip(compression_strain_list, target_poisson_list):
        computed_nu = compute_poisson_ratio_single_jax(
            crf, stiffnesses, edges, rest_lengths, positions_flat,
            top_nodes, bottom_nodes, left_nodes, right_nodes,
            cs, n_steps, d=d
        )
        total_loss = total_loss + (computed_nu - target_nu) ** 2
    return total_loss / n_pairs


# ============================================================================
# TRAINING HELPER FUNCTIONS (Cython-based, original)
# ============================================================================

def compute_poisson_ratio_single(network, top_nodes, bottom_nodes, left_nodes, right_nodes,
                                 compression_strain, n_strain_steps=100, tol=FORCE_TOL, force_type='quadratic'):
    """
    Compute Poisson ratio for a single compression strain.
    """
    traj = compute_quasistatic_trajectory_auxetic(
        network,
        compression_strain,
        top_nodes,
        bottom_nodes,
        n_steps=n_strain_steps,
        verbose=False,
        force_type=force_type,
        tol=tol,
        method='fire',
    )

    positions_free_final = traj[-1]
    positions_free_initial = traj[0]

    left_x1 = positions_free_final[left_nodes, 0].mean()
    right_x1 = positions_free_final[right_nodes, 0].mean()
    left_x2 = positions_free_initial[left_nodes, 0].mean()
    right_x2 = positions_free_initial[right_nodes, 0].mean()

    width_free_final = right_x1 - left_x1
    width_free_initial = right_x2 - left_x2
    lateral_strain = (width_free_final - width_free_initial) / width_free_initial
    poisson_ratio = -(lateral_strain / compression_strain)

    return poisson_ratio


def poisson_loss_batch_parallel(network, target_poisson_list, top_nodes, bottom_nodes,
                                left_nodes, right_nodes, compression_strain_list,
                                n_strain_steps=100, n_jobs_inner=4, force_type='quadratic', tol=FORCE_TOL):
    """
    Compute MSE loss across multiple compression-Poisson pairs in parallel.
    """
    computed_poisson_ratios = Parallel(n_jobs=n_jobs_inner)(
        delayed(compute_poisson_ratio_single)(
            network, top_nodes, bottom_nodes, left_nodes, right_nodes,
            cs, n_strain_steps, force_type=force_type, tol=tol
        )
        for cs in compression_strain_list
    )

    computed_poisson_ratios = np.array(computed_poisson_ratios)
    mse_loss = np.mean((computed_poisson_ratios - np.array(target_poisson_list))**2)

    return mse_loss, computed_poisson_ratios


def compute_gradient_entry_batch(i, network, target_poisson_list, top_nodes, bottom_nodes,
                                 left_nodes, right_nodes, compression_strain_list,
                                 epsilon, n_strain_steps, n_jobs_inner=4, force_type='quadratic', tol=FORCE_TOL):
    """
    Compute gradient for a single stiffness using finite differences.
    """
    orig = network.stiffnesses[i]

    # Perturb up
    network.stiffnesses[i] = orig + epsilon
    loss_plus, _ = poisson_loss_batch_parallel(
        network, target_poisson_list, top_nodes, bottom_nodes,
        left_nodes, right_nodes, compression_strain_list,
        n_strain_steps, n_jobs_inner, force_type=force_type, tol=tol
    )

    # Perturb down
    network.stiffnesses[i] = orig - epsilon
    loss_minus, _ = poisson_loss_batch_parallel(
        network, target_poisson_list, top_nodes, bottom_nodes,
        left_nodes, right_nodes, compression_strain_list,
        n_strain_steps, n_jobs_inner, force_type=force_type, tol=tol
    )

    # Restore
    network.stiffnesses[i] = orig

    # Gradient
    value = (loss_plus - loss_minus) / (2 * epsilon)

    return (i, value)


def finite_difference_gradient_parallel_batch(network, target_poisson_list, top_nodes, bottom_nodes,
                                             left_nodes, right_nodes, compression_strain_list,
                                             epsilon=1e-8, n_jobs_outer=4, n_jobs_inner=2,
                                             n_strain_steps=100, force_type='quadratic', tol=FORCE_TOL):
    """
    Compute gradient across all stiffnesses using parallel finite differences.

    - Outer parallelization: across stiffness indices
    - Inner parallelization: across compression strains
    """
    n_edges = len(network.stiffnesses)

    results = Parallel(n_jobs=n_jobs_outer)(
        delayed(compute_gradient_entry_batch)(
            i, network, target_poisson_list,
            top_nodes, bottom_nodes, left_nodes, right_nodes,
            compression_strain_list, epsilon, n_strain_steps,
            n_jobs_inner, force_type=force_type, tol=tol
        )
        for i in range(n_edges)
    )

    grad = np.zeros(n_edges)
    for i, value in results:
        grad[i] = value

    return grad


# ============================================================================
# TRAINING HELPER FUNCTIONS (Newton-loss + finite-difference gradient)
# ============================================================================

def _newton_fd_gradient_entry(i, network, target_poisson_list, top_nodes, bottom_nodes,
                              left_nodes, right_nodes, compression_strain_list,
                              epsilon, n_strain_steps, tol):
    """
    Central-difference gradient for a single stiffness, using the Newton-solved
    loss (compute_ift_gradient's forward pass) instead of its analytic gradient.
    """
    orig = network.stiffnesses[i]

    network.stiffnesses[i] = orig + epsilon
    loss_plus, _ = compute_ift_gradient(
        network, compression_strains=compression_strain_list, target_poissons=target_poisson_list,
        top_nodes=top_nodes, bottom_nodes=bottom_nodes, left_nodes=left_nodes, right_nodes=right_nodes,
        tol=tol, n_strain_steps=n_strain_steps,
    )

    network.stiffnesses[i] = orig - epsilon
    loss_minus, _ = compute_ift_gradient(
        network, compression_strains=compression_strain_list, target_poissons=target_poisson_list,
        top_nodes=top_nodes, bottom_nodes=bottom_nodes, left_nodes=left_nodes, right_nodes=right_nodes,
        tol=tol, n_strain_steps=n_strain_steps,
    )

    network.stiffnesses[i] = orig
    return (i, (loss_plus - loss_minus) / (2 * epsilon))


def finite_difference_gradient_newton_batch(network, target_poisson_list, top_nodes, bottom_nodes,
                                            left_nodes, right_nodes, compression_strain_list,
                                            epsilon=1e-8, n_jobs=4,
                                            n_strain_steps=100, tol=FORCE_TOL):
    """
    Gradient across all stiffnesses via central finite differences of the Newton-solved
    loss (base.simulate.compute_ift_gradient's Newton-Raphson forward pass), bypassing
    its analytic implicit-function-theorem gradient entirely.

    Parallelized across stiffness indices only (each Newton solve already accounts for
    all compression strains internally, unlike the Cython-FIRE finite-difference path
    above which parallelizes strains too).
    """
    n_edges = len(network.stiffnesses)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_newton_fd_gradient_entry)(
            i, network, target_poisson_list,
            top_nodes, bottom_nodes, left_nodes, right_nodes,
            compression_strain_list, epsilon, n_strain_steps, tol,
        )
        for i in range(n_edges)
    )

    grad = np.zeros(n_edges)
    for i, value in results:
        grad[i] = value

    return grad


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def finish_training_GD_auxetic_batch(
    network, history, learning_rate, n_steps,
    top_nodes, bottom_nodes, left_nodes, right_nodes,
    force_type='quadratic', n_strain_steps=100,
    source_compression_strain_list=[0.2], desired_target_extension_list=[0.2],
    verbose=False, stiffnesses_filename=None, force_tol=1e-8,
    vmin=1e-3, vmax=1e3,
    task_seed=None, realization_seed=None, save_interval=500, task_config=None, TARGETED_RESULTS_DIR=None, loss_tol=1e-6,
    method='newton', network_type=NETWORK_TYPE,
):
    """
    Train the network for auxetic response using gradient descent.

    Trains the network to achieve specific Poisson ratios at given compression strains
    by adjusting spring stiffnesses.

    Args:
        network: ElasticNetwork object to train
        history: Dictionary to store training history
        learning_rate: Gradient descent learning rate
        n_steps: Number of training steps
        top_nodes, bottom_nodes, left_nodes, right_nodes: Boundary node indices
        force_type: 'quartic' or 'quadratic'
        n_strain_steps: Steps per quasistatic trajectory
        source_compression_strain_list: List of compression strains (e.g., [0.04, 0.08])
        desired_target_extension_list: List of target lateral extensions (e.g., [-0.02, -0.02])
        verbose: Print detailed progress
        stiffnesses_filename: Optional file to save stiffnesses
        force_tol: Convergence tolerance for minimization
        vmin, vmax: Bounds for stiffness values
        task_seed: Task index (for saving intermediate results)
        realization_seed: Realization index (for saving intermediate results)
        save_interval: Save intermediate trajectories every N steps (default: 500)
        method: 'newton'    (default) — Newton quasistatic + analytic IFT gradient (fast);
                'newton_fd' — Newton quasistatic loss, but the gradient is a plain
                              central finite difference over stiffnesses of that loss
                              (no IFT/adjoint formula at all — much slower than 'newton');
                'fire'      — Cython FIRE quasistatic + finite-difference gradient (original).

    Returns:
        history: Updated history dictionary with training results
    """
    import copy

    # The quasistatic trajectory itself (used for best-step reconstruction below) is
    # always solved via Newton for both 'newton' and 'newton_fd' — they only differ in
    # how the training-time gradient is obtained, not in the loss/position solver.
    traj_method = 'newton' if method in ('newton', 'newton_fd') else 'fire'

    network = copy.copy(network)
    last_relaxed_positions = np.copy(network.positions)
    loss = np.inf
    min_loss = np.inf

    # Initialize history if needed
    if 'stiffnesses' not in history:
        history['stiffnesses'] = []
    if 'loss' not in history:
        history['loss'] = []
    if 'positions' not in history:
        history['positions'] = []
    if 'freetraj' not in history:
        history['freetraj'] = []

    # Import checkpoint saving functions if task/realization provided
    save_intermediate = (task_seed is not None and realization_seed is not None)
    if save_intermediate:
        from pathlib import Path
        # Import save functions (avoid circular import by importing here)
        import sys
        from pathlib import Path
        ensemble_dir = Path(__file__).parent
        if str(ensemble_dir) not in sys.path:
            sys.path.insert(0, str(ensemble_dir))
        from checkpoint_manager import get_training_result_path

    # Convert extensions to target Poisson ratios
    desired_poisson_list = [
        -(desired_target_extension / source_compression_strain)
        for source_compression_strain, desired_target_extension
        in zip(source_compression_strain_list, desired_target_extension_list)
    ]

    pbar = tqdm(range(n_steps), desc=f'(loss = {loss:.4e}, min loss={min_loss:.4e})')
    _last_traj_best_step = -1  # sentinel: best trajectory not yet computed

    for step in pbar:
        # --- Minimize positions ---
        network.update_positions(last_relaxed_positions)
        min_pos, force_norm = fire_minimize_network(
            network,
            constrained_dof_idx=None,  # Free minimization
            force_type=force_type,
            tol=force_tol
        )

        # Check convergence (Cython only)
        if force_norm is not None:
            assert force_norm < force_tol, f"FIRE did not converge: {force_norm:.3e} > {force_tol:.3e}"

        last_relaxed_positions = min_pos
        network.update_positions(min_pos)

        # --- Gradient update ---
        if method == 'newton':
            _, grad = compute_ift_gradient(
                network,
                compression_strains=source_compression_strain_list,
                target_poissons=desired_poisson_list,
                top_nodes=top_nodes,
                bottom_nodes=bottom_nodes,
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                tol=force_tol,
                n_strain_steps=n_strain_steps,
            )
            update = -grad
        elif method == 'newton_fd':
            update = -finite_difference_gradient_newton_batch(
                copy.deepcopy(network),
                target_poisson_list=desired_poisson_list,
                top_nodes=top_nodes,
                bottom_nodes=bottom_nodes,
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                compression_strain_list=source_compression_strain_list,
                epsilon=1e-8,
                n_strain_steps=n_strain_steps,
                tol=force_tol,
                n_jobs=4,
            )
        else:
            update = -finite_difference_gradient_parallel_batch(
                copy.deepcopy(network),
                target_poisson_list=desired_poisson_list,
                top_nodes=top_nodes,
                bottom_nodes=bottom_nodes,
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                compression_strain_list=source_compression_strain_list,
                epsilon=1e-8,
                n_strain_steps=n_strain_steps,
                n_jobs_outer=4,
                n_jobs_inner=2,
                force_type=force_type,
                tol=force_tol,
            )

        # --- Update stiffnesses ---
        # lr_scale is a pure function of the loss trajectory so far (steps
        # completed before this one) — no normalized-gradient step anymore,
        # and no separately persisted LR state to keep in sync on resume.
        lr_scale, _ = lr_schedule.lr_scale_for_step(history['loss'])
        current_lr = learning_rate * lr_scale
        network.stiffnesses = np.array(network.stiffnesses) + current_lr * np.array(update)
        network.stiffnesses = np.clip(network.stiffnesses, vmin, vmax)

        # Check for NaN in stiffnesses
        if np.isnan(network.stiffnesses).any():
            print(f"\n{'='*60}")
            print(f"WARNING: Stiffnesses contain NaN at step {step}")
            print("Stopping training and saving current results.")
            print(f"{'='*60}")
            # Save intermediate results before breaking
            if save_intermediate:
                save_training_results(
                    task_seed=task_seed,
                    realization_seed=realization_seed,
                    history=history,
                    network=network,
                    task_config=task_config,
                    results_dir=TARGETED_RESULTS_DIR,
                    network_type=network_type,
                )
            break

        # --- Loss computation (post-update, both methods) ---
        # history['loss'][i] and history['stiffnesses'][i] are now aligned:
        # both reflect the network state AFTER the stiffness update at step i.
        network.update_positions(min_pos)
        if method in ('newton', 'newton_fd'):
            loss, _ = compute_ift_gradient(
                network,
                compression_strains=source_compression_strain_list,
                target_poissons=desired_poisson_list,
                top_nodes=top_nodes,
                bottom_nodes=bottom_nodes,
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                tol=force_tol,
                n_strain_steps=n_strain_steps,
            )
        else:
            loss, computed_poisson_ratios = poisson_loss_batch_parallel(
                network,
                target_poisson_list=desired_poisson_list,
                top_nodes=top_nodes,
                bottom_nodes=bottom_nodes,
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                compression_strain_list=source_compression_strain_list,
                n_strain_steps=n_strain_steps,
                force_type=force_type,
                tol=force_tol,
            )

        # --- Update history ---
        history['stiffnesses'].append(np.copy(network.stiffnesses))
        history['loss'].append(loss)
        history['positions'].append(np.copy(min_pos))

        # Check for NaN in loss
        if np.isnan(loss):
            print(f"\n{'='*60}")
            print(f"WARNING: Loss is NaN at step {step}")
            print("Stopping training and saving current results.")
            print(f"{'='*60}")
            # Save intermediate results before breaking
            if save_intermediate:
                save_training_results(
                    task_seed=task_seed,
                    realization_seed=realization_seed,
                    history=history,
                    network=network,
                    task_config=task_config,
                    results_dir=TARGETED_RESULTS_DIR,
                    network_type=network_type,
                )
            break

        # Track minimum loss
        if loss < min_loss:
            min_loss = loss

        # Update progress bar
        pbar.set_description(f'(loss = {loss:.4e}, min loss={min_loss:.4e}), '
                              f'grad_norm = {np.linalg.norm(update):.4e}, '
                              f'lr_scale = {lr_scale:.3g}')

        # Verbose output
        if verbose and step % 100 == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss:.6e}")
            print(f"  Target Poisson ratios: {desired_poisson_list}")

        # Periodically save intermediate trajectories and checkpoint
        if save_intermediate and step % save_interval == 0:
            # Update best trajectory when the best step has changed since last save
            _cur_best = int(np.argmin(history['loss']))
            if _cur_best != _last_traj_best_step:
                _bn = copy.copy(network)
                _bn.stiffnesses = np.array(history['stiffnesses'][_cur_best])
                _bn.update_positions(np.array(history['positions'][_cur_best]))
                history['best_trajectory'] = compute_quasistatic_trajectory_auxetic(
                    _bn, min(source_compression_strain_list), top_nodes, bottom_nodes,
                    n_steps=n_strain_steps, tol=force_tol,
                    force_type=force_type, method=traj_method,
                )
                history['best_step'] = _cur_best
                _last_traj_best_step = _cur_best

            save_training_results(
                task_seed=task_seed,
                realization_seed=realization_seed,
                history=history,
                network=network,
                task_config=task_config,
                results_dir=TARGETED_RESULTS_DIR,
                network_type=network_type,
            )

            save_checkpoint(
                task_seed=task_seed,
                realization_seed=realization_seed,
                history=history,
                network=network,
                task_config=task_config,
                current_step=len(history['loss']),  # global step count, not
                                                      # loop-local `step` — see
                                                      # module note on resume.
                results_dir=TARGETED_RESULTS_DIR,
                network_type=network_type,
            )

        if loss < loss_tol:
            break

    # --- Best-step trajectory (final update) ---
    # Recompute only if the best step changed since the last checkpoint save,
    # or if no trajectory was computed yet (e.g., save_intermediate=False).
    best_step = int(np.argmin(history['loss']))
    if best_step != _last_traj_best_step:
        best_net = copy.copy(network)
        best_net.stiffnesses = np.array(history['stiffnesses'][best_step])
        best_net.update_positions(np.array(history['positions'][best_step]))
        history['best_trajectory'] = compute_quasistatic_trajectory_auxetic(
            best_net, min(source_compression_strain_list), top_nodes, bottom_nodes,
            n_steps=n_strain_steps, tol=force_tol,
            force_type=force_type, method=traj_method,
        )
        history['best_step'] = best_step

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final loss: {loss:.6e}")
    print(f"  Minimum loss: {min_loss:.6e}")
    print(f"  Best step: {best_step}  (trajectory stored in history['best_trajectory'])")
    print(f"{'='*60}")

    trained_network = network
    trained_network.update_positions(last_relaxed_positions)
    trained_network.stiffnesses = np.array(network.stiffnesses)

    return history, trained_network


# ============================================================================
# JAX-BASED TRAINING FUNCTION (autodiff gradients)
# ============================================================================

def finish_training_GD_auxetic_batch_jax(
    network, history, learning_rate, n_steps,
    top_nodes, bottom_nodes, left_nodes, right_nodes,
    force_type='quadratic', n_strain_steps=100,
    source_compression_strain_list=[0.2], desired_target_extension_list=[0.2],
    verbose=False, force_tol=1e-6,
    vmin=1e-6, vmax=1e3,
    task_seed=None, realization_seed=None, save_interval=10,
    task_config=None, TARGETED_RESULTS_DIR=None,
    fire_max_steps=100_000, fire_tol=FORCE_TOL, network_type=NETWORK_TYPE, loss_tol=1e-6,
    fire_dt_max=1.0, fire_finc=1.3, fire_dt_init=1e-2,
    opt_fire=False, opt_fire_dt_max=None, opt_fire_dt_min=None,
    opt_fire_alpha_start=0.1, opt_fire_finc=1.05, opt_fire_fdec=0.5, opt_fire_falpha=0.99,
):
    """
    Train the network for auxetic response using JAX autodiff gradients.

    Same goal as finish_training_GD_auxetic_batch but replaces finite-difference
    gradients with jax.grad through a differentiable FIRE solver (custom VJP).
    The gradient function is JIT-compiled once at the start.

    Args:
        network: ElasticNetwork object to train
        history: Dictionary to store training history
        learning_rate: Gradient descent learning rate
        n_steps: Number of training steps
        top_nodes, bottom_nodes, left_nodes, right_nodes: Boundary node indices
        force_type: 'quadratic' or 'quartic'
        n_strain_steps: Steps per quasistatic trajectory
        source_compression_strain_list: List of compression strains
        desired_target_extension_list: List of target lateral extensions
        verbose: Print detailed progress
        force_tol: Convergence tolerance for free FIRE minimization
        vmin, vmax: Bounds for stiffness values
        task_seed, realization_seed: For saving intermediate results
        save_interval: Save every N steps
        task_config, TARGETED_RESULTS_DIR: For checkpoint saving
        fire_max_steps: Max steps for JAX FIRE solver
        fire_tol: Convergence tolerance for JAX FIRE solver — applied at every
            strain step of every quasistatic trajectory. No step is
            under-converged.
        fire_dt_max, fire_finc, fire_dt_init: FIRE step-size hyperparameters
            (default max step 0.1, growth factor 1.1, initial step 0.01).
            Tuned defaults here (1.0, 1.3, 0.01) reach the *same* force-tol
            equilibrium in far fewer leapfrog iterations — validated (on the
            real targeted-auxetic networks) to converge to the same trajectory
            branch as the untuned defaults (identical final geometry/edge
            topology), just ~6-9x faster. dt_max=2.0 was tested and found to
            destabilize the integrator (diverges to NaN) — stay well under it;
            these defaults have margin. See docs/jax_solver_speedup.md.
        loss_tol: Early-stopping threshold — training stops once loss drops below this
        opt_fire: If True, replace the learning_rate/lr_schedule gradient-descent
            update with a FIRE-style adaptive step on the stiffness "landscape" —
            same P=vel.f power-criterion algorithm already used for the
            physics position relaxation (base.simulate.make_compute_response_fire),
            applied here to (stiffnesses, -grad) instead of (positions, force):
            velocity is bent toward the downhill direction and dt grows while
            consecutive steps keep making progress (P>=0); the moment a step
            would go uphill (P<0), velocity resets to zero and dt collapses —
            guarding against exactly the failure mode a naive accumulated-
            velocity update would show (carrying momentum through a bad step
            into a much harder-to-relax configuration), using the same
            already-validated mechanism as the physics solver rather than a
            new one. Self-adapts every step, so lr_schedule's 1000-step decay
            is bypassed entirely in this mode. Default off — opt-in only.
        opt_fire_dt_max, opt_fire_dt_min: Step-size bounds for opt_fire.
            Default to 10*learning_rate and 1e-5*learning_rate (None triggers
            these defaults) — learning_rate is reused as opt_fire's dt_init,
            so the calibrated starting LR still sets the initial scale.
            dt_min was originally 1e-3*learning_rate but production runs
            (2026-08-14, targeted task 2 / general task 27) showed FIRE
            trapped in an exact repeating limit cycle right at that floor:
            a few finc=1.05 growth steps followed by one P<0 step that resets
            dt straight back to the floor, forever, with zero net loss
            progress — the floor itself was too coarse to take a step small
            enough to get past whatever local discontinuity (stiffness clip
            or bond activation) triggers the bad step. Lowering the floor
            100x gives it room to shrink further before being forced to
            reset, instead of oscillating against a wall.
        opt_fire_alpha_start, opt_fire_finc, opt_fire_fdec, opt_fire_falpha:
            Standard FIRE hyperparameters (velocity-mixing rate, dt growth/
            shrink factors, alpha decay) — defaults match
            make_compute_response_fire's own defaults.

    Returns:
        (history, trained_network)
    """
    network = copy.copy(network)
    last_relaxed_positions = np.copy(network.positions)
    loss = np.inf
    min_loss = np.inf
    velocity = np.zeros(len(network.stiffnesses))  # opt_fire's velocity state; unused for plain GD

    _opt_dt = learning_rate
    _opt_alpha = opt_fire_alpha_start
    _opt_dt_max = opt_fire_dt_max if opt_fire_dt_max is not None else 10 * learning_rate
    _opt_dt_min = opt_fire_dt_min if opt_fire_dt_min is not None else 1e-5 * learning_rate

    # Initialize history
    for key in ('stiffnesses', 'loss', 'positions'):
        if key not in history:
            history[key] = []

    save_intermediate = (task_seed is not None and realization_seed is not None)

    # Convert extensions to target Poisson ratios
    desired_poisson_list = [
        -(ext / cs)
        for cs, ext in zip(source_compression_strain_list, desired_target_extension_list)
    ]

    # Build differentiable FIRE solver — same tol at every strain step as the
    # original (no step is under-converged); only the step-size hyperparameters
    # are tuned (see fire_dt_max/fire_finc docstring above) so the *same*
    # force-balanced equilibrium is reached in far fewer iterations. Network
    # topology is known concretely here (before anything becomes a JAX
    # tracer), so pass it through to enable the sparse IFT-adjoint backward.
    _edges_np = np.array(network.edges, dtype=np.int32)
    _n_dof_np = 2 * len(network.positions)
    _boundary_np = np.concatenate([np.asarray(top_nodes), np.asarray(bottom_nodes)])
    _source_dof_np = np.concatenate([_boundary_np * 2, _boundary_np * 2 + 1])
    crf_local = make_compute_response_fire(
        d=2, force_type=force_type,
        max_steps=fire_max_steps, tol=fire_tol,
        dt_max=fire_dt_max, finc=fire_finc, dt_init=fire_dt_init,
        edges_np=_edges_np, n_dof_np=_n_dof_np, source_nodes_dof_np=_source_dof_np,
    )

    # Pre-convert static arrays to JAX
    edges_jax = jnp.asarray(np.array(network.edges, dtype=np.int32), dtype=jnp.int32)
    rest_lengths_jax = jnp.asarray(np.array(network.rest_lengths, dtype=np.float64), dtype=jnp.float64)
    top_nodes_jax = jnp.asarray(top_nodes, dtype=jnp.int32)
    bottom_nodes_jax = jnp.asarray(bottom_nodes, dtype=jnp.int32)
    left_nodes_jax = jnp.asarray(left_nodes, dtype=jnp.int32)
    right_nodes_jax = jnp.asarray(right_nodes, dtype=jnp.int32)

    # Build and JIT the loss+grad function
    # positions_flat is passed as argument (changes each step after free relaxation)
    def loss_fn(stiffnesses_jax, positions_flat_jax):
        return poisson_loss_batch_jax(
            crf_local, stiffnesses_jax, edges_jax, rest_lengths_jax, positions_flat_jax,
            top_nodes_jax, bottom_nodes_jax, left_nodes_jax, right_nodes_jax,
            source_compression_strain_list, desired_poisson_list, n_strain_steps, d=2
        )

    # jax.value_and_grad: returns (loss, grad) in one forward+backward pass
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))

    pbar = tqdm(range(n_steps), desc=f'(loss = {loss:.4e}, min loss={min_loss:.4e})')

    for step in pbar:
        # --- Free-minimize positions at current stiffnesses (Cython FIRE) ---
        network.update_positions(last_relaxed_positions)
        min_pos, force_norm = fire_minimize_network(
            network,
            constrained_dof_idx=None,
            force_type=force_type,
            tol=force_tol
        )

        if force_norm is not None:
            assert force_norm < force_tol, f"FIRE did not converge: {force_norm:.3e} > {force_tol:.3e}"

        last_relaxed_positions = min_pos
        network.update_positions(min_pos)

        # --- JAX autodiff gradient ---
        stiffnesses_jax = jnp.asarray(np.array(network.stiffnesses, dtype=np.float64), dtype=jnp.float64)
        positions_flat_jax = jnp.asarray(min_pos.flatten(), dtype=jnp.float64)

        loss_val, grad = loss_and_grad_fn(stiffnesses_jax, positions_flat_jax)
        loss = float(loss_val)
        grad_np = np.array(grad)

        # --- Update history (before stiffness update, so loss and stiffnesses match) ---
        history['stiffnesses'].append(np.copy(network.stiffnesses))
        history['loss'].append(loss)
        history['positions'].append(np.copy(min_pos))

        # --- Update stiffnesses ---
        grad_norm = np.linalg.norm(grad_np)
        if opt_fire:
            # FIRE power criterion (mirrors make_compute_response_fire's
            # body_fn): f is the "force" (downhill direction). P>=0 means
            # this step's velocity still points downhill — bend it toward f
            # and grow dt; P<0 means it doesn't — reset velocity to zero and
            # collapse dt, exactly the guard plain momentum was missing.
            f = -grad_np
            P = float(np.dot(velocity, f))
            if P >= 0:
                vnorm = np.linalg.norm(velocity)
                fnorm = np.linalg.norm(f)
                if fnorm > 0:
                    velocity = (1 - _opt_alpha) * velocity + _opt_alpha * f * (vnorm / fnorm)
                _opt_dt = min(_opt_dt * opt_fire_finc, _opt_dt_max)
                _opt_alpha *= opt_fire_falpha
            else:
                velocity = np.zeros_like(velocity)
                _opt_dt = max(_opt_dt * opt_fire_fdec, _opt_dt_min)
                _opt_alpha = opt_fire_alpha_start
            velocity = velocity + _opt_dt * f
            current_lr = _opt_dt
            lr_scale = _opt_dt / learning_rate
            network.stiffnesses = np.array(network.stiffnesses) + _opt_dt * velocity
        else:
            # Plain gradient descent. lr_scale is a pure function of the loss
            # trajectory so far (see module note on resume) — no normalized-
            # gradient step, no momentum.
            lr_scale, _ = lr_schedule.lr_scale_for_step(history['loss'])
            current_lr = learning_rate * lr_scale
            network.stiffnesses = np.array(network.stiffnesses) - current_lr * grad_np
        network.stiffnesses = np.clip(network.stiffnesses, vmin, vmax)

        # Check for NaN
        if np.isnan(network.stiffnesses).any() or np.isnan(loss):
            label = "stiffnesses" if np.isnan(network.stiffnesses).any() else "loss"
            print(f"\n{'='*60}")
            print(f"WARNING: {label} contain NaN at step {step}")
            print("Stopping training and saving current results.")
            print(f"{'='*60}")
            if save_intermediate:
                save_training_results(
                    task_seed=task_seed, realization_seed=realization_seed,
                    history=history, network=network,
                    task_config=task_config, results_dir=TARGETED_RESULTS_DIR,
                    network_type=network_type,
                )
            break

        if loss < min_loss:
            min_loss = loss

        pbar.set_description(f'(loss = {loss:.4e}, min loss={min_loss:.4e}, init loss = {history["loss"][0]:.4e}, '
                              f'grad_norm = {grad_norm:.4e}, lr_scale = {lr_scale:.3g})')

        if verbose and step % save_interval == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss:.6e}")
            print(f"  Target Poisson ratios: {desired_poisson_list}")
            print(f"  Grad norm: {grad_norm:.6e}")

        if save_intermediate and step % save_interval == 0:
            save_training_results(
                task_seed=task_seed, realization_seed=realization_seed,
                history=history, network=network,
                task_config=task_config, results_dir=TARGETED_RESULTS_DIR,
                network_type=network_type,
            )
            save_checkpoint(
                task_seed=task_seed, realization_seed=realization_seed,
                history=history, network=network,
                task_config=task_config, current_step=len(history['loss']),
                results_dir=TARGETED_RESULTS_DIR,
                network_type=network_type,
            )

        if loss < loss_tol:
            break

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final loss: {loss:.6e}")
    print(f"  Minimum loss: {min_loss:.6e}")
    print(f"{'='*60}")

    trained_network = network
    trained_network.update_positions(last_relaxed_positions)
    trained_network.stiffnesses = np.array(network.stiffnesses)

    return history, trained_network

