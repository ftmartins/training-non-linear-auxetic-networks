"""
Smoke test for the JAX-based differentiable training pipeline.
Creates a small network, runs a few training steps, then replays the
stiffness history through the Cython-based loss to verify both methods agree.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from network_utils import create_auxetic_network
from training_functions_with_toggle import (
    finish_training_GD_auxetic_batch_jax,
    poisson_loss_batch_parallel,
    fire_minimize_network,
)


def main():
    # --- Create a small network ---
    print("Creating network...")
    network, boundary_dict = create_auxetic_network(
        n_nodes=64,
        packing_seed=42,
        force_type='quadratic',
        boundary_margin=0.05,
    )
    print(f"  Nodes: {len(network.positions)}, Edges: {len(network.edges)}")
    print(f"  Top: {len(boundary_dict['top'])}, Bottom: {len(boundary_dict['bottom'])}, "
          f"Left: {len(boundary_dict['left'])}, Right: {len(boundary_dict['right'])}")

    # --- Training config ---
    history = {}
    compression_strains = [-0.2]
    target_extensions = [0.1]  # want lateral expansion -> target nu < 0
    desired_poisson_list = [-(e / c) for c, e in zip(compression_strains, target_extensions)]
    n_strain_steps = 80
    force_type = 'quadratic'

    print(f"\nTarget Poisson ratios: {desired_poisson_list}")
    print("Starting JAX training (10 steps)...\n")

    # Save original network for replay
    network_original = copy.deepcopy(network)

    history, trained_network = finish_training_GD_auxetic_batch_jax(
        network=network,
        history=history,
        learning_rate=0.1,
        n_steps=10,
        top_nodes=boundary_dict['top'],
        bottom_nodes=boundary_dict['bottom'],
        left_nodes=boundary_dict['left'],
        right_nodes=boundary_dict['right'],
        force_type=force_type,
        n_strain_steps=n_strain_steps,
        source_compression_strain_list=compression_strains,
        desired_target_extension_list=target_extensions,
        verbose=True,
        fire_max_steps=100_000,
        fire_tol=1e-6,
    )

    jax_losses = history['loss']
    print("\nJAX loss history:", [f"{l:.6e}" for l in jax_losses])

    # --- Replay stiffness history through Cython-based loss ---
    print("\nReplaying stiffness history with Cython-based loss...")
    cython_losses = []
    replay_network = copy.deepcopy(network_original)

    for i, stiffnesses_i in enumerate(history['stiffnesses']):
        replay_network.stiffnesses = np.copy(stiffnesses_i)

        # Free-minimize at these stiffnesses
        min_pos, _ = fire_minimize_network(
            replay_network,
            constrained_dof_idx=None,
            force_type=force_type,
            tol=1e-5,
        )
        replay_network.update_positions(min_pos)

        # Compute loss with Cython trajectory
        loss_cy, _ = poisson_loss_batch_parallel(
            replay_network,
            target_poisson_list=desired_poisson_list,
            top_nodes=boundary_dict['top'],
            bottom_nodes=boundary_dict['bottom'],
            left_nodes=boundary_dict['left'],
            right_nodes=boundary_dict['right'],
            compression_strain_list=compression_strains,
            n_strain_steps=n_strain_steps,
            n_jobs_inner=1,
            force_type=force_type,
        )
        cython_losses.append(loss_cy)
        print(f"  Step {i}: JAX={jax_losses[i]:.6e}  Cython={loss_cy:.6e}")

    # --- Plot comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(len(jax_losses))
    ax.plot(steps, jax_losses, 'o-', label='JAX (during training)', markersize=6)
    ax.plot(steps, cython_losses, 's--', label='Cython (replay)', markersize=6)
    ax.set_xlabel('Training step')
    ax.set_ylabel('MSE loss')
    ax.set_title('Loss comparison: JAX vs Cython')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('test_jax_vs_cython_loss.png', dpi=150)
    print(f"\nPlot saved to test_jax_vs_cython_loss.png")
    plt.show()

    print("Test complete.")


if __name__ == "__main__":
    main()
