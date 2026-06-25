import numpy as np
import jax
import jax.numpy as jnp

class ElasticNetwork:
    def get_energy(self, force_type="quadratic"):
        """
        Compute the total energy of the network for the specified force law.
        Supports 'hookean' and 'quartic' (with scaling) energy.
        """
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        dists = np.linalg.norm(self.positions[j] - self.positions[i], axis=1)
        delta = dists - self.rest_lengths
        if force_type == "quadratic":
            energy = 0.5 * np.sum(self.stiffnesses * delta ** 2)
        elif force_type == "quartic":
            avg_rest_length = np.mean(self.rest_lengths)
            scale_coeff = 36.0 / (avg_rest_length ** 2 + 1e-12)
            energy = 0.25 * np.sum(self.stiffnesses * (scale_coeff * delta ** 4 - delta ** 2))
        else:
            raise ValueError(f"Unknown force_type: {force_type}")
        return energy
    @staticmethod
    @jax.jit
    def jax_hookean_energy_fn(positions, rest_lengths, stiffnesses, edges):
        pos_i = positions[edges[:, 0]]
        pos_j = positions[edges[:, 1]]
        dists = jnp.linalg.norm(pos_j - pos_i, axis=1)
        energy = 0.5 * jnp.sum(stiffnesses * (dists - rest_lengths) ** 2)
        return energy

    def get_jax_hookean_energy_fn(self):
        """
        Returns a stateless, JIT-compiled energy function for use in training/optimization.
        Usage: energy_fn = network.get_jax_energy_fn()
        """
        rest_lengths = jnp.array(self.rest_lengths)
        stiffnesses = jnp.array(self.stiffnesses)
        edges = jnp.array(self.edges)
        def energy(positions):
            return ElasticNetwork.jax_energy_fn(positions, rest_lengths, stiffnesses, edges)
        return jax.jit(energy)
    def save_original_parameters(self):
        """
        Save the original rest lengths and stiffnesses for later reference.
        """
        self.original_rest_lengths = np.copy(self.rest_lengths)
        self.original_stiffnesses = np.copy(self.stiffnesses)
        self.original_positions = np.copy(self.positions)

    def get_original_rest_length(self, edge_idx):
        return self.original_rest_lengths[edge_idx]

    def get_original_stiffness(self, edge_idx):
        return self.original_stiffnesses[edge_idx]
    def __init__(self, positions, edges, rest_lengths=None, stiffnesses=None):
        self.positions = np.array(positions, dtype=float)
        self.edges = np.array(edges, dtype=int)
        if rest_lengths is None:
            self.rest_lengths = np.linalg.norm(self.positions[self.edges[:, 1]] - self.positions[self.edges[:, 0]], axis=1)
        else:
            self.rest_lengths = np.array(rest_lengths, dtype=float)
        if stiffnesses is None:
            self.stiffnesses = np.ones(len(self.edges))
        else:
            self.stiffnesses = np.array(stiffnesses, dtype=float)

    def compute_hookean_forces(self):
        forces = np.zeros_like(self.positions)
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        vec = self.positions[j] - self.positions[i]  # shape (n_edges, dim)
        dist = np.linalg.norm(vec, axis=1)  # shape (n_edges,)
        # Avoid division by zero
        mask = dist > 0
        force_mag = np.zeros_like(dist)
        force_mag[mask] = self.stiffnesses[mask] * (dist[mask] - self.rest_lengths[mask])
        force_vec = np.zeros_like(vec)
        force_vec[mask] = (force_mag[mask][:, None] * vec[mask]) / dist[mask][:, None]
        np.add.at(forces, i, force_vec)
        np.add.at(forces, j, -force_vec)
        return forces

    def update_positions(self, new_positions):
        self.positions = np.array(new_positions, dtype=float)

    def set_rest_length(self, edge_idx, new_length):
        self.rest_lengths[edge_idx] = new_length

    def get_node_position(self, node_idx):
        return self.positions[node_idx]

    def get_edge_length(self, edge_idx):
        i, j = self.edges[edge_idx]
        return np.linalg.norm(self.positions[j] - self.positions[i])

    def get_all_edge_lengths(self):
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        return np.linalg.norm(self.positions[j] - self.positions[i], axis=1)

    def get_strain(self, edge_idx):
        return (self.get_edge_length(edge_idx) - self.rest_lengths[edge_idx]) / self.rest_lengths[edge_idx]

    def get_all_strains(self):
        edge_lengths = self.get_all_edge_lengths()
        return (edge_lengths - self.rest_lengths) / self.rest_lengths

    def get_hookean_energy(self):
        # Standard numpy version
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        dists = np.linalg.norm(self.positions[j] - self.positions[i], axis=1)
        energy = 0.5 * np.sum(self.stiffnesses * (dists - self.rest_lengths) ** 2)
        return energy



    def get_hookean_energy_jax(self):
        # JIT-compiled version for current state
        return float(self.get_jax_hookean_energy_fn()(jnp.array(self.positions)))
