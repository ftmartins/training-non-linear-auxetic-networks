import numpy as np
from lammps import lammps
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- Helper Functions ----------------
def incidence_to_edges(incidence):
    edges = []
    for row in incidence:
        nodes = np.where(row != 0)[0]
        if len(nodes) != 2:
            raise ValueError("Each row must have exactly 2 nonzeros")
        edges.append((nodes[0], nodes[1]))
    return np.array(edges, dtype=int)

def write_lammps_data(filename, positions, incidence, stiffnesses,
                                  id_outA=None, id_outB=None,
                                  target_output_distance=None,
                                  k_output=1e3, mass=1.0):
    N = positions.shape[0]
    edges = incidence_to_edges(incidence)
    M = len(edges)
    bond_coeff_name = "bond_coeffs_free.in"
    
    # Add output bond if specified
    if id_outA is not None and id_outB is not None and target_output_distance is not None:
        edges = np.vstack((edges,(id_outA, id_outB)))
        stiffnesses = np.append(stiffnesses, k_output)
        bond_coeff_name = "bond_coeffs_clamped.in"
        M += 1
    
    with open(filename, "w") as f:
        f.write("Elastic network with output spring\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"{M} bonds\n\n")
        f.write("1 atom types\n")
        f.write(f"{M} bond types\n\n")

        # Box
        margin = 5.0
        pos3d = np.hstack([positions, np.zeros((N,1))])
        mins = pos3d.min(axis=0) - margin
        maxs = pos3d.max(axis=0) + margin
        f.write(f"{mins[0]} {maxs[0]} xlo xhi\n")
        f.write(f"{mins[1]} {maxs[1]} ylo yhi\n")
        f.write(f"-0.1 0.1 zlo zhi\n\n")

        # Masses
        f.write("Masses\n\n")
        f.write(f"1 {mass}\n\n")

        # Atoms
        f.write("Atoms\n\n")
        for i, (x,y,z) in enumerate(pos3d, start=1):
            f.write(f"{i} 1 1 {x:.6f} {y:.6f} {z:.6f}\n")

        # Bonds
        f.write("\nBonds\n\n")
        for bi, (i,j) in enumerate(edges, start=1):
            f.write(f"{bi} {bi} {i+1} {j+1}\n")

    # Bond coefficients
    with open(bond_coeff_name, "w") as f:
        for bi, (i,j) in enumerate(edges, start=1):
            xi, yi = positions[i]
            xj, yj = positions[j]
            if id_outA is not None and id_outB is not None and set([i,j]) == set([id_outA,id_outB]):
                r0 = target_output_distance
                k = k_output
            else:
                r0 = np.linalg.norm([xj-xi, yj-yi])
                k = stiffnesses[bi-1]
            f.write(f"bond_coeff {bi} {k:.6f} {r0:.6f}\n")


# ---------------- Quasi-static Pulling ----------------
def strain_network(datafile, id_fixed, id_pull, clamped = False, dx=0.025, nsteps=200):
    """
    Quasi-static pulling of a single node in a 2D network.

    Parameters:
        datafile: LAMMPS data file path
        id_fixed: index of the fixed node (0-based)
        id_pull: index of the pulled node (0-based)
        dx: displacement per step along x
        nsteps: number of steps
        total_dx: maximum imposed displacement (for box margin check)
    Returns:
        data: array with step, extension, potential energy, Fx, Fy
        frames: list of node positions at each step
    """
    lmp = lammps()
    lmp.command("units lj")
    lmp.command("atom_style bond")
    lmp.command("dimension 3")
    lmp.command("boundary s s s")
    lmp.command(f"read_data {datafile}")
    lmp.command("bond_style harmonic")
    lmp.command("pair_style none")
    if clamped == False:
        lmp.command("include bond_coeffs_free.in")
    else:
        lmp.command("include bond_coeffs_clamped.in")

    N = lmp.get_natoms()

    # --- Robust comm / neighbor settings for large strain ---
    lmp.command("comm_modify mode single cutoff 10.0")
    lmp.command("neighbor 3.0 bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")

    # Fix z for all atoms
    lmp.command("fix freeze_z all setforce NULL NULL 0.0")
    lmp.command("velocity all set 0.0 0.0 0.0")

    # Groups
    lmp.command(f"group fixed id {id_fixed+1} {id_pull+1}")
    lmp.command("fix hold fixed setforce 0.0 0.0 0.0")
    lmp.command("group free subtract all fixed")

    frames = []

    # Gather initial positions
    coords = np.array(lmp.gather_atoms("x", 1, 3)).reshape(N, 3)

    for step in range(nsteps):
        # Move pulled node horizontally only
        coords[id_pull, 0] += dx

        # Update pulled node coordinates explicitly
        lmp.command(f"set atom {id_pull+1} x {coords[id_pull,0]:.6f}")
        lmp.command(f"set atom {id_pull+1} y {coords[id_pull,1]:.6f}")
        lmp.command(f"set atom {id_pull+1} z {coords[id_pull,2]:.6f}")

        # Force domain/neighbors to update before minimize
        lmp.command("run 0 post no")

        # Minimize energy
        lmp.command("min_style cg")
        lmp.command("minimize 1e-8 1e-8 1000 10000")

        # Re-freeze x,y of pulled node after relaxation
        lmp.command(f"set atom {id_pull+1} x {coords[id_pull,0]:.6f}")
        lmp.command(f"set atom {id_pull+1} y {coords[id_pull,1]:.6f}")
        lmp.command("run 0 post no")

        # Record positions
        coords = np.array(lmp.gather_atoms("x", 1, 3)).reshape(N, 3)
        frames.append(coords[:, :2].copy())
    return frames


def make_video(frames, incidence, stiffnesses,
                                        id_fixed, id_pull,
                                        id_outA=None, id_outB=None,
                                        filename="pulling_network.mp4",
                                        interval=50):
    edges = []
    for row in incidence:
        nodes = np.where(row != 0)[0]
        edges.append((nodes[0], nodes[1]))
    edges = np.array(edges)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.grid(True)

    all_coords = np.vstack(frames)
    xmin, xmax = all_coords[:,0].min()-0.5, all_coords[:,0].max()+0.5
    ymin, ymax = all_coords[:,1].min()-0.5, all_coords[:,1].max()+0.5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Bond thickness
    min_lw, max_lw = 1.0, 4.0
    kmin, kmax = stiffnesses.min(), stiffnesses.max()
    lw = min_lw + (stiffnesses - kmin)/(kmax - kmin + 1e-12)*(max_lw - min_lw)

    scat_fixed  = ax.scatter([], [], s=150, color='green', label='Fixed')
    scat_pulled = ax.scatter([], [], s=150, color='red', label='Pulled')
    scat_free   = ax.scatter([], [], s=100, color='blue', label='Free')
    if id_outA is not None and id_outB is not None:
        scat_output = ax.scatter([], [], s=150, color='orange', label='Output pair')

    bond_lines = []
    bond_colors = []
    bond_widths = []
    for (i,j), width in zip(edges, lw):
        # Highlight output pair bond
        if id_outA is not None and id_outB is not None and set([i,j]) == set([id_outA,id_outB]):
            line, = ax.plot([], [], 'orange', lw=6)  # thick and orange
        else:
            line, = ax.plot([], [], 'k-', lw=width)
        bond_lines.append(line)

    ax.legend(loc='upper right')
    all_indices = np.arange(frames[0].shape[0])

    def init():
        frame = frames[0]
        free_idx = all_indices[~np.isin(all_indices,[id_fixed,id_pull,id_outA,id_outB])]
        scat_fixed.set_offsets(frame[[id_fixed],:])
        scat_pulled.set_offsets(frame[[id_pull],:])
        scat_free.set_offsets(frame[free_idx,:])
        if id_outA is not None and id_outB is not None:
            scat_output.set_offsets(frame[[id_outA,id_outB],:])
        for line, (i,j) in zip(bond_lines, edges):
            line.set_data([frame[i,0],frame[j,0]], [frame[i,1],frame[j,1]])
        return bond_lines + [scat_fixed, scat_pulled, scat_free] + ([scat_output] if id_outA is not None else [])

    def update(frame):
        free_idx = all_indices[~np.isin(all_indices,[id_fixed,id_pull,id_outA,id_outB])]
        scat_fixed.set_offsets(frame[[id_fixed],:])
        scat_pulled.set_offsets(frame[[id_pull],:])
        scat_free.set_offsets(frame[free_idx,:])
        if id_outA is not None and id_outB is not None:
            scat_output.set_offsets(frame[[id_outA,id_outB],:])
        for line, (i,j) in zip(bond_lines, edges):
            line.set_data([frame[i,0],frame[j,0]], [frame[i,1],frame[j,1]])
        return bond_lines + [scat_fixed, scat_pulled, scat_free] + ([scat_output] if id_outA is not None else [])

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  init_func=init, blit=True, interval=interval)
    ani.save(filename, writer='ffmpeg', dpi=200)
    plt.close(fig)