import os
import re
import tempfile
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lammps import lammps

_STOPPING_CRITERION_RE = re.compile(r"Stopping criterion\s*=\s*(.+)")


warnings.filterwarnings("ignore")

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
                      k_output=1e3, mass=1.0, work_dir=None):
    N = positions.shape[0]
    edges = incidence_to_edges(incidence)
    M = len(edges)
    bond_coeff_name = "bond_coeffs_free.in"

    if id_outA is not None and id_outB is not None and target_output_distance is not None:
        edges = np.vstack((edges, (id_outA, id_outB)))
        stiffnesses = np.append(stiffnesses, k_output)
        bond_coeff_name = "bond_coeffs_clamped.in"
        M += 1

    if work_dir is not None:
        filename = os.path.join(work_dir, filename)
        bond_coeff_name = os.path.join(work_dir, bond_coeff_name)

    with open(filename, "w") as f:
        f.write("Elastic network with output spring\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"{M} bonds\n\n")
        f.write("1 atom types\n")
        f.write(f"{M} bond types\n\n")

        margin = 5.0
        pos3d = np.hstack([positions, np.zeros((N, 1))])
        mins = pos3d.min(axis=0) - margin
        maxs = pos3d.max(axis=0) + margin
        f.write(f"{mins[0]} {maxs[0]} xlo xhi\n")
        f.write(f"{mins[1]} {maxs[1]} ylo yhi\n")
        f.write(f"-0.1 0.1 zlo zhi\n\n")

        f.write("Masses\n\n")
        f.write(f"1 {mass}\n\n")

        f.write("Atoms\n\n")
        for i, (x, y, z) in enumerate(pos3d, start=1):
            f.write(f"{i} 1 1 {x:.6f} {y:.6f} {z:.6f}\n")

        f.write("\nBonds\n\n")
        for bi, (i, j) in enumerate(edges, start=1):
            f.write(f"{bi} {bi} {i+1} {j+1}\n")

    with open(bond_coeff_name, "w") as f:
        for bi, (i, j) in enumerate(edges, start=1):
            xi, yi = positions[i]
            xj, yj = positions[j]
            if id_outA is not None and id_outB is not None and set([i, j]) == set([id_outA, id_outB]):
                r0 = target_output_distance
                k = k_output
            else:
                r0 = np.linalg.norm([xj - xi, yj - yi])
                k = stiffnesses[bi - 1]
            f.write(f"bond_coeff {bi} {k:.6f} {r0:.6f}\n")


def strain_network(datafile, id_fixed, id_pull, clamped=False, dx=0.025, nsteps=200,
                   work_dir=None):
    """
    Quasi-static pulling of a single node in a 2D network via LAMMPS.

    Returns frames: list of (N, 2) node position arrays, one per step.
    """

    bond_coeffs_free    = "bond_coeffs_free.in"
    bond_coeffs_clamped = "bond_coeffs_clamped.in"
    if work_dir is not None:
        datafile             = os.path.join(work_dir, datafile)
        bond_coeffs_free    = os.path.join(work_dir, bond_coeffs_free)
        bond_coeffs_clamped = os.path.join(work_dir, bond_coeffs_clamped)

    # Route the LAMMPS log to a real file (instead of "none") so that, on
    # non-convergence, we can report *why* minimize stopped (energy
    # tolerance / force tolerance / line search failure / maxiter / maxeval)
    # rather than just the residual fnorm.
    log_dir = work_dir if work_dir is not None else tempfile.mkdtemp(prefix="strain_network_")
    log_path = os.path.join(log_dir, "clamped.log" if clamped else "free.log")

    args = ["-screen", "none", "-log", log_path]
    lmp = lammps(cmdargs=args)
    lmp.command("units lj")
    lmp.command("atom_style bond")
    lmp.command("dimension 3")
    lmp.command("boundary s s s")
    lmp.command(f"read_data {datafile}")
    lmp.command("bond_style harmonic")
    lmp.command("pair_style none")
    if not clamped:
        lmp.command(f"include {bond_coeffs_free}")
    else:
        lmp.command(f"include {bond_coeffs_clamped}")

    N = lmp.get_natoms()

    lmp.command("comm_modify mode single cutoff 10.0")
    lmp.command("neighbor 3.0 bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")

    lmp.command("fix freeze_z all setforce NULL NULL 0.0")
    lmp.command("velocity all set 0.0 0.0 0.0")
    lmp.command(f"group fixed id {id_fixed+1} {id_pull+1}")
    lmp.command("fix hold fixed setforce 0.0 0.0 0.0")
    lmp.command("group free subtract all fixed")

    frames = []
    coords = np.array(lmp.gather_atoms("x", 1, 3)).reshape(N, 3)
    log_pos = 0

    for step in range(nsteps):
        coords[id_pull, 0] += dx
        lmp.command(f"set atom {id_pull+1} x {coords[id_pull,0]:.6f}")
        lmp.command(f"set atom {id_pull+1} y {coords[id_pull,1]:.6f}")
        lmp.command(f"set atom {id_pull+1} z {coords[id_pull,2]:.6f}")
        lmp.command("run 0 post no")
        lmp.command("min_style fire")
        # etol=0.0: don't let the energy-tolerance criterion stop the run —
        # near-isostatic networks can have floppy/near-zero-energy modes
        # where energy stops changing well before force actually vanishes,
        # satisfying etol long before ftol. Only force/line-search/iteration
        # limits should end the minimization now.
        lmp.command("minimize 0.0 1e-8 100000 1000000")
        fnorm = lmp.get_thermo("fnorm")
        with open(log_path) as log_fh:
            log_fh.seek(log_pos)
            new_log_text = log_fh.read()
            log_pos = log_fh.tell()
        if fnorm/3*N > 1e-6:
            reasons = _STOPPING_CRITERION_RE.findall(new_log_text)
            stop_reason = reasons[-1].strip() if reasons else "unknown (log parse failed)"
            warnings.warn(
                f"LAMMPS minimize did not converge at step {step}: fnorm={(fnorm/3*N):.3e} > 1e-6 "
                f"(stopping criterion: {stop_reason})",
                RuntimeWarning,
            )
        lmp.command(f"set atom {id_pull+1} x {coords[id_pull,0]:.6f}")
        lmp.command(f"set atom {id_pull+1} y {coords[id_pull,1]:.6f}")
        lmp.command("run 0 post no")
        coords = np.array(lmp.gather_atoms("x", 1, 3)).reshape(N, 3)
        frames.append(coords[:, :2].copy())

    return frames


def make_video(frames, incidence, stiffnesses,
               id_fixed, id_pull,
               id_outA=None, id_outB=None,
               filename="pulling_network.mp4",
               interval=50):
    edges = incidence_to_edges(incidence)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.grid(True)

    all_coords = np.vstack(frames)
    xmin, xmax = all_coords[:, 0].min() - 0.5, all_coords[:, 0].max() + 0.5
    ymin, ymax = all_coords[:, 1].min() - 0.5, all_coords[:, 1].max() + 0.5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    min_lw, max_lw = 1.0, 4.0
    kmin, kmax = stiffnesses.min(), stiffnesses.max()
    lw = min_lw + (stiffnesses - kmin) / (kmax - kmin + 1e-12) * (max_lw - min_lw)

    scat_fixed = ax.scatter([], [], s=150, color='green', label='Fixed')
    scat_pulled = ax.scatter([], [], s=150, color='red', label='Pulled')
    scat_free = ax.scatter([], [], s=100, color='blue', label='Free')
    if id_outA is not None and id_outB is not None:
        scat_output = ax.scatter([], [], s=150, color='orange', label='Output pair')

    bond_lines = []
    for (i, j), width in zip(edges, lw):
        if id_outA is not None and id_outB is not None and set([i, j]) == set([id_outA, id_outB]):
            line, = ax.plot([], [], 'orange', lw=6)
        else:
            line, = ax.plot([], [], 'k-', lw=width)
        bond_lines.append(line)

    ax.legend(loc='upper right')
    all_indices = np.arange(frames[0].shape[0])

    def _update_frame(frame):
        free_idx = all_indices[~np.isin(all_indices, [id_fixed, id_pull, id_outA, id_outB])]
        scat_fixed.set_offsets(frame[[id_fixed], :])
        scat_pulled.set_offsets(frame[[id_pull], :])
        scat_free.set_offsets(frame[free_idx, :])
        if id_outA is not None and id_outB is not None:
            scat_output.set_offsets(frame[[id_outA, id_outB], :])
        for line, (i, j) in zip(bond_lines, edges):
            line.set_data([frame[i, 0], frame[j, 0]], [frame[i, 1], frame[j, 1]])
        extras = [scat_output] if id_outA is not None else []
        return bond_lines + [scat_fixed, scat_pulled, scat_free] + extras

    ani = animation.FuncAnimation(fig, _update_frame, frames=frames,
                                  init_func=lambda: _update_frame(frames[0]),
                                  blit=True, interval=interval)
    ani.save(filename, writer='ffmpeg', dpi=200)
    plt.close(fig)
