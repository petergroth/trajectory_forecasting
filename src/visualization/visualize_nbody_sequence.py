import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.patches import Circle

from src.utils import load_simulations



if __name__ == "__main__":
    rc("text", usetex=True)
    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

    ticksize = 20
    titlesize = 25
    plt.rcParams["axes.labelsize"] = titlesize
    plt.rcParams["axes.titlesize"] = titlesize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    # Visualise grid of 6 sequences
    sequence_idxs = [2, 4, 7, 6] #, 8, 9]
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))

    n_steps = 51
    alphas = np.linspace(0.1, 1, n_steps)
    colors = [
        "darkorange",
        "salmon",
        "sandybrown",
        "chocolate",
        "crimson",
        "darkred",
        "orangered",
        "orange",
        "peru",
        "peru",
    ]

    for counter, sequence_idx in enumerate(sequence_idxs):
        path = f"data/raw/nbody/nbody_10_particles_sim000{sequence_idx}.npy"
        full_arr = load_simulations(path=path)
        positions = full_arr[:, :, :2]
        velocities = full_arr[:, :, 2:4]
        sizes = full_arr[:, :, 4]
        n_particles = positions.shape[1]

        for t in range(n_steps):
            for i in range(n_particles):
                ax.flatten()[counter].add_patch(
                    Circle(
                        positions[t, i, :],
                        sizes[0, i],
                        facecolor=colors[i],
                        alpha=alphas[t],
                    )
                )

        for i in range(n_particles):
            ax.flatten()[counter].add_patch(
                Circle(
                    positions[n_steps - 1, i, :],
                    sizes[0, i],
                    edgecolor="k",
                    facecolor=colors[i],
                    alpha=alphas[t],
                )
            )

        ax.flatten()[counter].quiver(
            positions[n_steps - 1, :, 0],
            positions[n_steps - 1, :, 1],
            velocities[n_steps - 1, :, 0],
            velocities[n_steps - 1, :, 1],
            width=0.003,
            headwidth=5,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="k",
            alpha=1,
            zorder=2,
        )

        # if counter not in [4, 5]:
        ax.flatten()[counter].axes.xaxis.set_visible(False)
        # if counter in [1, 3, 5]:
        ax.flatten()[counter].axes.yaxis.set_visible(False)

        ax.flatten()[counter].set_xlim((-100, 100))
        ax.flatten()[counter].set_ylim((-100, 100))


        # plt.title(r"Simulation of $\textit{n}$-body problem")
        # ax.set_xlabel(r"$\textit{x}$")
        # ax.set_ylabel(r"$\textit{y}$", rotation="horizontal")

    plt.tight_layout()
    plt.savefig(f"../../thesis/graphics/synthetic/nbody_grid_example.pdf")
    plt.savefig(f"visualisations/nbody/nbody_grid_example.pdf")

    # Visualise single n-body sequence
    fig, ax = plt.subplots(figsize=(10, 10))
    sequence_idx = 1
    path = f"data/raw/nbody/nbody_10_particles_sim1505.npy"
    full_arr = load_simulations(path=path)
    positions = full_arr[:, :, :2]
    velocities = full_arr[:, :, 2:4]
    sizes = full_arr[:, :, 4]
    n_particles = positions.shape[1]

    for t in range(n_steps):
        for i in range(n_particles):
            ax.add_patch(
                Circle(
                    positions[t, i, :],
                    sizes[0, i],
                    facecolor=colors[i],
                    alpha=alphas[t],
                )
            )

    for i in range(n_particles):
        ax.add_patch(
            Circle(
                positions[n_steps - 1, i, :],
                sizes[0, i],
                edgecolor="k",
                facecolor=colors[i],
                alpha=alphas[t],
            )
        )

    ax.quiver(
        positions[n_steps - 1, :, 0],
        positions[n_steps - 1, :, 1],
        velocities[n_steps - 1, :, 0],
        velocities[n_steps - 1, :, 1],
        width=0.003,
        headwidth=5,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="k",
        alpha=1,
        zorder=2,
    )

    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    ax.set_xlim((-120, 80))
    ax.set_ylim((-75, 125))
    # plt.title(r"Simulation of $\textit{n}$-body problem")
    ax.set_xlabel(r"$\textit{x}$")
    ax.set_ylabel(r"$\textit{y}$", rotation="horizontal")

    plt.tight_layout()
    # plt.show()


    plt.savefig(f"../../thesis/graphics/synthetic/nbody_example.pdf")
    plt.savefig(f"visualisations/nbody/nbody_example.pdf")
