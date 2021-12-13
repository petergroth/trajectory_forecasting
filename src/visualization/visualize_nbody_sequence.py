import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.patches import Circle

from src.utils import load_simulations

rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

ticksize = 20
titlesize = 25
plt.rcParams["axes.labelsize"] = titlesize
plt.rcParams["axes.titlesize"] = titlesize
plt.rcParams["xtick.labelsize"] = ticksize
plt.rcParams["ytick.labelsize"] = ticksize

if __name__ == "__main__":
    path = "data/raw/nbody/nbody_10_particles_sim0003.npy"
    full_arr = load_simulations(path=path)
    positions = full_arr[:, :, :2]
    velocities = full_arr[:, :, 2:4]
    sizes = full_arr[:, :, 4]
    n_particles = positions.shape[1]
    n_steps = 91

    alphas = np.linspace(0.1, 1, n_steps)
    fig, ax = plt.subplots(figsize=(10, 10))
    np.random.seed(1)
    # colors = [
    #     (np.random.random(), np.random.random(), np.random.random())
    #     for _ in range(n_particles)
    # ]

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
    ]

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
                positions[-1, i, :],
                sizes[0, i],
                edgecolor="k",
                facecolor=colors[i],
                alpha=alphas[t],
            )
        )

    # for i in range(n_particles):
    #     ax.add_patch(
    #         Circle(
    #             positions[0, i, :], sizes[0, i], edgecolor="k", facecolor='k', alpha=0.1, linestyle=':'
    #         )
    #     )
    #
    # for i in range(n_particles):
    #     ax.add_patch(
    #         Circle(
    #             positions[-1, i, :], sizes[0, i], edgecolor="k", facecolor=colors[i], alpha=0.4, zorder=2
    #         )
    #     )

    ax.quiver(
        positions[-1, :, 0],
        positions[-1, :, 1],
        velocities[-1, :, 0],
        velocities[-1, :, 1],
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
    ax.set_xlim((-100, 100))
    ax.set_ylim((-100, 100))
    # plt.title(r"Simulation of $\textit{n}$-body problem")
    ax.set_xlabel(r"$\textit{x}$")
    ax.set_ylabel(r"$\textit{y}$", rotation="horizontal")

    # ax.yaxis.set_label_coords(-.1, 0)
    # ax.xaxis.set_label_coords(0, -.1)

    plt.tight_layout()
    plt.savefig(f"../../thesis/graphics/synthetic/nbody_example.pdf")
    # plt.show()
    print("finished")
