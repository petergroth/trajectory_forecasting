import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.patches import Circle
from src.training_modules.train_nbody_model import ConstantPhysicalBaselineModule

rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

ticksize = 20
titlesize = 25
plt.rcParams["axes.labelsize"] = titlesize
plt.rcParams["axes.titlesize"] = titlesize
plt.rcParams["xtick.labelsize"] = ticksize
plt.rcParams["ytick.labelsize"] = ticksize

if __name__ == "__main__":

    # Instantiate model




    positions = full_arr[:, :, :2]
    velocities = full_arr[:, :, 2:4]
    sizes = full_arr[:, :, 4]
    n_particles = positions.shape[1]
    n_steps = 51

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
                positions[n_steps-1, i, :],
                sizes[0, i],
                edgecolor="k",
                facecolor=colors[i],
                alpha=alphas[t],
            )
        )

    ax.quiver(
        positions[n_steps-1, :, 0],
        positions[n_steps-1, :, 1],
        velocities[n_steps-1, :, 0],
        velocities[n_steps-1, :, 1],
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
    # plt.savefig(f"../../thesis/graphics/synthetic/nbody_example.pdf")
    plt.show()
    print("finished")
