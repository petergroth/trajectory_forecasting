import argparse
import os

import hydra
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from matplotlib import rc
from matplotlib.patches import Circle, Ellipse, Rectangle
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from src.data.dataset_waymo import SequentialWaymoDataModule
from src.predictions.make_waymo_predictions import plot_time_step

if __name__ == "__main__":

    # Setup datamodule
    dm = SequentialWaymoDataModule()
    dm.setup("test")
    dataset = dm.test_dataset

    # Extract map
    batch = dataset.__getitem__(1)
    roadgraph = batch.u
    roadgraph[roadgraph > 1] = 1
    roadgraph = batch.u.squeeze().numpy()

    # Extract map information
    loc_x = batch.loc[:, 0].squeeze().numpy()
    loc_y = batch.loc[:, 1].squeeze().numpy()
    width = 150
    extent = (
        loc_x - width / 2,
        loc_x + width / 2,
        loc_y - width / 2,
        loc_y + width / 2,
    )
    x_min, x_max, y_min, y_max = extent

    # Extract groundtruth trajectories
    type_mask = batch.type[:, 1] == 1
    batch.x = batch.x[type_mask]
    batch.tracks_to_predict = batch.tracks_to_predict[type_mask]
    batch.type = batch.type[type_mask]

    states = batch.x.permute(1, 0, 2)
    n_steps = 51
    n_agents = states.shape[1]

    # Create colors and opacities for all agents in scene
    np.random.seed(5)
    agent_colors = [
        (np.random.random(), np.random.random(), np.random.random())
        for _ in range(n_agents)
    ]
    alphas = np.linspace(0.1, 1, n_steps)

    # Setup plotting
    rc("text", usetex=True)
    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

    ticksize = 20
    titlesize = 25
    plt.rcParams["axes.labelsize"] = titlesize
    plt.rcParams["axes.titlesize"] = titlesize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    # Plot each map channel separately using different colors/opacities
    layer_colors = [
        "Greys",
        "Spectral",
        "Greys",
        "Greys",
        "bwr",
        "bwr",
        "Greys",
        "Reds",
    ]
    layer_alphas = [0.2, 0.3, 0.5, 1, 0.5, 1, 1, 1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    titles = [
        "LaneCenters",
        "BikeLaneCenter",
        "BrokenWhite",
        "SolidWhite",
        "BrokenYellow",
        "SolidYellow",
        "Boundaries",
        "Stopsign/Crosswalk/SpeedBump",
    ]

    # Plot all channels together
    for layer_id in range(8):
        layer_mask = roadgraph[layer_id].astype(np.float)
        ax.imshow(
            layer_mask,
            aspect="equal",
            cmap=layer_colors[layer_id],
            norm=colors.Normalize(vmin=-2, vmax=1.0),
            extent=extent,
            origin="lower",
            alpha=layer_mask * layer_alphas[layer_id],
        )

        # Main loop
    for t in range(n_steps - 1):
        # Plot groundtruth
        ax = plot_time_step(
            ax=ax,
            t=t,
            states=states,
            alpha=alphas[t],
            colors=agent_colors,
            n_steps=n_steps,
        )

    ax.axis("equal")
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    # ax[0].set_title("Groundtruth trajectories")
    # ax[1].set_title("Predicted trajectories")

    # plt.tight_layout()
    # plt.show()

    plt.savefig(f"../../thesis/graphics/waymo/map_example_w_cars.pdf")
    plt.savefig(f"visualisations/waymo/map_example_w_cars.pdf")
