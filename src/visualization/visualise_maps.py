import argparse
import os

import hydra
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from matplotlib.patches import Circle, Ellipse, Rectangle
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from src.data.dataset_waymo import SequentialWaymoDataModule

# from src.training_modules.train_waymo_model import *


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

    plt.tight_layout()
    plt.savefig(f"../../thesis/graphics/waymo/map_example.pdf")
    plt.savefig(f"visualisations/waymo/map_example.pdf")

    # Setup grid plotting
    rc("text", usetex=True)
    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

    ticksize = 30
    titlesize = 35
    plt.rcParams["axes.labelsize"] = titlesize
    plt.rcParams["axes.titlesize"] = titlesize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    # Plot all channels together
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    for layer_id in range(0, 4):
        layer_mask = roadgraph[layer_id].astype(np.float)
        ax.flatten()[layer_id].imshow(
            layer_mask,
            aspect="equal",
            cmap=layer_colors[layer_id],
            norm=colors.Normalize(vmin=-2, vmax=1.0),
            extent=extent,
            origin="lower",
            alpha=layer_mask * layer_alphas[layer_id],
        )
        ax.flatten()[layer_id].set_title(titles[layer_id])

    plt.tight_layout()
    plt.savefig(f"../../thesis/graphics/waymo/map_example_grid_1of2.pdf")
    plt.savefig(f"visualisations/waymo/map_example_grid_1of2.pdf")

    # Plot all channels together
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    for layer_id in range(4, 8):
        layer_mask = roadgraph[layer_id].astype(np.float)
        ax.flatten()[layer_id - 4].imshow(
            layer_mask,
            aspect="equal",
            cmap=layer_colors[layer_id],
            norm=colors.Normalize(vmin=-2, vmax=1.0),
            extent=extent,
            origin="lower",
            alpha=layer_mask * layer_alphas[layer_id],
        )
        ax.flatten()[layer_id - 4].set_title(titles[layer_id])

    plt.tight_layout()
    plt.savefig(f"../../thesis/graphics/waymo/map_example_grid_2of2.pdf")
    plt.savefig(f"visualisations/waymo/map_example_grid_2of2.pdf")
