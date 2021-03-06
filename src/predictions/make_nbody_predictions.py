import argparse
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from matplotlib import rc
# from models import ConstantModel
from matplotlib.patches import Circle, Rectangle
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from src.data.dataset_nbody import *
from src.predictions.make_waymo_predictions import (plot_edges_all_agents,
                                                    plot_edges_single_agent)
from src.training_modules.train_nbody_model import (
    ConstantPhysicalBaselineModule, OneStepModule, SequentialModule)


def make_predictions(path: str, config: dict, n_steps: int = 51, sequence_idx: int = 0):
    # Set seed
    seed_everything(config["misc"]["seed"], workers=True)
    # Load datamodule
    config["datamodule"]["val_batch_size"] = 1
    config["datamodule"]["batch_size"] = 1
    config["datamodule"]["shuffle"] = False
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])

    # Load correct model
    if config["misc"]["model_type"] != "ConstantModel":
        regressor = eval(config["misc"]["regressor_type"]).load_from_checkpoint(path)
    else:
        regressor = ConstantPhysicalBaselineModule(**config["regressor"])
    # Setup
    regressor.eval()
    datamodule.setup()
    dataset = datamodule.test_dataset
    # Extract batch and add missing attributes
    batch = dataset.__getitem__(sequence_idx)
    batch.batch = torch.zeros(batch.x.size(0)).type(torch.int64)
    batch.num_graphs = 1

    # Make and return predictions
    y_hat, y_target = regressor.predict_step(batch, prediction_horizon=n_steps)
    return y_hat.detach(), y_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("ckpt_path")
    parser.add_argument("output_path")
    parser.add_argument("sequence_idx", type=int)
    parser.add_argument("--edges", type=bool, default=False)
    args = parser.parse_args()

    rc("text", usetex=True)
    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

    ticksize = 30
    titlesize = 35
    plt.rcParams["axes.labelsize"] = titlesize
    plt.rcParams["axes.titlesize"] = titlesize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    # Load yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Computes predictions for specified sequence
    y_hat, y_target = make_predictions(
        path=args.ckpt_path,
        config=config,
        sequence_idx=args.sequence_idx,
    )

    n_agents, n_steps, n_features = y_hat.shape

    # Convert to numpy arrays
    y_hat = y_hat.detach().numpy()
    y_target = y_target.detach().numpy()

    # Extract boundaries
    x_min, x_max, y_min, y_max = (
        np.min(y_target[:, :, 0]) - 10,
        np.max(y_target[:, :, 0]) + 10,
        np.min(y_target[:, :, 1]) - 10,
        np.max(y_target[:, :, 1]) + 10,
    )

    x_min, x_max, y_min, y_max = (-100, 70, -60, 110)

    # Setup plotting
    alphas = np.linspace(0.1, 1, n_steps)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

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

    # Extract target positions and sizes
    positions = y_target[:, :, :2]
    velocities = y_target[:, :, 2:4]
    sizes = y_target[:, :, 4]
    for t in range(n_steps):
        for i in range(n_agents):
            # Visualise each agent as a circle
            ax[0].add_patch(
                Circle(
                    positions[i, t],
                    sizes[i, t],
                    facecolor=colors[i],
                    alpha=alphas[t],
                    edgecolor="k" if t in (n_steps - 1, 10) else None,
                )
            )

    # Show final velocities
    ax[0].quiver(
        positions[:, -1, 0],
        positions[:, -1, 1],
        velocities[:, -1, 0],
        velocities[:, -1, 1],
        width=0.003,
        headwidth=5,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="k",
        zorder=2,
    )

    # Extract target positions and sizes
    positions = y_hat[:, :, :2]
    velocities = y_hat[:, :, 2:4]
    sizes = y_hat[:, :, 4]
    for t in range(n_steps):
        for i in range(n_agents):
            # Visualise each agent as a circle
            ax[1].add_patch(
                Circle(
                    positions[i, t],
                    sizes[i, t],
                    facecolor=colors[i],
                    alpha=alphas[t],
                    edgecolor="k" if t in (n_steps - 1, 10) else None,
                )
            )
        if args.edges:
            # ax[1] = plot_edges_all_agents(ax[1], t, torch.Tensor(positions).permute(1, 0, 2),
            #                               dist=config["regressor"]["min_dist"],
            #                               n_neighbours=config["regressor"]["n_neighbours"])
            ax[1] = plot_edges_single_agent(
                ax[1],
                t,
                torch.Tensor(positions).permute(1, 0, 2),
                2,
                torch.ones_like(torch.Tensor(positions))[:, :, 0],
                dist=config["regressor"]["min_dist"],
                n_neighbours=config["regressor"]["n_neighbours"],
            )

    # Show final velocities
    if not args.edges:
        ax[1].quiver(
            positions[:, -1, 0],
            positions[:, -1, 1],
            velocities[:, -1, 0],
            velocities[:, -1, 1],
            width=0.003,
            headwidth=5,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="k",
            zorder=10,
        )

    # ax[0].axis("equal")
    # ax[0].set_xlim((x_min, x_max))
    # ax[0].set_ylim((y_min, y_max))
    # ax[1].axis("equal")
    # ax[1].set_xlim((x_min, x_max))
    # ax[1].set_ylim((y_min, y_max))
    plt.setp(ax, xlim=(x_min, x_max), ylim=(y_min, y_max))
    ax[0].set_title("Groundtruth trajectories", pad=8)
    ax[1].set_title("Predicted trajectories", pad=8)

    plt.tight_layout()
    # plt.show()
    plt.savefig(
        f"../../thesis/graphics/synthetic/{args.output_path}_{args.sequence_idx}.pdf"
    )
    plt.savefig(f"visualisations/nbody/{args.output_path}_{args.sequence_idx}.pdf")


if __name__ == "__main__":
    main()
