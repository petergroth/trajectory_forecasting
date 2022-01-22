import argparse
import os

import hydra
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
# from models import ConstantModel
from matplotlib import rc
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib_scalebar.scalebar import ScaleBar
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from src.data.dataset_waymo import OneStepWaymoDataModule
from src.predictions.make_waymo_predictions import (eigsorted,
                                                    make_predictions,
                                                    plot_edges_single_agent,
                                                    plot_time_step)
from src.training_modules.train_waymo_model import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("ckpt_path")
    parser.add_argument("output_path")
    parser.add_argument("sequence_idx", type=int)
    parser.add_argument("n_steps", type=int, default=51)
    parser.add_argument("--groundtruth", action="store_true")
    parser.add_argument("seed", type=int)
    parser.add_argument("format", type=str)
    parser.add_argument("--covariance", action="store_true")
    # parser.add_argument("--remove_edges", action='store_true')
    # parser.add_argument("agent", type=int)
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Make predictions
    if not args.covariance:
        y_hat, y_target, mask, batch = make_predictions(
            path=args.ckpt_path,
            config=config,
            sequence_idx=args.sequence_idx,
            n_steps=args.n_steps,
            # remove_edges=args.remove_edges
        )
    else:
        y_hat, y_target, mask, batch, Sigma = make_predictions(
            path=args.ckpt_path,
            config=config,
            sequence_idx=args.sequence_idx,
            n_steps=args.n_steps,
            covariance=args.covariance,
        )
        nstd = 1

    # Extract sequence information and roadmap
    n_steps = args.n_steps
    _, n_agents, n_features = y_hat.shape
    roadgraph = batch.u.squeeze().numpy()
    # Remove zero-padding
    if roadgraph.shape[1] != 300:
        roadgraph = roadgraph[:, 40:-40, 40:-40]

    # Plotting options
    rc("text", usetex=True)
    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

    ticksize = 30
    titlesize = 35
    plt.rcParams["axes.labelsize"] = titlesize
    plt.rcParams["axes.titlesize"] = titlesize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

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

    # Create directory for visualisations
    output_dir = f"visualisations/seq_{args.sequence_idx:04}"
    os.makedirs(output_dir, exist_ok=True)
    groundtruth_path = output_dir + f"/groundtruth.{args.format}"
    prediction_path = output_dir + f"/{args.output_path}.{args.format}"

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
    # layer_alphas = np.array([0.2, 0.3, 0.5, 1, 0.5, 1, 1, 1])/10
    # Create colors and opacities for all agents in scene
    np.random.seed(args.seed)
    agent_colors = [
        (np.random.random(), np.random.random(), np.random.random())
        for _ in range(n_agents)
    ]
    alphas = np.linspace(0.1, 1, n_steps)

    # Plot and save groundtruth trajectories in separate figure
    if args.groundtruth:
        for layer_id in range(8):
            layer_mask = roadgraph[layer_id].astype(float)
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
                states=y_target,
                alpha=alphas[t],
                colors=agent_colors,
                n_steps=n_steps,
            )

        ax.axis("equal")
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        # plt.tight_layout()
        fig.savefig(groundtruth_path)
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

    for layer_id in range(8):
        layer_mask = roadgraph[layer_id].astype(float)
        ax.imshow(
            layer_mask,
            aspect="equal",
            cmap=layer_colors[layer_id],
            norm=colors.Normalize(vmin=-2, vmax=1.0),
            extent=extent,
            origin="lower",
            alpha=layer_mask * layer_alphas[layer_id],
            # interpolation="bessel"
        )

    # Main loop
    for t in range(n_steps - 1):
        ax = plot_time_step(
            ax=ax,
            t=t,
            states=y_hat,
            alpha=alphas[t],
            colors=agent_colors,
            n_steps=n_steps,
        )

        if args.covariance:
            for agent in range(n_agents):
                vals, vecs = eigsorted(Sigma[t, agent].numpy())
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * nstd * np.sqrt(vals)
                loc = y_hat[t, agent, :2].numpy()
                ell = Ellipse(
                    xy=loc,
                    width=w,
                    height=h,
                    angle=theta,
                    color=agent_colors[agent],
                    alpha=0.2,
                )
                ax.add_artist(ell)

        # ax = plot_edges_single_agent(ax=ax, t=t, states=y_hat, agent=23, mask=mask,
        #                              dist=config["regressor"]["min_dist"],
        #                              n_neighbours=config["regressor"]["n_neighbours"]
        #                              )
        # ax[1] = plot_edges_all_agents(
        #     ax=ax[1],
        #     t=t,
        #     states=y_hat,
        #     dist=config["regressor"]["min_dist"],
        #     n_neighbours=config["regressor"]["n_neighbours"],
        # )
        # ax[1] = plot_edges_single_agent(ax=ax[1], t=t, states=y_hat, alpha=alphas[t], agent=0, mask=mask)

    ax.axis("equal")
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    # plt.tight_layout()
    fig.savefig(prediction_path)


if __name__ == "__main__":
    main()
