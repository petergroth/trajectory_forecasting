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
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from src.data.dataset_waymo import OneStepWaymoDataModule
from src.training_modules.train_waymo_model import *


def make_sampled_predictions(path, config, n_steps=51, sequence_idx=0, seed=0):
    # Set seed
    seed_everything(seed, workers=True)
    # Load datamodule
    config["datamodule"]["val_batch_size"] = 1
    config["datamodule"]["batch_size"] = 1
    config["datamodule"]["shuffle"] = False
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])

    # Load correct model
    if config["misc"]["model_type"] != "ConstantModel":
        regressor = eval(config["misc"]["regressor_type"]).load_from_checkpoint(path)
    else:
        regressor = eval(config["misc"]["regressor_type"])(**config["regressor"])
        # Setup
    regressor.eval()
    datamodule.setup("test")
    dataset = datamodule.test_dataset
    # Extract batch and add missing attributes
    batch = dataset.__getitem__(sequence_idx)
    batch.batch = torch.zeros(batch.x.size(0)).type(torch.int64)
    batch.num_graphs = 1

    # Make and return predictions
    with torch.no_grad():
        y_hat, y_target, mask = regressor.sample_trajectories(
            batch, prediction_horizon=n_steps, n_trajectories=10
        )
    return y_hat.detach(), y_target, mask, batch


def plot_time_step(ax, t, states, alpha, colors, n_steps):
    # Scatter plot of all agent positions
    if t == n_steps - 1 or t == 10:
        edgecolors = "k"
    else:
        edgecolors = None

    ax.scatter(
        x=states[t, :, 0].numpy(),
        y=states[t, :, 1].numpy(),
        s=50,
        color=colors,
        alpha=alpha,
        edgecolors=edgecolors,
    )
    # Draw velocity arrows at first and final future predictions
    # if t == 10 or t == n_steps - 2:
    #     ax.quiver(
    #         states[t, :, 0].detach().numpy(),
    #         states[t, :, 1].detach().numpy(),
    #         states[t, :, 2].detach().numpy(),
    #         states[t, :, 3].detach().numpy(),
    #         width=0.003,
    #         headwidth=5,
    #         angles="xy",
    #         scale_units="xy",
    #         scale=1.0,
    #         color="lightgrey" if t == 10 else "k",
    #     )

    return ax


def plot_edges_single_agent(ax, t, states, agent, mask, dist, n_neighbours):
    if mask[agent, t]:

        edge_index = torch_geometric.nn.radius_graph(
            x=states[t, :, :2],
            r=dist,
            loop=False,
            max_num_neighbors=n_neighbours,
            flow="source_to_target",
        )

        edge_index = edge_index[:, edge_index[0] == agent]

        sources = states[t, edge_index[0], :2].numpy()
        targets = states[t, edge_index[1], :2].numpy()

        xs = sources[:, 0]
        ys = sources[:, 1]
        xt = targets[:, 0]
        yt = targets[:, 1]
        xx = np.hstack((xs, xt))
        yy = np.hstack((ys, yt))

        for i in range(edge_index.size(1)):
            ax.plot(
                xx[[i, i + edge_index.size(1)]],
                yy[[i, i + edge_index.size(1)]],
                color="k",
                alpha=0.2,
            )

    return ax


def plot_edges_all_agents(ax, t, states, dist, n_neighbours):
    # Create graph
    edge_index = torch_geometric.nn.radius_graph(
        x=states[t, :, :2],
        r=dist,
        loop=False,
        max_num_neighbors=n_neighbours,
        flow="source_to_target",
    )

    sources = states[t, edge_index[0], :2].numpy()
    targets = states[t, edge_index[1], :2].numpy()

    xs = sources[:, 0]
    ys = sources[:, 1]
    xt = targets[:, 0]
    yt = targets[:, 1]
    xx = np.hstack((xs, xt))
    yy = np.hstack((ys, yt))

    for i in range(edge_index.size(1)):
        ax.plot(
            xx[[i, i + edge_index.size(1)]],
            yy[[i, i + edge_index.size(1)]],
            color="k",
            alpha=0.2,
        )

    return ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("ckpt_path")
    parser.add_argument("output_path")
    parser.add_argument("sequence_idx", type=int)
    parser.add_argument("n_steps", type=int, default=51)
    parser.add_argument("format", type=str)
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Make predictions
    y_hat, y_target, mask, batch = make_sampled_predictions(
        path=args.ckpt_path,
        config=config,
        sequence_idx=args.sequence_idx,
        n_steps=args.n_steps,
    )
    # Create directory for visualisations
    output_dir = f"visualisations/seq_{args.sequence_idx:04}"
    os.makedirs(output_dir, exist_ok=True)
    prediction_path = output_dir + f"/{args.output_path}_sampled_alt.{args.format}"
    # Extract sequence information and roadmap
    n_steps = args.n_steps
    _, n_trajectories, n_agents, n_features = y_hat.shape
    roadgraph = batch.u.squeeze().numpy()
    # Remove zero-padding
    roadgraph = roadgraph[:, 40:-40, 40:-40]

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

    # Plotting options
    rc("text", usetex=True)
    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

    ticksize = 30
    titlesize = 35
    plt.rcParams["axes.labelsize"] = titlesize
    plt.rcParams["axes.titlesize"] = titlesize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

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
    np.random.seed(5)
    # Create colors and opacities for all agents in scene
    agent_colors = [
        (np.random.random(), np.random.random(), np.random.random())
        for _ in range(n_agents)
    ]
    alphas = np.linspace(0.1, 1, n_steps)

    # Main loop
    for t in range(n_steps - 1):
        for n in range(n_trajectories):
            ax = plot_time_step(
                ax=ax,
                t=t,
                states=y_hat[:, n],
                alpha=alphas[t],
                colors=agent_colors,
                n_steps=n_steps,
            )

    ax.axis("equal")
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    # ax[0].set_title("Groundtruth trajectories")
    # ax[1].set_title("Predicted trajectories")

    # plt.show()
    fig.savefig(prediction_path)


if __name__ == "__main__":
    main()
