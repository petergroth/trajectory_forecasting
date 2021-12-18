import argparse
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
# from models import ConstantModel
from matplotlib.patches import Circle, Rectangle, Ellipse
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from src.data.dataset_waymo import OneStepWaymoDataModule
# from src.training_modules.train_waymo_rnn_global import *
from src.training_modules.train_waymo_gauss import *

def make_predictions(path, config, n_steps=51, sequence_idx=0):
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
        regressor = eval(config["misc"]["regressor_type"])(**config["regressor"])
    # Setup
    regressor.eval()
    datamodule.setup()
    dataset = datamodule.val_dataset
    batch = dataset.__getitem__(sequence_idx)

    # agent = [0, 1, 2, 3]
    # batch.x = batch.x[agent]
    batch.batch = torch.zeros(batch.x.size(0)).type(torch.int64)
    batch.num_graphs = 1
    # batch.batch = batch.batch[agent]
    # batch.tracks_to_predict = batch.tracks_to_predict[agent]
    # batch.type = batch.type[agent]
    y_hat, y_target, mask, Sigma = regressor.predict_step(batch, prediction_horizon=n_steps)
    # torch.save((y_hat, y_target, mask), dirpath + f"/sequence_{i:04}.pt")
    return y_hat.detach(), y_target, mask, batch, Sigma


def plot_time_step(ax, t, states, alpha, colors, n_steps):
    # Scatter plot of all agent positions
    if t == n_steps-1 or t == 10:
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
    if t == 10 or t == n_steps - 2:
        ax.quiver(
            states[t, :, 0].detach().numpy(),
            states[t, :, 1].detach().numpy(),
            states[t, :, 2].detach().numpy(),
            states[t, :, 3].detach().numpy(),
            width=0.003,
            headwidth=5,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="lightgrey" if t == 10 else "k",
        )

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
    args = parser.parse_args()

    # Load yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    y_hat, y_target, mask, batch, Sigma = make_predictions(
        path=args.ckpt_path,
        config=config,
        sequence_idx=args.sequence_idx,
        n_steps=args.n_steps,
    )
    n_steps = args.n_steps
    _, n_agents, n_features = y_hat.shape
    roadgraph = batch.u.squeeze().numpy()
    roadgraph = np.sum(roadgraph, axis=0)
    roadgraph = roadgraph[40:-40, 40:-40]
    # print(roadgraph.shape)
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
    os.makedirs(args.output_path, exist_ok=True)

    # small_mask = torch.logical_and(y_target[:, :, 0] != -1, y_target[:, :, 0] != 0)
    # Extract boundaries
    # x_min, x_max, y_min, y_max = (
    #     torch.min(y_target[:, :, 0][small_mask]).item(),
    #     torch.max(y_target[:, :, 0][small_mask]).item(),
    #     torch.min(y_target[:, :, 1][small_mask]).item(),
    #     torch.max(y_target[:, :, 1][small_mask]).item(),
    # )

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    #
    # ax[0].imshow(roadgraph, aspect="equal", cmap="Greys", extent=extent, origin='lower', vmin=0, vmax=1,
    #                  alpha=0.5)
    # point = np.array([-28540, 41810])
    # ax[0].plot(point[0], point[1], color='k', marker='x')
    # ax[0].plot(x_span, y_span)
    #
    # plt.show()

    ax[0].imshow(
        roadgraph,
        aspect="equal",
        cmap="Greys",
        extent=extent,
        origin="lower",
        vmin=0,
        vmax=1,
        alpha=0.5,
    )
    ax[1].imshow(
        roadgraph,
        aspect="equal",
        cmap="Greys",
        extent=extent,
        origin="lower",
        vmin=0,
        vmax=1,
        alpha=0.5,
    )


    colors = [
        (np.random.random(), np.random.random(), np.random.random())
        for _ in range(n_agents)
    ]

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    alphas = np.linspace(0.1, 1, n_steps)

    nstd = 1

    for t in range(n_steps - 1):
        #Plot groundtruth
        ax[0] = plot_time_step(
            ax=ax[0],
            t=t,
            states=y_target,
            alpha=alphas[t],
            colors=colors,
            n_steps=n_steps,
        )
        ax[1] = plot_time_step(
            ax=ax[1], t=t, states=y_hat, alpha=alphas[t], colors=colors, n_steps=n_steps
        )

        # if t % 2 == 0:
        for agent in range(n_agents):
            vals, vecs = eigsorted(Sigma[t, agent].detach().numpy())
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)
            loc = y_hat[t, agent, :2].numpy()
            ell = Ellipse(xy=loc, width=w, height=h, angle=theta, color=colors[agent], alpha=0.1)
            ax[1].add_artist(ell)


        # ax[1] = plot_edges_single_agent(ax=ax[1], t=t, states=y_hat, alpha=alphas[t], agent=3, mask=mask)
        # ax[1] = plot_edges_all_agents(
        #     ax=ax[1],
        #     t=t,
        #     states=y_hat,
        #     dist=config["regressor"]["min_dist"],
        #     n_neighbours=config["regressor"]["n_neighbours"],
        # )
        # ax[1] = plot_edges_single_agent(ax=ax[1], t=t, states=y_hat, alpha=alphas[t], agent=0, mask=mask)

    ax[0].axis("equal")
    ax[0].set_xlim((x_min, x_max))
    ax[0].set_ylim((y_min, y_max))
    ax[1].axis("equal")
    ax[1].set_xlim((x_min, x_max))
    ax[1].set_ylim((y_min, y_max))
    ax[0].set_title("Groundtruth trajectories")
    ax[1].set_title("Predicted trajectories")

    # plt.show()
    fig.savefig(f"{args.output_path}/sequence_{args.sequence_idx:04}_{n_steps}.png")


if __name__ == "__main__":
    main()

