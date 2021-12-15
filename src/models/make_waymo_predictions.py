import argparse
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
# from models import ConstantModel
from matplotlib.patches import Circle, Rectangle
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from src.data.dataset_waymo import OneStepWaymoDataModule
from src.training_modules.train_waymo_rnn_global import *


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
    y_hat, y_target, mask = regressor.predict_step(batch, prediction_horizon=n_steps)
    # torch.save((y_hat, y_target, mask), dirpath + f"/sequence_{i:04}.pt")
    return y_hat.detach(), y_target, mask, batch


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


#
# #@hydra.main(config_path="../../configs/waymo/", config_name="config")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("config")
#     parser.add_argument("ckpt_path")
#     parser.add_argument("output_path")
#     parser.add_argument("sequence_idx", type=int)
#     args = parser.parse_args()
#
#     # Load yaml
#     with open(args.config) as f:
#         config = yaml.safe_load(f)
#
#     # Computes predictions for specified sequence
#     make_predictions(
#         path=args.ckpt_path,
#         config=config,
#         sequence_idx=args.sequence_idx,
#     )
#
#     # Create directory for current model
#     # vis_dir = "visualisations/predictions/waymo/" + config["logger"]["version"] + "/"
#     os.makedirs(args.output_path, exist_ok=True)
#     # Location of prediction files
#     dir = "src/predictions/raw_preds/waymo/" + config["logger"]["version"] + "/"
#     # Load first file in directory
#     path = "sequence_" + f"{args.sequence_idx:04}.pt"
#     y_hat, y_target, mask = torch.load(dir + path)
#     y_hat = y_hat.detach()
#     n_steps, n_agents, n_features = y_hat.shape
#
#     small_mask = torch.logical_and(y_target[:, :, 0] != -1, y_target[:, :, 0] != 0)
#     # Extract boundaries
#     x_min, x_max, y_min, y_max = (
#         torch.min(y_target[:, :, 0][small_mask]).item(),
#         torch.max(y_target[:, :, 0][small_mask]).item(),
#         torch.min(y_target[:, :, 1][small_mask]).item(),
#         torch.max(y_target[:, :, 1][small_mask]).item(),
#     )
#
#     figglob, axglob = plt.subplots(1, 2, figsize=(20, 10))
#
#     for agent in range(n_agents):
#         color = (np.random.random(), np.random.random(), np.random.random())
#         for t in range(n_steps):
#             # if t == 0 or t == 10 or t == 89:
#             if True:
#                 x = y_target[t, agent, 0].item()
#                 y = y_target[t, agent, 1].item()
#                 width = y_target[t, agent, 7].item()
#                 length = y_target[t, agent, 8].item()
#                 angle = y_target[t, agent, 5].item()
#                 # If car
#                 # if int(y_target[t, agent, 11].item()) == 1:
#                 if True:
#                     c, s = np.cos(angle), np.sin(angle)
#                     R = np.array(((c, -s), (s, c)))
#                     anchor = np.dot(R, np.array([-length / 2, -width / 2])) + np.array(
#                         [x, y]
#                     )
#                     rect = Rectangle(
#                         xy=anchor,
#                         width=length,
#                         height=width,
#                         angle=angle * 180 / np.pi,
#                         edgecolor="k",
#                         facecolor=color,
#                         alpha=0.05,
#                     )
#                     axglob[0].add_patch(rect)
#                 # Pedestrian
#                 # elif int(y_target[t, agent, 12].item()) == 1:
#                 #     axglob[0].plot(
#                 #         x, y, marker="o", color=color, alpha=0.05, markerfacecolor=None
#                 #     )
#                 # # Bike
#                 # elif int(y_target[t, agent, 13].item()) == 1:
#                 #     axglob[0].plot(
#                 #         x, y, marker="+", color=color, alpha=0.05, markerfacecolor=None
#                 #     )
#                 # else:
#                 #     axglob[0].plot(x, y, marker="x", color=color, alpha=0.05)
#
#             # Start
#             if t == 0:
#                 axglob[0].quiver(
#                     y_target[t, agent, 0].detach().numpy(),
#                     y_target[t, agent, 1].detach().numpy(),
#                     y_target[t, agent, 3].detach().numpy(),
#                     y_target[t, agent, 4].detach().numpy(),
#                     width=0.003,
#                     headwidth=5,
#                     angles="xy",
#                     scale_units="xy",
#                     scale=1.0,
#                     color="lightgrey",
#                 )
#             elif t == 10:
#                 axglob[0].quiver(
#                     y_target[t, agent, 0].detach().numpy(),
#                     y_target[t, agent, 1].detach().numpy(),
#                     y_target[t, agent, 3].detach().numpy(),
#                     y_target[t, agent, 4].detach().numpy(),
#                     width=0.003,
#                     headwidth=5,
#                     angles="xy",
#                     scale_units="xy",
#                     scale=1.0,
#                     color="gray",
#                 )
#             elif t == (n_steps - 1):
#                 axglob[0].quiver(
#                     y_target[t, agent, 0].detach().numpy(),
#                     y_target[t, agent, 1].detach().numpy(),
#                     y_target[t, agent, 3].detach().numpy(),
#                     y_target[t, agent, 4].detach().numpy(),
#                     width=0.003,
#                     headwidth=5,
#                     angles="xy",
#                     scale_units="xy",
#                     scale=1.0,
#                     color="k",
#                 )
#
#         for t in range(n_steps):
#             # if t == 0 or t == 10 or t == 89:
#             if True:
#                 x = y_hat[t, agent, 0].item()
#                 y = y_hat[t, agent, 1].item()
#                 width = y_hat[t, agent, 7].item()
#                 length = y_hat[t, agent, 8].item()
#                 angle = y_hat[t, agent, 5].item()
#                 # If car
#                 if True:
#                 # if int(y_hat[t, agent, 11].item()) == 1:
#                     c, s = np.cos(angle), np.sin(angle)
#                     R = np.array(((c, -s), (s, c)))
#                     anchor = np.dot(R, np.array([-length / 2, -width / 2])) + np.array(
#                         [x, y]
#                     )
#                     rect = Rectangle(
#                         xy=anchor,
#                         width=length,
#                         height=width,
#                         angle=angle * 180 / np.pi,
#                         edgecolor="k",
#                         facecolor=color,
#                         alpha=0.05,
#                     )
#                     axglob[1].add_patch(rect)
#                 # Pedestrian
#                 # elif int(y_hat[t, agent, 12].item()) == 1:
#                 #     axglob[1].plot(
#                 #         x, y, marker="o", color=color, alpha=0.05, markerfacecolor=None
#                 #     )
#                 # # Bike
#                 # elif int(y_hat[t, agent, 13].item()) == 1:
#                 #     axglob[1].plot(
#                 #         x, y, marker="+", color=color, alpha=0.05, markerfacecolor=None
#                 #     )
#                 # else:
#                 #     axglob[1].plot(x, y, marker="x", color=color, alpha=0.05)
#
#             if t == (n_steps - 1) or t == 0 or t == 10:
#                 axglob[1].quiver(
#                     y_hat[t, agent, 0].detach().numpy(),
#                     y_hat[t, agent, 1].detach().numpy(),
#                     y_hat[t, agent, 3].detach().numpy(),
#                     y_hat[t, agent, 4].detach().numpy(),
#                     width=0.003,
#                     headwidth=5,
#                     angles="xy",
#                     scale_units="xy",
#                     scale=1.0,
#                     alpha=1,
#                 )
#
#     axglob[0].axis("equal")
#     axglob[0].set_xlim((x_min, x_max))
#     axglob[0].set_ylim((y_min, y_max))
#     axglob[1].axis("equal")
#     axglob[1].set_xlim((x_min, x_max))
#     axglob[1].set_ylim((y_min, y_max))
#     axglob[0].set_title("Groundtruth trajectories")
#     axglob[1].set_title("Predicted trajectories")
#
#     plt.show()
#     figglob.savefig(f"{args.output_path}/sequence_{args.sequence_idx:04}.png")


# @hydra.main(config_path="../../configs/waymo/", config_name="config")
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

    y_hat, y_target, mask, batch = make_predictions(
        path=args.ckpt_path,
        config=config,
        sequence_idx=args.sequence_idx,
        n_steps=args.n_steps,
    )
    n_steps = args.n_steps
    _, n_agents, n_features = y_hat.shape
    roadgraph = batch.u.squeeze().numpy() / 2
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


    alphas = np.linspace(0.1, 1, n_steps)
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
