import argparse
import pytorch_lightning as pl
from src.data.dataset_waymo import OneStepWaymoDataModule
from src.models.train_waymo_model import *
import yaml
from pytorch_lightning.utilities.seed import seed_everything
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from matplotlib.patches import Circle


def main():

    ##################
    # Load data      #
    ##################

    dm = OneStepWaymoDataModule()
    dm.setup()
    loader = dm.val_dataloader()
    seq_idx = 0
    for i, batch in enumerate(loader):
        if i == seq_idx:
            break

    mask = batch.x[:, :, -1]
    valid_mask = mask[:, 10] > 0
    batch.x = batch.x[valid_mask]
    batch.type = batch.type[valid_mask]
    batch.batch = batch.batch[valid_mask]
    mask = batch.x[:, :, -1].bool()

    # Discard non-valid nodes as no initial trajectories will be known
    y_target = batch.x
    type = batch.type

    fig, ax = plt.subplots(2, 2, figsize=(20, 20))

    ##################
    # Raw data       #
    ##################

    # Extract boundaries
    small_mask = torch.logical_and(y_target[:, :, 0] != -1, y_target[:, :, 0] != 0)
    x_min, x_max, y_min, y_max = (
        torch.min(y_target[:, :, 0][small_mask]).item(),
        torch.max(y_target[:, :, 0][small_mask]).item(),
        torch.min(y_target[:, :, 1][small_mask]).item(),
        torch.max(y_target[:, :, 1][small_mask]).item(),
    )
    n_agents, n_steps, n_features = y_target.shape

    np.random.seed(42)
    for agent in range(n_agents):

        color = (np.random.random(), np.random.random(), np.random.random())
        for t in range(n_steps):
            if mask[agent, t]:
                x = y_target[agent, t, 0].item()
                y = y_target[agent, t, 1].item()
                width = y_target[agent, t, 7].item()
                length = y_target[agent, t, 8].item()
                angle = y_target[agent, t, 5].item()
                # If car
                if int(type[agent, 1].item()) == 1:
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array(((c, -s), (s, c)))
                    anchor = np.dot(R, np.array([-length / 2, -width / 2])) + np.array(
                        [x, y]
                    )
                    rect = Rectangle(
                        xy=anchor,
                        width=length,
                        height=width,
                        angle=angle * 180 / np.pi,
                        edgecolor="k",
                        facecolor=color,
                        alpha=0.05,
                    )
                    ax[0, 0].add_patch(rect)
                    # Pedestrian
                elif int(type[agent, 2].item()) == 1:
                    ax[0, 0].plot(
                        x, y, marker="o", color=color, alpha=0.05, markerfacecolor=None
                    )
                # Bike
                elif int(type[agent, 2].item()) == 1:
                    ax[0, 0].plot(
                        x, y, marker="+", color=color, alpha=0.05, markerfacecolor=None
                    )
                else:
                    ax[0, 0].plot(x, y, marker="x", color=color, alpha=0.05)

                # Start
                if t == 0:
                    ax[0, 0].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="lightgrey",
                    )
                elif t == 10:
                    ax[0, 0].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="gray",
                    )
                elif t == (n_steps - 1):
                    ax[0, 0].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="k",
                    )

    ###################
    # Centred locs    #
    ###################

    y_target = batch.x.clone()

    norm_idx = [0, 1, 2]
    for t in range(n_steps):
        y_target[:, t, norm_idx] -= batch.loc[batch.batch][:, norm_idx]
        # y_target[:, t, norm_idx] /= batch.std[batch.batch][:, norm_idx]

    np.random.seed(42)
    for agent in range(n_agents):
        color = (np.random.random(), np.random.random(), np.random.random())
        for t in range(n_steps):
            if mask[agent, t]:
                x = y_target[agent, t, 0].item()
                y = y_target[agent, t, 1].item()
                width = y_target[agent, t, 7].item()
                length = y_target[agent, t, 8].item()
                angle = y_target[agent, t, 5].item()
                # If car
                if int(type[agent, 1].item()) == 1:
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array(((c, -s), (s, c)))
                    anchor = np.dot(R, np.array([-length / 2, -width / 2])) + np.array(
                        [x, y]
                    )
                    rect = Rectangle(
                        xy=anchor,
                        width=length,
                        height=width,
                        angle=angle * 180 / np.pi,
                        edgecolor="k",
                        facecolor=color,
                        alpha=0.05,
                    )
                    ax[0, 1].add_patch(rect)
                    # Pedestrian
                elif int(type[agent, 2].item()) == 1:
                    ax[0, 1].plot(
                        x, y, marker="o", color=color, alpha=0.05, markerfacecolor=None
                    )
                # Bike
                elif int(type[agent, 2].item()) == 1:
                    ax[0, 1].plot(
                        x, y, marker="+", color=color, alpha=0.05, markerfacecolor=None
                    )
                else:
                    ax[0, 1].plot(x, y, marker="x", color=color, alpha=0.05)

                # Start
                if t == 0:
                    ax[0, 1].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="lightgrey",
                    )
                elif t == 10:
                    ax[0, 1].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="gray",
                    )
                elif t == (n_steps - 1):
                    ax[0, 1].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="k",
                    )

    ###################
    # Normed locs     #
    ###################

    y_target = batch.x.clone()

    norm_idx = [0, 1, 2]
    for t in range(n_steps):
        y_target[:, t, norm_idx] -= batch.loc[batch.batch][:, norm_idx]
        y_target[:, t, norm_idx] /= batch.std[batch.batch][:, norm_idx]

    np.random.seed(42)
    for agent in range(n_agents):
        color = (np.random.random(), np.random.random(), np.random.random())
        for t in range(n_steps):
            if mask[agent, t]:
                x = y_target[agent, t, 0].item()
                y = y_target[agent, t, 1].item()
                width = y_target[agent, t, 7].item()
                length = y_target[agent, t, 8].item()
                angle = y_target[agent, t, 5].item()
                # If car
                if int(type[agent, 1].item()) == 1:
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array(((c, -s), (s, c)))
                    anchor = np.dot(R, np.array([-length / 2, -width / 2])) + np.array(
                        [x, y]
                    )
                    rect = Rectangle(
                        xy=anchor,
                        width=length,
                        height=width,
                        angle=angle * 180 / np.pi,
                        edgecolor="k",
                        facecolor=color,
                        alpha=0.05,
                    )
                    ax[1, 0].add_patch(rect)
                    # Pedestrian
                elif int(type[agent, 2].item()) == 1:
                    ax[1, 0].plot(
                        x, y, marker="o", color=color, alpha=0.05, markerfacecolor=None
                    )
                # Bike
                elif int(type[agent, 2].item()) == 1:
                    ax[1, 0].plot(
                        x, y, marker="+", color=color, alpha=0.05, markerfacecolor=None
                    )
                else:
                    ax[1, 0].plot(x, y, marker="x", color=color, alpha=0.05)

                # Start
                if t == 0:
                    ax[1, 0].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="lightgrey",
                    )
                elif t == 10:
                    ax[1, 0].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="gray",
                    )
                elif t == (n_steps - 1):
                    ax[1, 0].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="k",
                    )

    ###################
    #      #
    ###################

    y_target = batch.x.clone()
    loc_idx = [0, 1, 2]
    std_idx = [0, 1, 2, 3, 4, 7, 8, 9]

    global_scale = 8.025897979736328

    for t in range(n_steps):
        y_target[:, t, loc_idx] -= batch.loc[batch.batch][:, loc_idx]
        y_target[:, t, std_idx] /= global_scale

    np.random.seed(42)
    for agent in range(n_agents):
        color = (np.random.random(), np.random.random(), np.random.random())
        for t in range(n_steps):
            if mask[agent, t]:
                x = y_target[agent, t, 0].item()
                y = y_target[agent, t, 1].item()
                width = y_target[agent, t, 7].item()
                length = y_target[agent, t, 8].item()
                angle = y_target[agent, t, 5].item()
                # If car
                if int(type[agent, 1].item()) == 1:
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array(((c, -s), (s, c)))
                    anchor = np.dot(R, np.array([-length / 2, -width / 2])) + np.array(
                        [x, y]
                    )
                    rect = Rectangle(
                        xy=anchor,
                        width=length,
                        height=width,
                        angle=angle * 180 / np.pi,
                        edgecolor="k",
                        facecolor=color,
                        alpha=0.05,
                    )
                    ax[1, 1].add_patch(rect)
                    # Pedestrian
                elif int(type[agent, 2].item()) == 1:
                    ax[1, 1].plot(
                        x, y, marker="o", color=color, alpha=0.05, markerfacecolor=None
                    )
                # Bike
                elif int(type[agent, 2].item()) == 1:
                    ax[1, 1].plot(
                        x, y, marker="+", color=color, alpha=0.05, markerfacecolor=None
                    )
                else:
                    ax[1, 0].plot(x, y, marker="x", color=color, alpha=0.05)

                # Start
                if t == 0:
                    ax[1, 1].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="lightgrey",
                    )
                elif t == 10:
                    ax[1, 1].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="gray",
                    )
                elif t == (n_steps - 1):
                    ax[1, 1].quiver(
                        y_target[agent, t, 0].detach().numpy(),
                        y_target[agent, t, 1].detach().numpy(),
                        y_target[agent, t, 3].detach().numpy(),
                        y_target[agent, t, 4].detach().numpy(),
                        width=0.003,
                        headwidth=5,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="k",
                    )

    ax[0, 0].axis("equal")
    ax[0, 1].axis("equal")
    ax[1, 1].axis("equal")
    ax[0, 0].set_title("Unnormalised trajectories")
    ax[0, 1].set_title("Centred locations")
    ax[1, 0].set_title("Normalised locations")
    ax[1, 1].set_title(
        f"Centred locations. All attributes are scaled by global scaling constant ({global_scale:.3})"
    )

    plt.show()
    fig.savefig(
        "visualisations/"
        + "comparison_of_normalisations_sequence_"
        + f"{seq_idx:03}"
        + ".png"
    )


if __name__ == "__main__":
    main()
