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

# from models import ConstantModel
from matplotlib.patches import Circle


def make_predictions(path, config_file, sequence_idx=0):
    # Parse config file
    with open("configs/waymo/" + config_file, "r") as file:
        config = yaml.safe_load(file)
    # Set seed
    seed_everything(config["misc"]["seed"], workers=True)
    # Load datamodule
    config["datamodule"]["val_batch_size"] = 1
    config["datamodule"]["batch_size"] = 1
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])
    # Load correct model
    if config["misc"]["model_type"] != "ConstantModel":
        regressor = eval(config["misc"]["regressor_type"]).load_from_checkpoint(path)
    else:
        regressor = eval(config["misc"]["regressor_type"])(None, **config["regressor"])
    # Setup
    regressor.eval()
    datamodule.setup()
    loader = datamodule.train_dataloader()
    # Define output path
    dirpath = "src/predictions/raw_preds/waymo/" + config["logger"]["version"]
    os.makedirs(dirpath, exist_ok=True)
    for i, batch in enumerate(loader):
        if i == sequence_idx:
            y_hat, y_target, mask = regressor.predict_step(batch)
            torch.save((y_hat, y_target, mask), dirpath + f"/sequence_{i:03}.pt")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("config")
    parser.add_argument("--sequence_idx", default=0, type=int)
    args = parser.parse_args()

    # Computes predictions for specified sequence
    make_predictions(
        path=args.path, config_file=args.config, sequence_idx=args.sequence_idx
    )

    # Loads config file
    with open("configs/waymo/" + args.config, "r") as file:
        config = yaml.safe_load(file)

    # Create directory for current model
    vis_dir = "visualisations/predictions/waymo/" + config["logger"]["version"] + "/"
    os.makedirs(vis_dir, exist_ok=True)
    # Location of prediction files
    dir = "src/predictions/raw_preds/waymo/" + config["logger"]["version"] + "/"
    # Load first file in directory
    path = "sequence_" + f"{args.sequence_idx:03}.pt"
    y_hat, y_target, mask = torch.load(dir + path)

    n_steps, n_agents, n_features = y_hat.shape
    mask = mask.permute(1, 0)

    small_mask = y_target[:, :, 0] > 0
    # Extract boundaries
    x_min, x_max, y_min, y_max = (
        torch.min(y_target[:, :, 0][small_mask]).item(),
        torch.max(y_target[:, :, 0][small_mask]).item(),
        torch.min(y_target[:, :, 1][small_mask]).item(),
        torch.max(y_target[:, :, 1][small_mask]).item(),
    )

    figglob, axglob = plt.subplots(1, 2, figsize=(20, 10))

    for agent in range(n_agents):
        color = (np.random.random(), np.random.random(), np.random.random())
        for t in range(n_steps):
            # if t == 0 or t == 10 or t == 89:
            if True:
                x = y_target[t, agent, 0].item()
                y = y_target[t, agent, 1].item()
                width = y_target[t, agent, 7].item()
                length = y_target[t, agent, 8].item()
                angle = y_target[t, agent, 5].item()
                # If car
                if int(y_target[t, agent, 11].item()) == 1:
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array(((c, -s), (s, c)))
                    anchor = np.dot(R, np.array([-length / 2, -width / 2])) + np.array([x, y])
                    rect = Rectangle(
                        xy=anchor,
                        width=length,
                        height=width,
                        angle=angle * 180 / np.pi,
                        edgecolor='k',
                        facecolor=color,
                        alpha=0.05,
                    )
                    axglob[0].add_patch(rect)
                # Pedestrian
                elif int(y_target[t, agent, 12].item()) == 1:
                    axglob[0].plot(x, y, marker='o', color=color, alpha=0.05, markerfacecolor=None)
                # Bike
                elif int(y_target[t, agent, 13].item()) == 1:
                    axglob[0].plot(x, y, marker='+', color=color, alpha=0.05, markerfacecolor=None)
                else:
                    axglob[0].plot(x, y, marker='x', color=color,  alpha=0.05)

            if t == (n_steps - 1) or t == 0 or t == 10:
                axglob[0].quiver(
                    y_target[t, agent, 0].detach().numpy(),
                    y_target[t, agent, 1].detach().numpy(),
                    y_target[t, agent, 3].detach().numpy(),
                    y_target[t, agent, 4].detach().numpy(),
                    width=0.003,
                    headwidth=5,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    alpha=1,
                )

        for t in range(n_steps):
            # if t == 0 or t == 10 or t == 89:
            if True:
                x = y_hat[t, agent, 0].item()
                y = y_hat[t, agent, 1].item()
                width = y_hat[t, agent, 7].item()
                length = y_hat[t, agent, 8].item()
                angle = y_hat[t, agent, 5].item()
                # If car
                if int(y_hat[t, agent, 11].item()) == 1:
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array(((c, -s), (s, c)))
                    anchor = np.dot(R, np.array([-length / 2, -width / 2])) + np.array([x, y])
                    rect = Rectangle(
                        xy=anchor,
                        width=length,
                        height=width,
                        angle=angle * 180 / np.pi,
                        edgecolor='k',
                        facecolor=color,
                        alpha=0.05,
                    )
                    axglob[1].add_patch(rect)
                # Pedestrian
                elif int(y_hat[t, agent, 12].item()) == 1:
                    axglob[1].plot(x, y, marker='o', color=color, alpha=0.05, markerfacecolor=None)
                # Bike
                elif int(y_hat[t, agent, 13].item()) == 1:
                    axglob[1].plot(x, y, marker='+', color=color, alpha=0.05, markerfacecolor=None)
                else:
                    axglob[1].plot(x, y, marker='x', color=color, alpha=0.05)

            if t == (n_steps - 1) or t == 0 or t == 10:
                axglob[1].quiver(
                    y_hat[t, agent, 0].detach().numpy(),
                    y_hat[t, agent, 1].detach().numpy(),
                    y_hat[t, agent, 3].detach().numpy(),
                    y_hat[t, agent, 4].detach().numpy(),
                    width=0.003,
                    headwidth=5,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    alpha=1,
                )

    axglob[0].axis("equal")
    axglob[0].set_xlim((x_min, x_max))
    axglob[0].set_ylim((y_min, y_max))
    axglob[1].axis("equal")
    axglob[1].set_xlim((x_min, x_max))
    axglob[1].set_ylim((y_min, y_max))
    axglob[0].set_title("Groundtruth trajectories")
    axglob[1].set_title("Predicted trajectories")


    plt.show()
    figglob.savefig(
        vis_dir
        + config["misc"]["model_type"]
        + "_sequence_"
        + f"{args.sequence_idx:03}.png"
    )
