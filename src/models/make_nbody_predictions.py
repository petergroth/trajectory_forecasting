import argparse
import pytorch_lightning as pl
from src.data.dataset_nbody import *
from src.models.train_nbody_model import *
import yaml
from pytorch_lightning.utilities.seed import seed_everything
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# from models import ConstantModel
from matplotlib.patches import Circle


def make_predictions(path, config, sequence_idx=0):
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
        regressor = ConstantPhysicalBaselineModule(**config["regressor"])
    # Setup
    regressor.eval()
    datamodule.setup()
    loader = datamodule.val_dataloader()
    # Define output path
    dirpath = "src/predictions/raw_preds/nbody/" + config["logger"]["version"]
    os.makedirs(dirpath, exist_ok=True)
    for i, batch in enumerate(loader):
        if i == sequence_idx:
            y_hat, y_target = regressor.predict_step(batch)
            torch.save((y_hat, y_target), dirpath + f"/sequence_{i:04}.pt")
            return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("ckpt_path")
    parser.add_argument("output_path")
    parser.add_argument("sequence_idx", type=int)
    args = parser.parse_args()

    # Load yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Computes predictions for specified sequence
    make_predictions(
        path=args.ckpt_path,
        config=config,
        sequence_idx=args.sequence_idx,
    )

    # Create directory for current model
    # vis_dir = "visualisations/predictions/waymo/" + config["logger"]["version"] + "/"
    os.makedirs(args.output_path, exist_ok=True)
    # Location of prediction files
    dir = "src/predictions/raw_preds/nbody/" + config["logger"]["version"] + "/"
    # Load first file in directory
    path = "sequence_" + f"{args.sequence_idx:04}.pt"
    y_hat, y_target = torch.load(dir + path)
    y_hat = y_hat.detach()
    n_steps, n_agents, n_features = y_hat.shape
    # Extract boundaries
    x_min, x_max, y_min, y_max = (
        torch.min(y_target[:, :, 0]).item(),
        torch.max(y_target[:, :, 0]).item(),
        torch.min(y_target[:, :, 1]).item(),
        torch.max(y_target[:, :, 1]).item(),
    )

    figglob, axglob = plt.subplots(1, 2, figsize=(20, 10))

    colors = [
        (np.random.random(), np.random.random(), np.random.random())
        for _ in range(n_agents)
    ]
    for agent in range(n_agents):
        # color = (np.random.random(), np.random.random(), np.random.random())
        x = y_target[:, agent, 0].detach().numpy()
        y = y_target[:, agent, 1].detach().numpy()
        axglob[0].scatter(x=x, y=y, s=50, color=colors[agent], alpha=0.2, edgecolors="k")
        axglob[0].quiver(
            y_target[0, agent, 0].detach().numpy(),
            y_target[0, agent, 1].detach().numpy(),
            y_target[0, agent, 2].detach().numpy(),
            y_target[0, agent, 3].detach().numpy(),
            width=0.003,
            headwidth=5,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="lightgrey",
        )
        axglob[0].quiver(
            y_target[10, agent, 0].detach().numpy(),
            y_target[10, agent, 1].detach().numpy(),
            y_target[10, agent, 2].detach().numpy(),
            y_target[10, agent, 3].detach().numpy(),
            width=0.003,
            headwidth=5,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="gray",
        )
        axglob[0].quiver(
            y_target[n_steps - 1, agent, 0].detach().numpy(),
            y_target[n_steps - 1, agent, 1].detach().numpy(),
            y_target[n_steps - 1, agent, 2].detach().numpy(),
            y_target[n_steps - 1, agent, 3].detach().numpy(),
            width=0.003,
            headwidth=5,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="k",
        )

        x = y_hat[:, agent, 0].detach().numpy()
        y = y_hat[:, agent, 1].detach().numpy()
        axglob[1].scatter(x=x, y=y, s=30, color=colors[agent], alpha=0.2, edgecolors="k")
        axglob[1].quiver(
            y_hat[0, agent, 0].detach().numpy(),
            y_hat[0, agent, 1].detach().numpy(),
            y_hat[0, agent, 2].detach().numpy(),
            y_hat[0, agent, 3].detach().numpy(),
            width=0.003,
            headwidth=5,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="lightgrey",
        )
        axglob[1].quiver(
            y_hat[10, agent, 0].detach().numpy(),
            y_hat[10, agent, 1].detach().numpy(),
            y_hat[10, agent, 2].detach().numpy(),
            y_hat[10, agent, 3].detach().numpy(),
            width=0.003,
            headwidth=5,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="gray",
        )
        axglob[1].quiver(
            y_hat[n_steps - 1, agent, 0].detach().numpy(),
            y_hat[n_steps - 1, agent, 1].detach().numpy(),
            y_hat[n_steps - 1, agent, 2].detach().numpy(),
            y_hat[n_steps - 1, agent, 3].detach().numpy(),
            width=0.003,
            headwidth=5,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="k",
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
    figglob.savefig(f"{args.output_path}/sequence_{args.sequence_idx:04}.png")


if __name__ == "__main__":
    main()
