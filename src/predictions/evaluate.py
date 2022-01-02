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
import matplotlib.colors as colors
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from src.data.dataset_waymo import OneStepWaymoDataModule
from src.training_modules.train_waymo_UA import *

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
    datamodule.setup("test")
    dataset = datamodule.test_dataset
    # Extract batch and add missing attributes
    batch = dataset.__getitem__(sequence_idx)
    batch.batch = torch.zeros(batch.x.size(0)).type(torch.int64)
    batch.num_graphs = 1

    # Make and return predictions
    y_hat, y_target, mask, Sigma = regressor.predict_step(batch, prediction_horizon=n_steps)
    return y_hat.detach(), y_target, mask, batch, Sigma.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("ckpt_path")
    parser.add_argument("output_path")
    parser.add_argument("sequence_idx", type=int)
    parser.add_argument("n_steps", type=int, default=51)
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed
    seed_everything(config["misc"]["seed"], workers=True)
    # Load datamodule
    config["datamodule"]["val_batch_size"] = 1
    config["datamodule"]["batch_size"] = 1
    config["datamodule"]["shuffle"] = False
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])

    # Load correct model
    if config["misc"]["model_type"] != "ConstantModel":
        regressor = eval(config["misc"]["regressor_type"]).load_from_checkpoint(args.ckpt_path)
        regressor.eval()
    else:
        regressor = eval(config["misc"]["regressor_type"])(**config["regressor"])

    # Load trainer
    trainer = pl.Trainer(**config["trainer"])
    datamodule.setup("fit")
    trainer.validate(regressor, ckpt_path=args.ckpt_path, datamodule=datamodule)

    # Test
    datamodule.setup("test")
    trainer.test(regressor, datamodule=datamodule, ckpt_path=args.ckpt_path)



if __name__ == "__main__":
    main()

