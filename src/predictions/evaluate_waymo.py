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
from matplotlib.patches import Circle, Ellipse, Rectangle
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from src.training_modules.train_waymo_model import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("ckpt_path")
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed
    seed_everything(config["misc"]["seed"], workers=True)
    # Load datamodule
    config["datamodule"]["val_batch_size"] = 32
    config["datamodule"]["shuffle"] = False
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])

    # Load correct model
    if config["misc"]["model_type"] != "ConstantModel":
        regressor = eval(config["misc"]["regressor_type"]).load_from_checkpoint(
            args.ckpt_path
        )
        regressor.eval()
    else:
        regressor = eval(config["misc"]["regressor_type"])(**config["regressor"])

    # Load trainer
    trainer = pl.Trainer(**config["trainer"])
    # datamodule.setup("validate")
    # trainer.validate(regressor, ckpt_path=args.ckpt_path, datamodule=datamodule)

    # Test
    datamodule.setup("test")
    trainer.test(regressor, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
