import argparse
import math
import os
import random
from typing import Union

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric.nn
import torchmetrics
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch_geometric.data import Batch

from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule
from src.training_modules.train_waymo_nodel import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("ckpt_path")
    parser.add_argument("max_epochs", type=int)
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["trainer"]["max_epochs"] = args.max_epochs
    config["trainer"]["max_time"] = None

    # Seed for reproducibility
    seed_everything(config["misc"]["seed"], workers=True)
    # Load data, model, and regressor
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])

    # Define LightningModule
    regressor = eval(config["misc"]["regressor_type"]).load_from_checkpoint(
        args.ckpt_path
    )

    # Setup logging (using saved yaml file)
    wandb_logger = WandbLogger(
        entity="petergroth",
        **config["logger"],
    )
    wandb_logger.watch(regressor, log_freq=config["misc"]["log_freq"])

    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=config["logger"]["version"],
        monitor="val_total_loss",
        save_last=True,
        save_top_k=3,
    )
    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    # Create trainer, fit, and validate
    trainer = pl.Trainer(
        logger=wandb_logger,
        **config["trainer"],
        callbacks=[checkpoint_callback, summary_callback],
    )

    trainer.fit(model=regressor, datamodule=datamodule, ckpt_path=args.ckpt_path)
    trainer.validate(regressor, datamodule=datamodule, ckpt_path="best")
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
