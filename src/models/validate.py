import argparse
import pytorch_lightning as pl
import torch
import torch_geometric.nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule
import torchmetrics
from torch.nn.functional import one_hot
from torch_geometric.data import Batch
from src.models.model import *
from src.models.train_waymo_model import *
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import math


@hydra.main(config_path="../../configs/waymo/", config_name="config")
def main(config):
        # Define model and trainer
    regressor = eval(config["misc"]["regressor_type"]).load_from_checkpoint(config["misc"]["ckpt_path"])
    trainer = pl.Trainer(**config["trainer"])
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])
    trainer.validate(model=regressor, datamodule=datamodule)

if __name__ == "__main__":
    main()