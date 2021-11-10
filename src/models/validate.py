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
    # Print configuration file
    print(OmegaConf.to_yaml(config))

    # Define model and trainer
    model = eval(config["misc"]["model_type"])(**config["model"])
    regressor = eval(config["misc"]["regressor_type"])(model, **config["regressor"])
    trainer = pl.Trainer(**config["trainer"])
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])
    # datamodule.setup()
    # val_dataloader = datamodule.val_dataloader()
    trainer.validate(model=regressor, datamodule=datamodule, ckpt_path=config["misc"]["ckpt_path"])



if __name__ == "__main__":
    main()