import argparse

import hydra
import pytorch_lightning as pl
import yaml

from src.data.dataset_waymo import (OneStepWaymoDataModule,
                                    SequentialWaymoDataModule)
from src.models.model import *
from src.models.train_waymo_model import *

# @hydra.main(config_path="../../configs/waymo/", config_name="config")
# def main(config):
#     # Define model and trainer
#     regressor = eval(config["misc"]["regressor_type"]).load_from_checkpoint(config["misc"]["ckpt_path"])
#     trainer = pl.Trainer(**config["trainer"])
#     datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])
#     trainer.validate(model=regressor, datamodule=datamodule)


@hydra.main(config_path="../../configs/waymo/", config_name="config")
def main(config):
    # Define model and trainer
    regressor = eval(config["misc"]["regressor_type"]).load_from_checkpoint(
        config["misc"]["ckpt_path"]
    )
    config["logger"]["offline"] = False
    config["logger"]["version"] = "validation_test_02"
    config["logger"]["project"] = "meeting_nov_09"
    # Setup logging
    wandb_logger = WandbLogger(
        entity="petergroth", config=dict(config), **config["logger"]
    )
    wandb_logger.watch(regressor, log_freq=1)
    trainer = pl.Trainer(logger=wandb_logger, **config["trainer"])
    config["datamodule"]["val_batch_size"] = 1
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])
    trainer.validate(model=regressor, datamodule=datamodule)


if __name__ == "__main__":
    main()
