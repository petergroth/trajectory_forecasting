import argparse
import math
import os
from typing import Union

import hydra
import optuna
import pytorch_lightning as pl
import torch
import torch_geometric.nn
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch_geometric.data import Batch

from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule
from src.models.model import *
from src.models.train_waymo_rnn import *


class Objective(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, trial):

        # Suggest hyperparameters

        # Model
        hidden_size = trial.suggest_categorical(
            "hidden_size", [16, 32, 64, 96, 128, 192, 256]
        )
        dropout = trial.suggest_float("dropout", low=0.0, high=0.7, step=0.1)
        rnn_size = trial.suggest_categorical("rnn_size", [8, 16, 32, 64])
        rnn_edge_size = trial.suggest_categorical("rnn_edge_size", [4, 8, 16, 32, 64])
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
        latent_edge_features = trial.suggest_categorical(
            "latent_edge_features", [4, 8, 16, 32, 64]
        )

        # Regressor
        weight_decay = trial.suggest_categorical(
            "weight_decay", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1, 0.0]
        )
        training_horizon = trial.suggest_categorical(
            "training_horizon", [15, 25, 30, 40, 50, 70, 90]
        )
        teacher_forcing_ratio = trial.suggest_float(
            "teacher_forcing_ratio", low=0.0, high=0.3, step=0.05
        )
        min_dist = trial.suggest_float("min_dist", low=1.0, high=20.0, step=1.0)
        lr = trial.suggest_categorical("lr", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        edge_dropout = trial.suggest_float("edge_dropout", low=0.0, high=0.5, step=0.05)
        noise = trial.suggest_float("noise", low=0.0, high=0.01)

        # Datamodule
        # batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64, 96, 128])

        # Pack regressor parameters together
        model_kwargs = {
            "dropout": dropout,
            "rnn_size": rnn_size,
            "rnn_edge_size": rnn_edge_size,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "latent_edge_features": latent_edge_features,
        }
        regressor_kwargs = {
            "weight_decay": weight_decay,
            "min_dist": min_dist,
            "lr": lr,
            "training_horizon": training_horizon,
            "teacher_forcing_ratio": teacher_forcing_ratio,
            "noise": noise,
            "edge_dropout": edge_dropout,
        }

        # Update model arguments
        # self.config.datamodule.batch_size = batch_size

        regressor_dict = dict(self.config.regressor)
        regressor_dict.update(regressor_kwargs)
        self.config.regressor = DictConfig(regressor_dict)

        model_dict = dict(self.config.model)
        model_dict.update(model_kwargs)
        self.config.model = DictConfig(model_dict)

        # Seed for reproducibility
        seed_everything(self.config["misc"]["seed"], workers=True)
        # Load data, model, and regressor
        datamodule = eval(self.config["misc"]["dm_type"])(**self.config["datamodule"])
        # Define model
        model_dict = self.config["model"]
        model_type = self.config["misc"]["model_type"]

        # Define LightningModule
        regressor = eval(self.config["misc"]["regressor_type"])(
            model_type=model_type,
            model_dict=dict(model_dict),
            **self.config["regressor"]
        )

        log_dict = regressor_kwargs
        # log_dict.update(trainer_kwargs)
        log_dict.update(model_kwargs)
        # log_dict["batch_size"] = batch_size

        # Setup logging
        wandb_logger = WandbLogger(
            entity="petergroth",
            config=log_dict,
            project=self.config["logger"]["project"],
            reinit=True,
        )
        wandb_logger.watch(
            regressor, log_freq=self.config["misc"]["log_freq"], log_graph=False
        )

        callbacks = [
            EarlyStopping(monitor="val_total_loss", patience=4, min_delta=1),
            PyTorchLightningPruningCallback(trial, monitor="val_total_loss"),
        ]

        # Create trainer, fit, and validate
        trainer = pl.Trainer(
            logger=wandb_logger,
            **self.config["trainer"],
            enable_checkpointing=False,
            callbacks=callbacks
        )
        trainer.fit(model=regressor, datamodule=datamodule)

        val_total_loss = trainer.early_stopping_callback.best_score.item()
        wandb_logger.log_metrics({"best_total_val_loss": val_total_loss})
        wandb_logger.finalize("0")
        wandb_logger.experiment.finish()
        wandb.finish()
        del trainer

        return val_total_loss


@hydra.main(config_path="../../../configs/waymo/", config_name="config")
def main(config):
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize", study_name=config.logger.version, pruner=pruner
    )
    study.optimize(Objective(config), n_trials=100, timeout=32000, gc_after_trial=True)


if __name__ == "__main__":
    main()
