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
from src.models.train_waymo_model import *


class Objective(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, trial):

        # Suggest hyperparameters

        # Model
        hidden_size = trial.suggest_categorical(
            "hidden_size", [32, 64, 96, 128, 192, 256]
        )
        latent_edge_features = trial.suggest_categorical(
            "latent_edge_features", [32, 64, 96, 128]
        )
        rnn_size = trial.suggest_categorical("rnn_size", [8, 16, 24, 32, 64])
        dropout = trial.suggest_float("dropout", low=0.0, high=0.5)
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
        rnn_type = trial.suggest_categorical("rnn_type", ["GRU", "LSTM"])
        aggregate = trial.suggest_categorical("aggregate", [True, False])

        # Regressor
        weight_decay = trial.suggest_categorical(
            "weight_decay", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1, 0, 1, 10]
        )
        training_horizon = trial.suggest_categorical(
            "training_horizon", [25, 30, 40, 50, 70, 90]
        )
        teacher_forcing_ratio = trial.suggest_float(
            "teacher_forcing_ratio", low=0.0, high=0.3, step=0.05
        )
        min_dist = trial.suggest_float("min_dist", low=1.0, high=20.0, step=1.0)
        fully_connected = trial.suggest_categorical("fully_connected", [True, False])
        lr = trial.suggest_categorical("lr", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        # noise = trial.suggest_float("noise", low=0.0, high=1.0)

        # Datamodule
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64, 96, 128])

        # Trainer
        # stochastic_weight_avg = trial.suggest_categorical("stochastic_weight_avg", ["True", "False"])
        # gradient_clip_val = trial.suggest_float("gradient_clip_val", low=0.0, high=1.0)

        # Pack regressor parameters together
        model_kwargs = {
            "hidden_size": hidden_size,
            "latent_edge_features": latent_edge_features,
            "rnn_size": rnn_size,
            "dropout": dropout,
            "num_layers": num_layers,
            "rnn_type": rnn_type,
            "aggregate": aggregate,
        }
        regressor_kwargs = {
            "weight_decay": weight_decay,
            "fully_connected": fully_connected,
            "min_dist": min_dist,
            "lr": lr,
            "training_horizon": training_horizon,
            "teacher_forcing_ratio": teacher_forcing_ratio,
            # "noise": noise
        }
        # trainer_kwargs = {
        #     "gradient_clip_val": gradient_clip_val,
        #     # "stochastic_weight_avg": stochastic_weight_avg
        # }

        # Update model arguments
        self.config.datamodule.batch_size = batch_size

        regressor_dict = dict(self.config.regressor)
        regressor_dict.update(regressor_kwargs)
        self.config.regressor = DictConfig(regressor_dict)

        # trainer_dict = dict(self.config.trainer)
        # trainer_dict.update(trainer_kwargs)
        # self.config.trainer = DictConfig(trainer_dict)

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
        log_dict["batch_size"] = batch_size

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

        callbacks = [EarlyStopping(monitor="val_total_loss", patience=4, min_delta=1)]

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
        del trainer

        return val_total_loss


@hydra.main(config_path="../../../configs/waymo/", config_name="config")
def main(config):
    # pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", study_name=config.logger.version)
    study.optimize(Objective(config), n_trials=100, timeout=32000, gc_after_trial=True)


if __name__ == "__main__":
    main()
