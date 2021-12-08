from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import optuna
import pytorch_lightning as pl
import torch
import torch_geometric.nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule
from src.models.train_waymo_model_reduced import *
import torchmetrics
from torch_geometric.data import Batch
from src.models.model import *
import hydra
from src.models.optuna_objectives import *


@hydra.main(config_path="../../configs/waymo/", config_name="config")
def main(config):
    # pruner = optuna.pruners.MedianPruner()
    if config.model.model_type == ""

    study = optuna.create_study(direction="minimize", study_name=config.logger.version)
    study.optimize(Objective(config), n_trials=100, timeout=32000, gc_after_trial=True)