import argparse
import math
import os
import random
from typing import Union

import hydra
import pytorch_lightning as pl
import torch
import torch_geometric.nn
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch_geometric.data import Batch

from src.data.dataset_waymo import (OneStepWaymoDataModule,
                                    SequentialWaymoDataModule)
from src.models.model import *


class WaymoModule(pl.LightningModule):
    def __init__(
        self,
        model_type: Union[None, str],
        model_dict: Union[None, dict],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        noise: Union[None, float] = None,
        teacher_forcing_ratio: float = 0.0,
        min_dist: int = 0,
        n_neighbours: int = 30,
        edge_weight: bool = False,
        self_loop: bool = False,
        out_features: int = 6,
        node_features: int = 9,
        edge_features: int = 1,
        normalise: bool = True,
        training_horizon: int = 90,
        edge_dropout: float = 0,
        prediction_horizon: int = 91,
    ):
        super().__init__()
        # Training metrics
        self.train_ade_loss = torchmetrics.MeanSquaredError()
        self.train_fde_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        # Validation metrics
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_fde_ttp_loss = torchmetrics.MeanSquaredError()
        self.val_ade_ttp_loss = torchmetrics.MeanSquaredError()
        # Testing metrics
        self.test_ade_loss = torchmetrics.MeanSquaredError()
        self.test_fde_loss = torchmetrics.MeanSquaredError()
        self.test_vel_loss = torchmetrics.MeanSquaredError()
        self.test_fde_ttp_loss = torchmetrics.MeanSquaredError()
        self.test_ade_ttp_loss = torchmetrics.MeanSquaredError()

        # Instantiate model
        self.model_type = model_type
        self.model = eval(model_type)(**model_dict)

        # Learning parameters
        self.normalise = normalise
        self.global_scale = 8.025897979736328
        # self.global_scale = 1
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.training_horizon = training_horizon
        self.norm_index = [0, 1, 2, 3, 4, 5, 6]
        self.pos_index = [0, 1]
        self.edge_dropout = edge_dropout
        self.prediction_horizon = prediction_horizon

        # Model parameters
        self.rnn_type = model_dict["rnn_type"]
        self.out_features = out_features
        self.edge_features = edge_features
        self.node_features = node_features

        # Graph parameters
        self.min_dist = min_dist
        self.n_neighbours = n_neighbours
        self.edge_weight = edge_weight
        self.self_loop = self_loop

        self.save_hyperparameters()

    def training_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Determine valid initialisations at t=11
        mask = batch.x[:, :, -1]
        valid_mask = mask[:, 10] > 0

        # Discard non-valid nodes as no initial trajectories will be known
        batch.x = batch.x[valid_mask]
        batch.batch = batch.batch[valid_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[valid_mask]
        batch.type = batch.type[valid_mask]

        # CARS
        type_mask = batch.type[:, 1] == 1
        batch.x = batch.x[type_mask]
        batch.batch = batch.batch[type_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[type_mask]
        batch.type = batch.type[type_mask]

        # Discard future values not used for training
        batch.x = batch.x[:, : (self.training_horizon + 1)]

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Discard masks and extract static features
        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = batch.x[:, 10, self.out_features :]
        static_features = static_features.type_as(batch.x)
        edge_attr = None
        # Extract dimensions and allocate predictions
        n_nodes = batch.num_nodes
        y_predictions = torch.zeros((n_nodes, self.training_horizon, self.out_features))
        y_predictions = y_predictions.type_as(batch.x)

        # Tensor of position and velocity targets
        y_target = batch.x[:, 1 : (self.training_horizon + 1), : self.out_featuers]
        y_target = y_target.type_as(batch.x)

        assert y_target.shape == y_predictions.shape

        # Initial hidden state
        if self.rnn_type == "GRU":
            h_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            h_node = h_node.type_as(batch.x)
            h_edge = h_edge.type_as(batch.x)
            c_node, c_edge = None, None
        else:
            h_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            h_node = h_node.type_as(batch.x)
            h_edge = h_edge.type_as(batch.x)
            c_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            c_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            c_node = c_node.type_as(batch.x)
            c_edge = c_edge.type_as(batch.x)

        ######################
        # History            #
        ######################

        for t in range(11):

            # Extract current input
            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :]
            x_t = x_t.type_as(batch.x)

            # Add noise if specified
            if self.noise is not None:
                x_t[:, : self.out_features] += self.noise * torch.randn_like(
                    x_t[:, : self.out_features]
                )

            ######################
            # Graph construction #
            ######################

            # Construct edges
            edge_index = torch_geometric.nn.radius_graph(
                x=x_t[:, :2],
                r=self.min_dist,
                batch=batch.batch[mask_t],
                loop=self.self_loop,
                max_num_neighbors=self.n_neighbours,
                flow="source_to_target",
            )

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(
                edge_index, num_nodes=x_t.shape[0]
            )

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            if self.edge_dropout > 0:
                edge_index, edge_attr = dropout_adj(
                    edge_index=edge_index, edge_attr=edge_attr, p=self.edge_dropout
                )

            #######################
            # Training 1/2        #
            #######################

            # Normalise input graph
            if self.normalise:
                # Center node positions
                x_t[:, self.pos_index] -= batch.loc[batch.batch][mask_t][
                    :, self.pos_index
                ]
                # Scale all features (except yaws) with global scaler
                x_t[:, self.norm_index] /= self.global_scale
                if edge_attr is not None:
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain predicted delta dynamics
            if self.rnn_type == "GRU":
                hidden_in = (h_node[:, mask_t], h_edge[:, mask_t])
                delta_x, h_t = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=hidden_in,
                )
                # Update hidden states
                h_node[:, mask_t] = h_t[0]
                h_edge[:, mask_t] = h_t[1]

            else:  # LSTM
                hidden_in = (
                    (h_node[:, mask_t], c_node[:, mask_t]),
                    (h_edge[:, mask_t], c_edge[:, mask_t]),
                )

                delta_x, (h_node_out, h_edge_out) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=hidden_in,
                )
                h_node[:, mask_t] = h_node_out[0]
                c_node[:, mask_t] = h_node_out[1]
                h_edge[:, mask_t] = h_edge_out[0]
                c_edge[:, mask_t] = h_edge_out[1]

            vel = delta_x[:, [0, 1]]
            pos = batch.x[mask_t, t][:, self.pos_index] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features[mask_t]], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[mask_t, t, :] = x_t[:, : self.out_features]

        # If using teacher_forcing, draw sample and accept <teach_forcing_ratio*100> % of the time. Else, deny.
        use_groundtruth = random.random() < self.teacher_forcing_ratio

        ######################
        # Future             #
        ######################

        for t in range(11, self.training_horizon):
            # Use groundtruth 'teacher_forcing_ratio' % of the time
            if use_groundtruth:
                # x_t = torch.cat([batch.x[:, t, :], batch.type], dim=1)
                x_t = batch.x[:, t, :].clone()
            x_prev = x_t.clone()

            ######################
            # Graph construction #
            ######################

            # Construct edges
            edge_index = torch_geometric.nn.radius_graph(
                x=x_t[:, :2],
                r=self.min_dist,
                batch=batch.batch,
                loop=self.self_loop,
                max_num_neighbors=self.n_neighbours,
                flow="source_to_target",
            )

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(
                edge_index, num_nodes=x_t.shape[0]
            )

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            if self.edge_dropout > 0:
                edge_index, edge_attr = dropout_adj(
                    edge_index=edge_index, edge_attr=edge_attr, p=self.edge_dropout
                )

            #######################
            # Training 2/2        #
            #######################

            # Normalise input graph
            if self.normalise:
                # Center node positions
                x_t[:, self.pos_index] -= batch.loc[batch.batch][:, self.pos_index]
                # Scale all features (except yaws) with global scaler
                x_t[:, self.norm_index] /= self.global_scale
                if edge_attr is not None:
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            else:
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )

            vel = delta_x[:, [0, 1]]
            pos = x_prev[:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[:, t, :] = x_t[:, : self.out_features]

        # Determine valid input and target pairs. Compute loss mask as their intersection
        loss_mask_target = mask[:, 1 : (self.training_horizon + 1)]
        loss_mask_input = mask[:, 0 : self.training_horizon]
        loss_mask = torch.logical_and(loss_mask_input, loss_mask_target)

        # Determine valid end-points
        fde_mask_target = mask[:, -1]
        fde_mask_input = mask[:, -2]
        fde_mask = torch.logical_and(fde_mask_input, fde_mask_target)

        assert (y_target[:, :, [0, 1]][loss_mask] == 0).sum() == 0
        assert (y_predictions[:, :, [0, 1]][loss_mask] == 0).sum() == 0

        # Compute and log loss
        fde_loss = self.train_fde_loss(
            y_predictions[fde_mask, -1][:, [0, 1]], y_target[fde_mask, -1][:, [0, 1]]
        )
        ade_loss = self.train_ade_loss(
            y_predictions[:, :, [0, 1]][loss_mask], y_target[:, :, [0, 1]][loss_mask]
        )
        vel_loss = self.train_vel_loss(
            y_predictions[:, :, [2, 3]][loss_mask], y_target[:, :, [2, 3]][loss_mask]
        )

        self.log(
            "train_fde_loss",
            fde_loss,
            on_step=True,
            on_epoch=True,
            batch_size=fde_mask.sum().item(),
        )
        self.log(
            "train_ade_loss",
            ade_loss,
            on_step=True,
            on_epoch=True,
            batch_size=loss_mask.sum().item(),
        )
        self.log(
            "train_vel_loss",
            vel_loss,
            on_step=True,
            on_epoch=True,
            batch_size=loss_mask.sum().item(),
        )

        loss = ade_loss

        self.log(
            "train_total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=loss_mask.sum().item(),
        )

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Validate on sequential dataset. First 11 observations are used to prime the model.
        # Loss is computed on remaining 80 samples using rollout.

        # Determine valid initialisations at t=11
        mask = batch.x[:, :, -1]
        valid_mask = mask[:, 10] > 0

        # Discard non-valid nodes as no initial trajectories will be known
        batch.x = batch.x[valid_mask]
        batch.batch = batch.batch[valid_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[valid_mask]
        batch.type = batch.type[valid_mask]

        # CARS
        type_mask = batch.type[:, 1] == 1
        batch.x = batch.x[type_mask]
        batch.batch = batch.batch[type_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[type_mask]
        batch.type = batch.type[type_mask]

        # Update input using prediction horizon
        batch.x = batch.x[:, : self.prediction_horizon]

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((self.prediction_horizon - 11, n_nodes, self.out_features))
        y_hat = y_hat.type_as(batch.x)
        y_target = torch.zeros(
            (self.prediction_horizon - 11, n_nodes, self.out_features)
        )
        y_target = y_target.type_as(batch.x)
        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = batch.x[:, 10, self.out_features :]
        static_features = static_features.type_as(batch.x)
        edge_attr = None

        # Initial hidden state
        if self.rnn_type == "GRU":
            h_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            h_node = h_node.type_as(batch.x)
            h_edge = h_edge.type_as(batch.x)
            c_node, c_edge = None, None
        else:
            h_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            h_node = h_node.type_as(batch.x)
            h_edge = h_edge.type_as(batch.x)
            c_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            c_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            c_node = c_node.type_as(batch.x)
            c_edge = c_edge.type_as(batch.x)

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :]
            x_t = x_t.type_as(batch.x)

            # Construct edges
            edge_index = torch_geometric.nn.radius_graph(
                x=x_t[:, :2],
                r=self.min_dist,
                batch=batch.batch[mask_t],
                loop=self.self_loop,
                max_num_neighbors=self.n_neighbours,
                flow="source_to_target",
            )

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(
                edge_index, num_nodes=x_t.shape[0]
            )

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Validation 1/2     #
            ######################

            # Normalise input graph
            if self.normalise:
                # Center node positions
                x_t[:, self.pos_index] -= batch.loc[batch.batch][mask_t][
                    :, self.pos_index
                ]
                # Scale all features (except yaws) with global scaler
                x_t[:, self.norm_index] /= self.global_scale
                if edge_attr is not None:
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if self.rnn_type == "GRU":
                hidden_in = (h_node[:, mask_t], h_edge[:, mask_t])
                delta_x, h_t = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=hidden_in,
                )
                # Update hidden states
                h_node[:, mask_t] = h_t[0]
                h_edge[:, mask_t] = h_t[1]

            else:  # LSTM
                hidden_in = (
                    (h_node[:, mask_t], c_node[:, mask_t]),
                    (h_edge[:, mask_t], c_edge[:, mask_t]),
                )
                delta_x, (h_node_out, h_edge_out) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=hidden_in,
                )
                h_node[:, mask_t] = h_node_out[0]
                c_node[:, mask_t] = h_node_out[1]
                h_edge[:, mask_t] = h_edge_out[0]
                c_edge[:, mask_t] = h_edge_out[1]

            if t == 10:
                vel = delta_x[:, [0, 1]]
                pos = batch.x[mask_t, t][:, self.pos_index] + 0.1 * vel
                predicted_graph = torch.cat([pos, vel, static_features[mask_t]], dim=-1)
                predicted_graph = predicted_graph.type_as(batch.x)

        # Save first prediction and target
        y_hat[0, mask_t, :] = predicted_graph[:, : self.out_features]
        y_target[0, mask_t, :] = batch.x[mask_t, 11, : self.out_features]

        ######################
        # Future             #
        ######################

        for t in range(11, self.prediction_horizon - 1):

            ######################
            # Graph construction #
            ######################

            # Latest prediction as input
            x_t = predicted_graph.clone()

            # Construct edges
            edge_index = torch_geometric.nn.radius_graph(
                x=x_t[:, :2],
                r=self.min_dist,
                batch=batch.batch,
                loop=self.self_loop,
                max_num_neighbors=self.n_neighbours,
                flow="source_to_target",
            )

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(
                edge_index, num_nodes=x_t.shape[0]
            )

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Validation 2/2     #
            ######################

            # Normalise input graph
            if self.normalise:
                # Center node positions
                x_t[:, self.pos_index] -= batch.loc[batch.batch][:, self.pos_index]
                # Scale all features (except yaws) with global scaler
                x_t[:, self.norm_index] /= self.global_scale
                if edge_attr is not None:
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            else:
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )

            vel = delta_x[:, [0, 1]]
            pos = predicted_graph[:, [0, 1]] + 0.1 * vel
            predicted_graph = torch.cat([pos, vel, static_features], dim=-1)

            predicted_graph = predicted_graph.type_as(batch.x)

            # Save prediction alongside true value (next time step state)
            y_hat[t - 10, :, :] = predicted_graph[:, : self.out_features]
            y_target[t - 10, :, :] = batch.x[:, t + 1, : self.out_features]

        fde_mask = mask[:, -1]
        val_mask = mask[:, 11:].permute(1, 0)

        # Compute and log loss
        fde_loss = self.val_fde_loss(
            y_hat[-1, fde_mask][:, [0, 1]], y_target[-1, fde_mask][:, [0, 1]]
        )
        ade_loss = self.val_ade_loss(
            y_hat[:, :, [0, 1]][val_mask], y_target[:, :, [0, 1]][val_mask]
        )
        vel_loss = self.val_vel_loss(
            y_hat[:, :, [2, 3]][val_mask], y_target[:, :, [2, 3]][val_mask]
        )

        # Compute losses on "tracks_to_predict"
        fde_ttp_mask = torch.logical_and(fde_mask, batch.tracks_to_predict)
        fde_ttp_loss = self.val_fde_ttp_loss(
            y_hat[-1, fde_ttp_mask][:, [0, 1]], y_target[-1, fde_ttp_mask][:, [0, 1]]
        )
        ade_ttp_mask = torch.logical_and(
            val_mask,
            batch.tracks_to_predict.expand(
                (self.prediction_horizon - 11, mask.size(0))
            ),
        )
        ade_ttp_loss = self.val_ade_loss(
            y_hat[:, :, [0, 1]][ade_ttp_mask], y_target[:, :, [0, 1]][ade_ttp_mask]
        )

        ######################
        # Logging            #
        ######################

        self.log("val_ade_loss", ade_loss, batch_size=val_mask.sum().item())
        self.log("val_fde_loss", fde_loss, batch_size=fde_mask.sum().item())
        self.log("val_vel_loss", vel_loss, batch_size=val_mask.sum().item())
        loss = ade_loss
        self.log("val_total_loss", loss, batch_size=val_mask.sum().item())
        self.log("val_fde_ttp_loss", fde_ttp_loss, batch_size=fde_ttp_mask.sum().item())
        self.log("val_ade_ttp_loss", ade_ttp_loss, batch_size=ade_ttp_mask.sum().item())

        return loss

    def test_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Test on sequential dataset. First 11 observations are used to prime the model.

        # Determine valid initialisations at t=11
        mask = batch.x[:, :, -1]
        valid_mask = mask[:, 10] > 0

        # Discard non-valid nodes as no initial trajectories will be known
        batch.x = batch.x[valid_mask]
        batch.batch = batch.batch[valid_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[valid_mask]
        batch.type = batch.type[valid_mask]

        # CARS
        type_mask = batch.type[:, 1] == 1
        batch.x = batch.x[type_mask]
        batch.batch = batch.batch[type_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[type_mask]
        batch.type = batch.type[type_mask]

        # Update input using prediction horizon
        batch.x = batch.x[:, : self.prediction_horizon]

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((self.prediction_horizon - 11, n_nodes, self.out_features))
        y_hat = y_hat.type_as(batch.x)
        y_target = torch.zeros(
            (self.prediction_horizon - 11, n_nodes, self.out_features)
        )
        y_target = y_target.type_as(batch.x)
        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = batch.x[:, 10, self.out_features :]
        static_features = static_features.type_as(batch.x)
        edge_attr = None

        # Initial hidden state
        if self.rnn_type == "GRU":
            h_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            h_node = h_node.type_as(batch.x)
            h_edge = h_edge.type_as(batch.x)
            c_node, c_edge = None, None
        else:
            h_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            h_node = h_node.type_as(batch.x)
            h_edge = h_edge.type_as(batch.x)
            c_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            c_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            c_node = c_node.type_as(batch.x)
            c_edge = c_edge.type_as(batch.x)

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :]
            x_t = x_t.type_as(batch.x)

            # Construct edges
            edge_index = torch_geometric.nn.radius_graph(
                x=x_t[:, :2],
                r=self.min_dist,
                batch=batch.batch[mask_t],
                loop=self.self_loop,
                max_num_neighbors=self.n_neighbours,
                flow="source_to_target",
            )

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(
                edge_index, num_nodes=x_t.shape[0]
            )

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Testing 1/2        #
            ######################

            # Normalise input graph
            if self.normalise:
                # Center node positions
                x_t[:, self.pos_index] -= batch.loc[batch.batch][mask_t][
                    :, self.pos_index
                ]
                # Scale all features (except yaws) with global scaler
                x_t[:, self.norm_index] /= self.global_scale
                if edge_attr is not None:
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if self.rnn_type == "GRU":
                hidden_in = (h_node[:, mask_t], h_edge[:, mask_t])
                delta_x, h_t = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=hidden_in,
                )
                # Update hidden states
                h_node[:, mask_t] = h_t[0]
                h_edge[:, mask_t] = h_t[1]

            else:  # LSTM
                hidden_in = (
                    (h_node[:, mask_t], c_node[:, mask_t]),
                    (h_edge[:, mask_t], c_edge[:, mask_t]),
                )
                delta_x, (h_node_out, h_edge_out) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=hidden_in,
                )
                h_node[:, mask_t] = h_node_out[0]
                c_node[:, mask_t] = h_node_out[1]
                h_edge[:, mask_t] = h_edge_out[0]
                c_edge[:, mask_t] = h_edge_out[1]

            if t == 10:
                vel = delta_x[:, [0, 1]]
                pos = batch.x[mask_t, t][:, self.pos_index] + 0.1 * vel
                predicted_graph = torch.cat([pos, vel, static_features[mask_t]], dim=-1)
                predicted_graph = predicted_graph.type_as(batch.x)

        # Save first prediction and target
        y_hat[0, mask_t, :] = predicted_graph[:, : self.out_features]
        y_target[0, mask_t, :] = batch.x[mask_t, 11, : self.out_features]

        ######################
        # Future             #
        ######################

        for t in range(11, self.prediction_horizon - 1):

            ######################
            # Graph construction #
            ######################

            # Latest prediction as input
            x_t = predicted_graph.clone()

            # Construct edges
            edge_index = torch_geometric.nn.radius_graph(
                x=x_t[:, :2],
                r=self.min_dist,
                batch=batch.batch,
                loop=self.self_loop,
                max_num_neighbors=self.n_neighbours,
                flow="source_to_target",
            )

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(
                edge_index, num_nodes=x_t.shape[0]
            )

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Testing 2/2        #
            ######################

            # Normalise input graph
            if self.normalise:
                # Center node positions
                x_t[:, self.pos_index] -= batch.loc[batch.batch][:, self.pos_index]
                # Scale all features (except yaws) with global scaler
                x_t[:, self.norm_index] /= self.global_scale
                if edge_attr is not None:
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            else:
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )

            vel = delta_x[:, [0, 1]]
            pos = predicted_graph[:, [0, 1]] + 0.1 * vel
            predicted_graph = torch.cat([pos, vel, static_features], dim=-1)

            predicted_graph = predicted_graph.type_as(batch.x)

            # Save prediction alongside true value (next time step state)
            y_hat[t - 10, :, :] = predicted_graph[:, : self.out_features]
            y_target[t - 10, :, :] = batch.x[:, t + 1, : self.out_features]

        fde_mask = mask[:, -1]
        val_mask = mask[:, 11:].permute(1, 0)

        # Compute and log loss
        fde_loss = self.test_fde_loss(
            y_hat[-1, fde_mask][:, [0, 1]], y_target[-1, fde_mask][:, [0, 1]]
        )
        ade_loss = self.test_ade_loss(
            y_hat[:, :, [0, 1]][val_mask], y_target[:, :, [0, 1]][val_mask]
        )
        vel_loss = self.test_vel_loss(
            y_hat[:, :, [2, 3]][val_mask], y_target[:, :, [2, 3]][val_mask]
        )

        # Compute losses on "tracks_to_predict"
        fde_ttp_mask = torch.logical_and(fde_mask, batch.tracks_to_predict)
        fde_ttp_loss = self.test_fde_ttp_loss(
            y_hat[-1, fde_ttp_mask][:, [0, 1]], y_target[-1, fde_ttp_mask][:, [0, 1]]
        )
        ade_ttp_mask = torch.logical_and(
            val_mask,
            batch.tracks_to_predict.expand(
                (self.prediction_horizon - 11, mask.size(0))
            ),
        )
        ade_ttp_loss = self.test_ade_loss(
            y_hat[:, :, [0, 1]][ade_ttp_mask], y_target[:, :, [0, 1]][ade_ttp_mask]
        )

        ######################
        # Logging            #
        ######################

        self.log("test_ade_loss", ade_loss, batch_size=val_mask.sum().item())
        self.log("test_fde_loss", fde_loss, batch_size=fde_mask.sum().item())
        self.log("test_vel_loss", vel_loss, batch_size=val_mask.sum().item())
        loss = ade_loss
        self.log("test_total_loss", loss, batch_size=val_mask.sum().item())
        self.log(
            "test_fde_ttp_loss", fde_ttp_loss, batch_size=fde_ttp_mask.sum().item()
        )
        self.log(
            "test_ade_ttp_loss", ade_ttp_loss, batch_size=ade_ttp_mask.sum().item()
        )

        return loss

    def predict_step(self, batch, batch_idx=None, prediction_horizon: int = 91):

        ######################
        # Initialisation     #
        ######################

        # Determine valid initialisations at t=11
        mask = batch.x[:, :, -1]
        valid_mask = mask[:, 10] > 0

        # Discard non-valid nodes as no initial trajectories will be known
        batch.x = batch.x[valid_mask]
        batch.batch = batch.batch[valid_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[valid_mask]
        batch.type = batch.type[valid_mask]

        # CARS
        type_mask = batch.type[:, 1] == 1
        batch.x = batch.x[type_mask]
        batch.batch = batch.batch[type_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[type_mask]
        batch.type = batch.type[type_mask]

        # Reduction: Limit to x/y
        batch.x = batch.x[:, :, self.node_indices]

        batch.x = batch.x[:, :prediction_horizon]

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((prediction_horizon - 1, n_nodes, self.node_features))
        y_target = torch.zeros((prediction_horizon - 1, n_nodes, self.node_features))
        # Ensure device placement
        y_hat = y_hat.type_as(batch.x)
        y_target = y_target.type_as(batch.x)

        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = batch.x[:, 10, self.out_features :]
        edge_attr = None

        # Initial hidden state
        if self.rnn_type == "GRU":
            h_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            h_node = h_node.type_as(batch.x)
            h_edge = h_edge.type_as(batch.x)
            c_node, c_edge = None, None
        else:
            h_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            h_node = h_node.type_as(batch.x)
            h_edge = h_edge.type_as(batch.x)
            c_node = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            c_edge = torch.zeros(
                (self.model.num_layers, n_nodes, self.model.rnn_edge_size)
            )
            c_node = c_node.type_as(batch.x)
            c_edge = c_edge.type_as(batch.x)

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :]

            # Construct edges
            edge_index = torch_geometric.nn.radius_graph(
                x=x_t[:, :2],
                r=self.min_dist,
                batch=batch.batch[mask_t],
                loop=self.self_loop,
                max_num_neighbors=self.n_neighbours,
                flow="source_to_target",
            )

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(
                edge_index, num_nodes=x_t.shape[0]
            )

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Predictions 1/2    #
            ######################

            # Normalise input graph
            if self.normalise:
                # Center node positions
                x_t[:, self.pos_index] -= batch.loc[batch.batch][mask_t][
                    :, self.pos_index
                ]
                # Scale all features (except yaws) with global scaler
                x_t[:, self.norm_index] /= self.global_scale
                if edge_attr is not None:
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if self.rnn_type == "GRU":
                hidden_in = (h_node[:, mask_t], h_edge[:, mask_t])
                delta_x, h_t = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=hidden_in,
                )
                # Update hidden states
                h_node[:, mask_t] = h_t[0]
                h_edge[:, mask_t] = h_t[1]

            else:  # LSTM
                hidden_in = (
                    (h_node[:, mask_t], c_node[:, mask_t]),
                    (h_edge[:, mask_t], c_edge[:, mask_t]),
                )
                delta_x, (h_node_out, h_edge_out) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=hidden_in,
                )
                h_node[:, mask_t] = h_node_out[0]
                c_node[:, mask_t] = h_node_out[1]
                h_edge[:, mask_t] = h_edge_out[0]
                c_edge[:, mask_t] = h_edge_out[1]

            vel = delta_x[:, self.pos_index]
            pos = batch.x[mask_t, t][:, self.pos_index] + 0.1 * vel
            predicted_graph = torch.cat([pos, vel, static_features[mask_t]], dim=-1)
            predicted_graph = predicted_graph.type_as(batch.x)

            # Save predictions and targets
            y_hat[t, mask_t, :] = predicted_graph
            # y_target[t, mask_t, :] = torch.cat(
            #     [batch.x[mask_t, t + 1, :], batch.type[mask_t]], dim=1
            # )
            y_target[t, mask_t, :] = batch.x[mask_t, t + 1, :]

        ######################
        # Future             #
        ######################

        for t in range(11, (prediction_horizon - 1)):

            ######################
            # Graph construction #
            ######################

            x_t = predicted_graph.clone()

            # Construct edges
            edge_index = torch_geometric.nn.radius_graph(
                x=x_t[:, :2],
                r=self.min_dist,
                batch=batch.batch,
                loop=self.self_loop,
                max_num_neighbors=self.n_neighbours,
                flow="source_to_target",
            )

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(
                edge_index, num_nodes=x_t.shape[0]
            )

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Predictions 2/2    #
            ######################

            # Normalise input graph
            if self.normalise:
                # Center node positions
                x_t[:, self.pos_index] -= batch.loc[batch.batch][:, self.pos_index]
                # Scale all features (except yaws) with global scaler
                x_t[:, self.norm_index] /= self.global_scale
                if edge_attr is not None:
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            else:
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )

            vel = delta_x[:, self.pos_index]
            pos = predicted_graph[:, self.pos_index] + 0.1 * vel
            predicted_graph = torch.cat([pos, vel, static_features], dim=-1)
            predicted_graph = predicted_graph.type_as(batch.x)

            # Save prediction alongside true value (next time step state)
            y_hat[t, :, :] = predicted_graph
            # y_target[t, :, :] = torch.cat([batch.x[:, t + 1, :], batch.type], dim=1)
            y_target[t, :, :] = batch.x[:, t + 1, :]

        return y_hat, y_target, mask

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class ConstantPhysicalBaselineModule(pl.LightningModule):
    def __init__(self, out_features: int = 6, prediction_horizon: int = 91, **kwargs):
        super().__init__()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_yaw_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_fde_ttp_loss = torchmetrics.MeanSquaredError()
        self.val_ade_ttp_loss = torchmetrics.MeanSquaredError()

        self.prediction_horizon = prediction_horizon
        self.out_features = out_features
        self.save_hyperparameters()

    def training_step(self, batch: Batch, batch_idx: int):
        pass

    def validation_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Validate on sequential dataset. First 11 observations are used to prime the model.
        # Loss is computed on remaining 80 samples using rollout.

        # Determine valid initialisations at t=11
        mask = batch.x[:, :, -1]
        valid_mask = mask[:, 10] > 0

        # Discard non-valid nodes as no initial trajectories will be known
        batch.x = batch.x[valid_mask]
        batch.batch = batch.batch[valid_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[valid_mask]
        batch.type = batch.type[valid_mask]

        # CARS
        type_mask = batch.type[:, 1] == 1
        batch.x = batch.x[type_mask]
        batch.batch = batch.batch[type_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[type_mask]
        batch.type = batch.type[type_mask]

        # Update input using prediction horizon
        batch.x = batch.x[:, : self.prediction_horizon]

        # Limit to x, y, x_vel, y_vel
        batch.x = batch.x[:, :, [0, 1, 3, 4, 10]]

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((self.prediction_horizon - 11, n_nodes, self.out_features))
        y_target = torch.zeros(
            (self.prediction_horizon - 11, n_nodes, self.out_features)
        )
        # Remove valid flag from features
        batch.x = batch.x[:, :, :-1]
        # Find valid agents at time t=11
        initial_mask = mask[:, 10]
        # Extract final dynamic states to use for predictions
        last_pos = batch.x[initial_mask, 10][:, [0, 1]]
        last_vel = batch.x[initial_mask, 10][:, [2, 3]]
        # Constant change in positions
        delta_pos = last_vel * 0.1
        # First updated position
        predicted_pos = last_pos + delta_pos
        predicted_graph = torch.cat([predicted_pos, last_vel], dim=1)
        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph[:, : self.out_features]
        y_target[0, :, :] = batch.x[:, 11, : self.out_features]

        for t in range(11, self.prediction_horizon - 1):
            predicted_pos += delta_pos
            predicted_graph = torch.cat([predicted_pos, last_vel], dim=1)
            y_hat[t - 10, :, :] = predicted_graph[:, : self.out_features]
            y_target[t - 10, :, :] = batch.x[:, t + 1, : self.out_features]

        # Extract loss mask
        fde_mask = mask[:, -1]
        val_mask = mask[:, 11:].permute(1, 0)

        # Compute and log loss
        fde_loss = self.val_fde_loss(
            y_hat[-1, fde_mask][:, [0, 1]], y_target[-1, fde_mask][:, [0, 1]]
        )
        ade_loss = self.val_ade_loss(
            y_hat[:, :, [0, 1]][val_mask], y_target[:, :, [0, 1]][val_mask]
        )
        vel_loss = self.val_vel_loss(
            y_hat[:, :, [2, 3]][val_mask], y_target[:, :, [2, 3]][val_mask]
        )

        # Compute losses on "tracks_to_predict"
        fde_ttp_mask = torch.logical_and(fde_mask, batch.tracks_to_predict)
        fde_ttp_loss = self.val_fde_ttp_loss(
            y_hat[-1, fde_ttp_mask][:, [0, 1]], y_target[-1, fde_ttp_mask][:, [0, 1]]
        )
        ade_ttp_mask = torch.logical_and(
            val_mask,
            batch.tracks_to_predict.expand(
                (self.prediction_horizon - 11, mask.size(0))
            ),
        )
        ade_ttp_loss = self.val_ade_loss(
            y_hat[:, :, [0, 1]][ade_ttp_mask], y_target[:, :, [0, 1]][ade_ttp_mask]
        )

        ######################
        # Logging            #
        ######################

        self.log("val_ade_loss", ade_loss)
        self.log("val_fde_loss", fde_loss)
        self.log("val_vel_loss", vel_loss)
        loss = ade_loss
        self.log("val_total_loss", loss)
        self.log("val_fde_ttp_loss", fde_ttp_loss)
        self.log("val_ade_ttp_loss", ade_ttp_loss)

        return loss

    def predict_step(self, batch, batch_idx=None):

        ######################
        # Initialisation     #
        ######################

        # Determine valid initialisations at t=11
        mask = batch.x[:, :, -1]
        valid_mask = mask[:, 10] > 0

        # Discard non-valid nodes as no initial trajectories will be known
        batch.x = batch.x[valid_mask]
        batch.batch = batch.batch[valid_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[valid_mask]
        batch.type = batch.type[valid_mask]

        # CARS
        type_mask = batch.type[:, 1] == 1
        batch.x = batch.x[type_mask]
        batch.batch = batch.batch[type_mask]
        batch.tracks_to_predict = batch.tracks_to_predict[type_mask]
        batch.type = batch.type[type_mask]

        # Update input using prediction horizon
        batch.x = batch.x[:, : self.prediction_horizon]

        # Limit to x, y, x_vel, y_vel
        batch.x = batch.x[:, :, [0, 1, 3, 4, 10]]

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((self.prediction_horizon - 1, n_nodes, 4))
        # Remove valid flag from features
        batch.x = batch.x[:, :, :-1]

        # Fill in targets
        y_target = batch.x[:, 1:]
        y_target = y_target.permute(1, 0, 2)

        for t in range(11):
            mask_t = mask[:, t]

            last_pos = batch.x[mask_t, t][:, [0, 1]]
            last_vel = batch.x[mask_t, t][:, [2, 3]]

            delta_pos = last_vel * 0.1
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat([predicted_pos, last_vel], dim=-1)
            y_hat[t, mask_t, :] = predicted_graph

        for t in range(11, 90):
            last_pos = predicted_pos
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat([predicted_pos, last_vel], dim=-1)
            y_hat[t, :, :] = predicted_graph

        return y_hat, y_target, mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


@hydra.main(config_path="../../../configs/waymo/", config_name="config")
def main(config):
    # Print configuration for online monitoring
    print(OmegaConf.to_yaml(config))
    # Save complete yaml file for logging and reproducibility
    log_dir = f"logs/{config.logger.project}/{config.logger.version}"
    os.makedirs(log_dir, exist_ok=True)
    yaml_path = f"{log_dir}/{config.logger.version}.yaml"
    OmegaConf.save(config, f=yaml_path)

    # Seed for reproducibility
    seed_everything(config["misc"]["seed"], workers=True)
    # Load data, model, and regressor
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])
    # Define model
    if config["misc"]["model_type"] != "ConstantModel":
        model_dict = config["model"]
        model_type = config["misc"]["model_type"]
    else:
        model_dict, model_type = None, None

    # Define LightningModule
    regressor = eval(config["misc"]["regressor_type"])(
        model_type=model_type, model_dict=dict(model_dict), **config["regressor"]
    )

    # Setup logging (using saved yaml file)
    wandb_logger = WandbLogger(
        entity="petergroth",
        config=OmegaConf.to_container(config, resolve=True),
        **config["logger"],
    )
    wandb_logger.watch(regressor, log_freq=config["misc"]["log_freq"], log_graph=False)
    # Add default dir for logs

    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=config["logger"]["version"], monitor="val_total_loss", save_last=True
    )
    # Create trainer, fit, and validate
    trainer = pl.Trainer(
        logger=wandb_logger, **config["trainer"], callbacks=[checkpoint_callback]
    )

    if config["misc"]["train"]:
        trainer.fit(model=regressor, datamodule=datamodule)

    trainer.validate(regressor, datamodule=datamodule)


if __name__ == "__main__":
    main()
