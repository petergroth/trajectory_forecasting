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
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch_geometric.data import Batch

from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule
from src.models.model import *
from src.training_modules import *


class SequentialModule(pl.LightningModule):
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
        local_map_resolution: int = 40,
        map_channels: int = 8,
        n_components: int = 2,
    ):
        super().__init__()
        # Set up metrics
        self.train_ade_loss = torchmetrics.MeanSquaredError()
        self.train_fde_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_fde_ttp_loss = torchmetrics.MeanSquaredError()
        self.val_ade_ttp_loss = torchmetrics.MeanSquaredError()

        # Instantiate map encoder
        self.map_encoder = road_encoder(
            width=local_map_resolution + 1,
            hidden_size=model_dict["map_encoding_size"],
            in_map_channels=map_channels,
        )
        self.local_map_resolution = local_map_resolution
        self.local_map_resolution_half = int(local_map_resolution / 2)

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
        self.n_components = n_components
        self.out_features = out_features
        self.edge_features = edge_features
        self.node_features = node_features

        # Instantiate models
        self.model_type = model_type
        model_dict["out_features"] = (
            2 + self.n_components * 4 if self.n_components > 1 else 5
        )
        self.model = eval(model_type)(**model_dict)

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
        batch.u[batch.u > 1] = 1

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
        static_features = batch.x[:, 10, 4:]
        static_features = static_features.type_as(batch.x)
        edge_attr, edge_attr_k = None, None
        n_nodes = batch.num_nodes

        y_predictions = torch.zeros(
            (n_nodes, self.n_components, self.training_horizon, 4)
        )
        y_predictions = y_predictions.type_as(batch.x)

        # Tensor of position and velocity targets
        y_target = batch.x[:, 1 : (self.training_horizon + 1), :4]
        y_target = y_target.type_as(batch.x)

        likelihoods = torch.zeros((n_nodes, self.n_components, self.training_horizon))
        Sigma_pos = (
            torch.eye(2).reshape(1, 2, 2).repeat(n_nodes, self.n_components, 1, 1)
        )
        Sigma_vel = (
            torch.eye(2).reshape(1, 2, 2).repeat(n_nodes, self.n_components, 1, 1)
        )
        likelihoods = likelihoods.type_as(batch.x)
        Sigma_pos = Sigma_pos.type_as(batch.x)
        Sigma_vel = Sigma_vel.type_as(batch.x)

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
        # Map preparation    #
        ######################

        # Zero pad each map for edge-cases
        batch.u = nn.functional.pad(
            batch.u,
            (
                self.local_map_resolution,
                self.local_map_resolution,
                self.local_map_resolution,
                self.local_map_resolution,
            ),
        )
        # Extract centre locations of each scene
        center_values = batch.loc[:, :2]
        # Compute x/y-ranges
        map_ranges = torch.vstack(
            [
                center_values[:, 0] - 150 / 2,
                center_values[:, 0] + 150 / 2,
                center_values[:, 1] - 150 / 2,
                center_values[:, 1] + 150 / 2,
            ]
        ).T
        # Allocate and compute all values in x/y-ranges for localisation
        interval_x = torch.zeros((batch.num_graphs, 150 * 2 + 1))
        interval_y = torch.zeros((batch.num_graphs, 150 * 2 + 1))
        interval_x = interval_x.type_as(batch.x)
        interval_y = interval_y.type_as(batch.x)
        for i in range(batch.num_graphs):
            interval_x[i] = torch.linspace(
                map_ranges[i, 0], map_ranges[i, 1], 150 * 2 + 1
            )
            interval_y[i] = torch.linspace(
                map_ranges[i, 2], map_ranges[i, 3], 150 * 2 + 1
            )

        ######################
        # History            #
        ######################

        for t in range(11):

            # Extract current input
            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :].unsqueeze(1).expand(-1, self.n_components, -1)
            x_t = x_t.type_as(batch.x)

            ######################
            # Graph construction #
            ######################

            # Construct edges
            edge_index = []
            edge_attr = []

            for k in range(self.n_components):
                edge_index_k = torch_geometric.nn.radius_graph(
                    x=x_t[:, k, :2],
                    r=self.min_dist,
                    batch=batch.batch[mask_t],
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

                # Remove duplicates and sort
                edge_index_k = torch_geometric.utils.coalesce(
                    edge_index_k, num_nodes=x_t.shape[0]
                )

                # Create edge_attr if specified
                if self.edge_weight:
                    # Encode distance between nodes as edge_attr
                    row, col = edge_index_k
                    edge_attr_k = (
                        (x_t[row, k, :2] - x_t[col, k, :2]).norm(dim=-1).unsqueeze(1)
                    )
                    edge_attr_k = edge_attr_k.type_as(batch.x)

                if self.edge_dropout > 0:
                    edge_index_k, edge_attr_k = dropout_adj(
                        edge_index=edge_index_k,
                        edge_attr=edge_attr_k,
                        p=self.edge_dropout,
                    )

                edge_index.append(edge_index_k)
                edge_attr.append(edge_attr_k)

            #######################
            # Map encoding 1/2    #
            #######################

            # Allocate local maps
            u_local = torch.zeros(
                (
                    x_t.size(0),
                    batch.u.size(1),
                    self.local_map_resolution + 1,
                    self.local_map_resolution + 1,
                )
            )
            u_local = u_local.type_as(batch.x)

            # Find closest pixels in x and y directions
            center_pixel_x = (
                torch.argmax(
                    (interval_x[batch.batch[mask_t]] > x_t[:, 0].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 1
            )
            center_pixel_y = (
                torch.argmax(
                    (interval_y[batch.batch[mask_t]] > x_t[:, 1].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 2
            )

            # Compute pixel boundaries
            idx_x_low = center_pixel_y + self.local_map_resolution_half
            idx_x_high = (
                center_pixel_y
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )
            idx_y_low = center_pixel_x + self.local_map_resolution_half
            idx_y_high = (
                center_pixel_x
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )

            # Extract local maps for all agents in current time-step
            for node_idx, graph_idx in enumerate(batch.batch[mask_t]):
                if not (
                    center_pixel_x[node_idx] == -1 or center_pixel_y[node_idx] == -2
                ):
                    u_local[node_idx] = batch.u[
                        graph_idx,
                        :,
                        idx_x_low[node_idx] : idx_x_high[node_idx],
                        idx_y_low[node_idx] : idx_y_high[node_idx],
                    ]

            # Perform map encoding
            u = self.map_encoder(u_local)

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
                    u=u,
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
                    u=u,
                )
                h_node[:, mask_t] = h_node_out[0]
                c_node[:, mask_t] = h_node_out[1]
                h_edge[:, mask_t] = h_edge_out[0]
                c_edge[:, mask_t] = h_edge_out[1]

            vel = delta_x[:, [0, 1]]
            pos = batch.x[mask_t, t][:, self.pos_index] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features[mask_t]], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Update current velocity covariance matrix
            sigma_x = torch.nn.functional.softplus(delta_x[:, 2])
            sigma_y = torch.nn.functional.softplus(delta_x[:, 3])
            cov_xy = torch.tanh(delta_x[:, 4]) * sigma_x * sigma_y
            Sigma_vel[mask_t, 0, 0] = sigma_x ** 2
            Sigma_vel[mask_t, 1, 1] = sigma_y ** 2
            Sigma_vel[mask_t, 1, 0] = cov_xy
            Sigma_vel[mask_t, 0, 1] = cov_xy

            # Compute likelihood of current estimates
            Sigma_pos[mask_t] += 0.01 * Sigma_vel[mask_t]

            likelihoods[mask_t, t] = torch.distributions.MultivariateNormal(
                loc=pos, covariance_matrix=Sigma_pos[mask_t]
            ).log_prob(y_target[mask_t, t, :2])

            # Save deltas for loss computation
            y_predictions[mask_t, t, :2] = pos
            y_predictions[mask_t, t, 2:] = delta_x

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
            # Map encoding 2/2    #
            #######################

            # Allocate local maps
            u_local = torch.zeros(
                (
                    x_t.size(0),
                    batch.u.size(1),
                    self.local_map_resolution + 1,
                    self.local_map_resolution + 1,
                )
            )
            u_local = u_local.type_as(batch.x)

            # Find closest pixels in x and y directions
            center_pixel_x = (
                torch.argmax(
                    (interval_x[batch.batch] > x_t[:, 0].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 1
            )
            center_pixel_y = (
                torch.argmax(
                    (interval_y[batch.batch] > x_t[:, 1].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 2
            )

            # Compute pixel boundaries
            idx_x_low = center_pixel_y + self.local_map_resolution_half
            idx_x_high = (
                center_pixel_y
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )
            idx_y_low = center_pixel_x + self.local_map_resolution_half
            idx_y_high = (
                center_pixel_x
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )

            # Extract local maps for all agents in current time-step
            for node_idx, graph_idx in enumerate(batch.batch):
                if not (
                    center_pixel_x[node_idx] == -1 or center_pixel_y[node_idx] == -2
                ):
                    u_local[node_idx] = batch.u[
                        graph_idx,
                        :,
                        idx_x_low[node_idx] : idx_x_high[node_idx],
                        idx_y_low[node_idx] : idx_y_high[node_idx],
                    ]

            # Perform map encoding
            u = self.map_encoder(u_local)

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
                    u=u,
                )
            else:
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                    u=u,
                )

            vel = delta_x[:, [0, 1]]
            pos = x_prev[:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Update current velocity covariance matrix
            sigma_x = torch.nn.functional.softplus(delta_x[:, 2])
            sigma_y = torch.nn.functional.softplus(delta_x[:, 3])
            cov_xy = torch.tanh(delta_x[:, 4]) * sigma_x * sigma_y
            Sigma_vel[:, 0, 0] = sigma_x ** 2
            Sigma_vel[:, 1, 1] = sigma_y ** 2
            Sigma_vel[:, 1, 0] = cov_xy
            Sigma_vel[:, 0, 1] = cov_xy

            # Compute likelihood of current estimates
            Sigma_pos += 0.01 * Sigma_vel

            likelihoods[:, t] = torch.distributions.MultivariateNormal(
                loc=pos, covariance_matrix=Sigma_pos
            ).log_prob(y_target[:, t, :2])

            # Save deltas for loss computation
            y_predictions[:, t, :2] = pos
            y_predictions[:, t, 2:] = delta_x

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

        # Compute likelihoods of all agents at all times
        mean_nllh = torch.mean(-likelihoods[loss_mask])
        self.log(
            "train_mean_nllh",
            mean_nllh,
            on_step=True,
            on_epoch=True,
            batch_size=loss_mask.sum().item(),
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

        loss = mean_nllh

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
        batch.u[batch.u > 1] = 1

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

        # Allocate prediction tensors
        n_nodes = batch.num_nodes

        y_predictions = torch.zeros(
            (n_nodes, self.prediction_horizon - 11, self.out_features)
        )
        y_predictions = y_predictions.type_as(batch.x)

        # Tensor of position and velocity targets
        y_target = batch.x[:, 11:, :4]
        y_target = y_target.type_as(batch.x)

        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = batch.x[:, 10, 4:]
        static_features = static_features.type_as(batch.x)
        edge_attr = None

        # Allocate likelihood tensor and covariance matrices
        likelihoods = torch.zeros((n_nodes, self.prediction_horizon - 11))
        Sigma_pos = torch.eye(2).reshape(1, 2, 2).repeat(n_nodes, 1, 1)
        Sigma_vel = torch.eye(2).reshape(1, 2, 2).repeat(n_nodes, 1, 1)
        likelihoods = likelihoods.type_as(batch.x)
        Sigma_pos = Sigma_pos.type_as(batch.x)
        Sigma_vel = Sigma_vel.type_as(batch.x)

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
        # Map preparation    #
        ######################

        # Zero pad each map for edge-cases
        batch.u = nn.functional.pad(
            batch.u,
            (
                self.local_map_resolution,
                self.local_map_resolution,
                self.local_map_resolution,
                self.local_map_resolution,
            ),
        )
        # Extract centre locations of each scene
        center_values = batch.loc[:, :2]
        # Compute x/y-ranges
        map_ranges = torch.vstack(
            [
                center_values[:, 0] - 150 / 2,
                center_values[:, 0] + 150 / 2,
                center_values[:, 1] - 150 / 2,
                center_values[:, 1] + 150 / 2,
            ]
        ).T
        # Allocate and compute all values in x/y-ranges for localisation
        interval_x = torch.zeros((batch.num_graphs, 150 * 2 + 1))
        interval_y = torch.zeros((batch.num_graphs, 150 * 2 + 1))
        interval_x = interval_x.type_as(batch.x)
        interval_y = interval_y.type_as(batch.x)
        for i in range(batch.num_graphs):
            interval_x[i] = torch.linspace(
                map_ranges[i, 0], map_ranges[i, 1], 150 * 2 + 1
            )
            interval_y[i] = torch.linspace(
                map_ranges[i, 2], map_ranges[i, 3], 150 * 2 + 1
            )

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

            #######################
            # Map encoding 1/2    #
            #######################

            # Allocate local maps
            u_local = torch.zeros(
                (
                    x_t.size(0),
                    batch.u.size(1),
                    self.local_map_resolution + 1,
                    self.local_map_resolution + 1,
                )
            )
            u_local = u_local.type_as(batch.x)

            # Find closest pixels in x and y directions
            center_pixel_x = (
                torch.argmax(
                    (interval_x[batch.batch[mask_t]] > x_t[:, 0].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 1
            )
            center_pixel_y = (
                torch.argmax(
                    (interval_y[batch.batch[mask_t]] > x_t[:, 1].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 2
            )

            # Compute pixel boundaries
            idx_x_low = center_pixel_y + self.local_map_resolution_half
            idx_x_high = (
                center_pixel_y
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )
            idx_y_low = center_pixel_x + self.local_map_resolution_half
            idx_y_high = (
                center_pixel_x
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )

            # Extract local maps for all agents in current time-step
            for node_idx, graph_idx in enumerate(batch.batch[mask_t]):
                if not (
                    center_pixel_x[node_idx] == -1 or center_pixel_y[node_idx] == -2
                ):
                    u_local[node_idx] = batch.u[
                        graph_idx,
                        :,
                        idx_x_low[node_idx] : idx_x_high[node_idx],
                        idx_y_low[node_idx] : idx_y_high[node_idx],
                    ]

            # Perform map encoding
            u = self.map_encoder(u_local)

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
                    u=u,
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
                    u=u,
                )
                h_node[:, mask_t] = h_node_out[0]
                c_node[:, mask_t] = h_node_out[1]
                h_edge[:, mask_t] = h_edge_out[0]
                c_edge[:, mask_t] = h_edge_out[1]

            # Update current velocity covariance matrix
            sigma_x = torch.nn.functional.softplus(delta_x[:, 2])
            sigma_y = torch.nn.functional.softplus(delta_x[:, 3])
            cov_xy = torch.tanh(delta_x[:, 4]) * sigma_x * sigma_y
            Sigma_vel[mask_t, 0, 0] = sigma_x ** 2
            Sigma_vel[mask_t, 1, 1] = sigma_y ** 2
            Sigma_vel[mask_t, 1, 0] = cov_xy
            Sigma_vel[mask_t, 0, 1] = cov_xy

            # Compute likelihood of current estimates
            Sigma_pos[mask_t] += 0.01 * Sigma_vel[mask_t]

            if t == 10:
                vel = delta_x[:, [0, 1]]
                pos = batch.x[mask_t, t][:, self.pos_index] + 0.1 * vel
                predicted_graph = torch.cat([pos, vel, static_features[mask_t]], dim=-1)
                predicted_graph = predicted_graph.type_as(batch.x)

                likelihoods[:, 0] = torch.distributions.MultivariateNormal(
                    loc=pos, covariance_matrix=Sigma_pos
                ).log_prob(y_target[:, 0, :2])
                y_predictions[mask_t, 0, :2] = pos
                y_predictions[mask_t, 0, 2:] = delta_x

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

            #######################
            # Map encoding 2/2    #
            #######################

            # Allocate local maps
            u_local = torch.zeros(
                (
                    x_t.size(0),
                    batch.u.size(1),
                    self.local_map_resolution + 1,
                    self.local_map_resolution + 1,
                )
            )
            u_local = u_local.type_as(batch.x)

            # Find closest pixels in x and y directions
            center_pixel_x = (
                torch.argmax(
                    (interval_x[batch.batch] > x_t[:, 0].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 1
            )
            center_pixel_y = (
                torch.argmax(
                    (interval_y[batch.batch] > x_t[:, 1].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 2
            )

            # Compute pixel boundaries
            idx_x_low = center_pixel_y + self.local_map_resolution_half
            idx_x_high = (
                center_pixel_y
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )
            idx_y_low = center_pixel_x + self.local_map_resolution_half
            idx_y_high = (
                center_pixel_x
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )

            # Extract local maps for all agents in current time-step
            for node_idx, graph_idx in enumerate(batch.batch):
                if not (
                    center_pixel_x[node_idx] == -1 or center_pixel_y[node_idx] == -2
                ):
                    u_local[node_idx] = batch.u[
                        graph_idx,
                        :,
                        idx_x_low[node_idx] : idx_x_high[node_idx],
                        idx_y_low[node_idx] : idx_y_high[node_idx],
                    ]

            # Perform map encoding
            u = self.map_encoder(u_local)

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
                    u=u,
                )
            else:
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                    u=u,
                )

            vel = delta_x[:, [0, 1]]
            pos = predicted_graph[:, [0, 1]] + 0.1 * vel
            predicted_graph = torch.cat([pos, vel, static_features], dim=-1)
            predicted_graph = predicted_graph.type_as(batch.x)

            # Update current velocity covariance matrix
            sigma_x = torch.nn.functional.softplus(delta_x[:, 2])
            sigma_y = torch.nn.functional.softplus(delta_x[:, 3])
            cov_xy = torch.tanh(delta_x[:, 4]) * sigma_x * sigma_y
            Sigma_vel[:, 0, 0] = sigma_x ** 2
            Sigma_vel[:, 1, 1] = sigma_y ** 2
            Sigma_vel[:, 1, 0] = cov_xy
            Sigma_vel[mask_t, 0, 1] = cov_xy

            # Compute likelihood of current estimates
            Sigma_pos += 0.01 * Sigma_vel

            likelihoods[:, t - 10] = torch.distributions.MultivariateNormal(
                loc=pos, covariance_matrix=Sigma_pos
            ).log_prob(y_target[:, t - 10, :2])

            # Save prediction alongside true value (next time step state)
            y_predictions[:, t - 10, :2] = pos
            y_predictions[:, t - 10, 2:] = delta_x

        fde_mask = mask[:, -1]
        val_mask = mask[:, 11:]

        # Compute and log loss
        fde_loss = self.val_fde_loss(
            y_predictions[fde_mask, -1][:, [0, 1]], y_target[fde_mask, -1][:, [0, 1]]
        )
        ade_loss = self.val_ade_loss(
            y_predictions[:, :, [0, 1]][val_mask], y_target[:, :, [0, 1]][val_mask]
        )
        vel_loss = self.val_vel_loss(
            y_predictions[:, :, [2, 3]][val_mask], y_target[:, :, [2, 3]][val_mask]
        )

        # Compute losses on "tracks_to_predict"
        fde_ttp_mask = torch.logical_and(fde_mask, batch.tracks_to_predict)
        fde_ttp_loss = self.val_fde_ttp_loss(
            y_predictions[fde_ttp_mask, -1][:, [0, 1]],
            y_target[fde_ttp_mask, -1][:, [0, 1]],
        )
        ade_ttp_mask = torch.logical_and(
            val_mask,
            batch.tracks_to_predict.unsqueeze(1).expand(val_mask.shape),
        )
        ade_ttp_loss = self.val_ade_loss(
            y_predictions[:, :, [0, 1]][ade_ttp_mask],
            y_target[:, :, [0, 1]][ade_ttp_mask],
        )

        # Compute likelihoods of all agents at all times
        mean_nllh = torch.mean(-likelihoods[val_mask])
        self.log(
            "val_mean_nllh",
            mean_nllh,
            batch_size=val_mask.sum().item(),
        )

        ######################
        # Logging            #
        ######################

        self.log("val_ade_loss", ade_loss, batch_size=val_mask.sum().item())
        self.log("val_fde_loss", fde_loss, batch_size=fde_mask.sum().item())
        self.log("val_vel_loss", vel_loss, batch_size=val_mask.sum().item())
        loss = mean_nllh
        self.log("val_total_loss", loss, batch_size=val_mask.sum().item())
        self.log("val_fde_ttp_loss", fde_ttp_loss, batch_size=fde_ttp_mask.sum().item())
        self.log("val_ade_ttp_loss", ade_ttp_loss, batch_size=ade_ttp_mask.sum().item())

        return loss

    def predict_step(self, batch, batch_idx=None, prediction_horizon: int = 91):

        ######################
        # Initialisation     #
        ######################

        # Determine valid initialisations at t=11
        mask = batch.x[:, :, -1]
        valid_mask = mask[:, 10] > 0
        batch.u[batch.u > 1] = 1
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

        batch.x = batch.x[:, :prediction_horizon]

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((prediction_horizon - 1, n_nodes, 7))
        y_target = torch.zeros((prediction_horizon - 1, n_nodes, 7))
        # Ensure device placement
        y_hat = y_hat.type_as(batch.x)
        y_target = y_target.type_as(batch.x)

        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = batch.x[:, 10, 4:]
        edge_attr = None

        # Allocate likelihood tensor and covariance matrices
        # likelihoods = torch.zeros((n_nodes, self.training_horizon))
        Sigma_vel = torch.eye(2).reshape(1, 2, 2).repeat(n_nodes, 1, 1)
        Sigma_pos = (
            torch.eye(2).reshape(1, 1, 2, 2).repeat(prediction_horizon, n_nodes, 1, 1)
        )
        Sigma_pos = Sigma_pos.type_as(batch.x)
        Sigma_vel = Sigma_vel.type_as(batch.x)

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
        # Map preparation    #
        ######################

        # Zero pad each map for edge-cases
        batch.u = nn.functional.pad(
            batch.u,
            (
                self.local_map_resolution,
                self.local_map_resolution,
                self.local_map_resolution,
                self.local_map_resolution,
            ),
        )
        # Extract centre locations of each scene
        center_values = batch.loc[:, :2]
        # Compute x/y-ranges
        map_ranges = torch.vstack(
            [
                center_values[:, 0] - 150 / 2,
                center_values[:, 0] + 150 / 2,
                center_values[:, 1] - 150 / 2,
                center_values[:, 1] + 150 / 2,
            ]
        ).T
        # Allocate and compute all values in x/y-ranges for localisation
        interval_x = torch.zeros((batch.num_graphs, 150 * 2 + 1))
        interval_y = torch.zeros((batch.num_graphs, 150 * 2 + 1))
        interval_x = interval_x.type_as(batch.x)
        interval_y = interval_y.type_as(batch.x)
        for i in range(batch.num_graphs):
            interval_x[i] = torch.linspace(
                map_ranges[i, 0], map_ranges[i, 1], 150 * 2 + 1
            )
            interval_y[i] = torch.linspace(
                map_ranges[i, 2], map_ranges[i, 3], 150 * 2 + 1
            )

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

            #######################
            # Map encoding 1/2    #
            #######################

            # Allocate local maps
            u_local = torch.zeros(
                (
                    x_t.size(0),
                    batch.u.size(1),
                    self.local_map_resolution + 1,
                    self.local_map_resolution + 1,
                )
            )
            u_local = u_local.type_as(batch.x)

            # Find closest pixels in x and y directions
            center_pixel_x = (
                torch.argmax(
                    (interval_x[batch.batch[mask_t]] > x_t[:, 0].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 1
            )
            center_pixel_y = (
                torch.argmax(
                    (interval_y[batch.batch[mask_t]] > x_t[:, 1].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 2
            )

            # Compute pixel boundaries
            idx_x_low = center_pixel_y + self.local_map_resolution_half
            idx_x_high = (
                center_pixel_y
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )
            idx_y_low = center_pixel_x + self.local_map_resolution_half
            idx_y_high = (
                center_pixel_x
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )

            # Extract local maps for all agents in current time-step
            for node_idx, graph_idx in enumerate(batch.batch[mask_t]):
                if not (
                    center_pixel_x[node_idx] == -1 or center_pixel_y[node_idx] == -2
                ):
                    u_local[node_idx] = batch.u[
                        graph_idx,
                        :,
                        idx_x_low[node_idx] : idx_x_high[node_idx],
                        idx_y_low[node_idx] : idx_y_high[node_idx],
                    ]

            # Perform map encoding
            u = self.map_encoder(u_local)

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
                    u=u,
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
                    u=u,
                )
                h_node[:, mask_t] = h_node_out[0]
                c_node[:, mask_t] = h_node_out[1]
                h_edge[:, mask_t] = h_edge_out[0]
                c_edge[:, mask_t] = h_edge_out[1]

            vel = delta_x[:, self.pos_index]
            pos = batch.x[mask_t, t][:, self.pos_index] + 0.1 * vel
            predicted_graph = torch.cat([pos, vel, static_features[mask_t]], dim=-1)
            predicted_graph = predicted_graph.type_as(batch.x)

            # Update current velocity covariance matrix
            sigma_x = torch.nn.functional.softplus(delta_x[:, 2])
            sigma_y = torch.nn.functional.softplus(delta_x[:, 3])
            cov_xy = torch.tanh(delta_x[:, 4]) * sigma_x * sigma_y
            Sigma_vel[mask_t, 0, 0] = sigma_x ** 2
            Sigma_vel[mask_t, 1, 1] = sigma_y ** 2
            Sigma_vel[mask_t, 1, 0] = cov_xy
            Sigma_vel[mask_t, 0, 1] = cov_xy

            # Compute likelihood of current estimates
            Sigma_pos[t, mask_t] += 0.01 * Sigma_vel[mask_t]

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

            #######################
            # Map encoding 2/2    #
            #######################

            # Allocate local maps
            u_local = torch.zeros(
                (
                    x_t.size(0),
                    batch.u.size(1),
                    self.local_map_resolution + 1,
                    self.local_map_resolution + 1,
                )
            )
            u_local = u_local.type_as(batch.x)

            # Find closest pixels in x and y directions
            center_pixel_x = (
                torch.argmax(
                    (interval_x[batch.batch] > x_t[:, 0].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 1
            )
            center_pixel_y = (
                torch.argmax(
                    (interval_y[batch.batch] > x_t[:, 1].unsqueeze(-1)).float(),
                    dim=1,
                ).type(torch.LongTensor)
                - 2
            )

            # Compute pixel boundaries
            idx_x_low = center_pixel_y + self.local_map_resolution_half
            idx_x_high = (
                center_pixel_y
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )
            idx_y_low = center_pixel_x + self.local_map_resolution_half
            idx_y_high = (
                center_pixel_x
                + self.local_map_resolution_half
                + self.local_map_resolution
                + 1
            )

            # Extract local maps for all agents in current time-step
            for node_idx, graph_idx in enumerate(batch.batch):
                if not (
                    center_pixel_x[node_idx] == -1 or center_pixel_y[node_idx] == -2
                ):
                    u_local[node_idx] = batch.u[
                        graph_idx,
                        :,
                        idx_x_low[node_idx] : idx_x_high[node_idx],
                        idx_y_low[node_idx] : idx_y_high[node_idx],
                    ]

            # Perform map encoding
            u = self.map_encoder(u_local)

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
                    u=u,
                )
            else:
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                    u=u,
                )

            vel = delta_x[:, self.pos_index]
            pos = predicted_graph[:, self.pos_index] + 0.1 * vel
            predicted_graph = torch.cat([pos, vel, static_features], dim=-1)
            predicted_graph = predicted_graph.type_as(batch.x)

            # Update current velocity covariance matrix
            sigma_x = torch.nn.functional.softplus(delta_x[:, 2])
            sigma_y = torch.nn.functional.softplus(delta_x[:, 3])
            cov_xy = torch.tanh(delta_x[:, 4]) * sigma_x * sigma_y
            Sigma_vel[:, 0, 0] = sigma_x ** 2
            Sigma_vel[:, 1, 1] = sigma_y ** 2
            Sigma_vel[:, 1, 0] = cov_xy
            Sigma_vel[:, 0, 1] = cov_xy

            # Compute likelihood of current estimates
            Sigma_pos[t] += 0.01 * Sigma_vel

            # Save prediction alongside true value (next time step state)
            y_hat[t, :, :] = predicted_graph
            # y_target[t, :, :] = torch.cat([batch.x[:, t + 1, :], batch.type], dim=1)
            y_target[t, :, :] = batch.x[:, t + 1, :]

        return y_hat, y_target, mask, Sigma_pos

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
    wandb_logger.watch(regressor, log_freq=config["misc"]["log_freq"], log_graph=True)
    # Add default dir for logs

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

    if config["misc"]["train"]:
        trainer.fit(model=regressor, datamodule=datamodule)

    trainer.validate(regressor, datamodule=datamodule)


if __name__ == "__main__":
    main()
