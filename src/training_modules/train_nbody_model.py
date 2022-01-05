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
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch_geometric.data import Batch, Data

from src.data.dataset_nbody import (OneStepNBodyDataModule,
                                    SequentialNBodyDataModule)
from src.models.model import *


class OneStepModule(pl.LightningModule):
    def __init__(
        self,
        model_type: Union[None, str],
        model_dict: Union[None, dict],
        noise: Union[None, float] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        min_dist: int = 0,
        edge_weight: bool = False,
        self_loop: bool = True,
        undirected: bool = False,
        grav_attraction: bool = False,
        node_features: int = 5,
        edge_features: int = 0,
        out_features: int = 4,
        normalise: bool = True,
    ):
        super().__init__()

        # Instantiate model
        self.model_type = model_type
        self.model = eval(model_type)(**model_dict)

        # Setup metrics
        self.train_pos_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        self.train_yaw_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_yaw_loss = torchmetrics.MeanSquaredError()

        self.model = model_type
        self.save_hyperparameters()
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_dist = min_dist
        self.edge_weight = edge_weight
        self.self_loop = self_loop
        self.undirected = undirected
        self.grav_attraction = grav_attraction
        self.normalise = normalise

    def training_step(self, batch: Batch, batch_idx: int):
        # Extract node features and edge_index
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = None

        ######################
        # Graph construction #
        ######################

        # Determine whether to compute edge_index as a function of distances
        if self.min_dist is not None:
            edge_index = torch_geometric.nn.radius_graph(
                x=x[:, :2],
                r=self.min_dist,
                batch=batch.batch,
                loop=self.self_loop,
                max_num_neighbors=30,
                flow="source_to_target",
            )
            # 1 nearest neighbour to ensure connected graphs
            nn_edge_index = torch_geometric.nn.knn_graph(
                x=x[:, :2], k=1, batch=batch.batch
            )
            # Remove duplicates
            edge_index = torch_geometric.utils.coalesce(
                torch.cat((edge_index, nn_edge_index), dim=1)
            )
            self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

        # Determine whether to add random noise to features
        if self.noise is not None:
            x += self.noise * torch.randn_like(x)
            # # Add noise to dynamic states
            # x_d = x[:, :4] + self.noise*torch.randn_like(x[:, :4])
            # # Concatenate dynamic and static states
            # x = torch.cat((x_d, x[:, 4:]), dim=1)

        if self.edge_weight:
            # Encode distance between nodes as edge_attr
            row, col = edge_index
            edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
            if self.grav_attraction:
                # Compute gravitational attraction between all nodes
                m1 = x[row, 4] * 1e10
                m2 = x[col, 4] * 1e10
                attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                # Replace inf values with 0
                edge_attr = torch.nan_to_num(edge_attr, posinf=0)
            edge_attr = edge_attr.type_as(batch.x)

        if self.undirected:
            edge_index, edge_attr = torch_geometric.utils.to_undirected(
                edge_index, edge_attr
            )

        ######################
        # Training 1/1       #
        ######################

        # Obtain target delta dynamic nodes
        y_target = batch.y[:, :4] - x[:, :4]

        # Update normalisation state and normalise
        if edge_attr is None:
            self.update_in_normalisation(x.clone())
        else:
            self.update_in_normalisation(x.clone(), edge_attr.clone())
        self.update_out_normalisation(y_target.clone())

        # Obtain normalised input graph and normalised target nodes
        x_nrm, edge_attr_nrm = self.in_normalise(x, edge_attr)
        y_target_nrm = self.out_normalise(y_target)

        # Obtain normalised predicted delta dynamics
        y_hat = self.model(
            x=x_nrm, edge_index=edge_index, edge_attr=edge_attr_nrm, batch=batch.batch
        )

        # Compute and log loss
        pos_loss = self.train_pos_loss(y_hat[:, :2], y_target_nrm[:, :2])
        vel_loss = self.train_pos_loss(y_hat[:, 2:], y_target_nrm[:, 2:])

        self.log("train_pos_loss", pos_loss, on_step=True, on_epoch=True)
        self.log("train_vel_loss", vel_loss, on_step=True, on_epoch=True)
        self.log(
            "train_total_loss", (pos_loss + vel_loss) / 2, on_step=True, on_epoch=True
        )

        return pos_loss + vel_loss

    def validation_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Validate on sequential dataset. First 11 observations are used to prime the model.
        # Loss is computed on remaining 80 samples using rollout.

        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, 4))
        y_target = torch.zeros((80, n_nodes, 4))
        edge_index = batch.edge_index
        static_features = batch.x[:, 0, 4:]
        edge_attr = None

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            x = batch.x[:, t, :]

            # Permutation for baseline experiment

            x = x

            if self.min_dist is not None:
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to ensure connected graphs
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=1, batch=batch.batch
                )
                # Remove duplicates
                edge_index = torch_geometric.utils.coalesce(
                    torch.cat((edge_index, nn_edge_index), dim=1)
                )
                self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x[row, 4] * 1e10
                    m2 = x[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            ######################
            # Validation 1/2     #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise output dynamics
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (batch.x[:, t, :4] + x, static_features), dim=-1
            )  # [n_nodes, n_features]

        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph[:, :4]
        y_target[0, :, :] = batch.x[:, 11, :4]

        ######################
        # Future             #
        ######################

        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            x = predicted_graph

            if self.min_dist is not None:
                # Edge indices of close nodes
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to entotal_valsure connected graphs
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=1, batch=batch.batch
                )
                # Remove duplicates
                edge_index = torch_geometric.utils.coalesce(
                    torch.cat((edge_index, nn_edge_index), dim=1)
                )
                self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x[row, 4] * 1e10
                    m2 = x[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            ######################
            # Validation 2/2     #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise deltas
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (predicted_graph[:, :4] + x, static_features), dim=-1
            )  # [n_nodes, n_features]

            # Save prediction alongside true value (next time step state)
            y_hat[t - 10, :, :] = predicted_graph[:, :4]
            y_target[t - 10, :, :] = batch.x[:, t + 1, :4]

        # Compute and log loss
        fde_loss = self.val_fde_loss(y_hat[-1, :, :2], y_target[-1, :, :2])
        ade_loss = self.val_ade_loss(y_hat[:, :, :2], y_target[:, :, :2])
        vel_loss = self.val_vel_loss(y_hat[:, :, 2:], y_target[:, :, 2:])

        ######################
        # Logging            #
        ######################

        self.log("val_ade_loss", ade_loss)
        self.log("val_fde_loss", fde_loss)
        self.log("val_vel_loss", vel_loss)
        self.log("val_total_loss", (ade_loss + vel_loss) / 2)

        # Log normalisation states
        if self.log_norm:
            self.log(
                "in_std",
                {f"in_std_{i}": std for i, std in enumerate(self.in_std)},
            )
            self.log(
                "in_mean",
                {f"in_mean_{i}": mean for i, mean in enumerate(self.in_mean)},
            )
            self.log(
                "out_std",
                {f"out_std_{i}": std for i, std in enumerate(self.out_std)},
            )
            self.log(
                "out_mean",
                {f"out_mean_{i}": mean for i, mean in enumerate(self.out_mean)},
            )

        return (ade_loss + vel_loss) / 2

    def predict_step(self, batch, batch_idx=None):

        ######################
        # Initialisation     #
        ######################

        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        n_features = 5
        y_hat = torch.zeros((90, n_nodes, n_features))
        y_target = torch.zeros((90, n_nodes, n_features))
        edge_index = batch.edge_index
        static_features = batch.x[:, 0, 4:]
        edge_attr = None

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            x = batch.x[:, t, :]

            if self.min_dist is not None:
                # Edge indices of close nodes
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to ensure connected graphs
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=1, batch=batch.batch
                )
                # Remove duplicates
                edge_index = torch_geometric.utils.coalesce(
                    torch.cat((edge_index, nn_edge_index), dim=1)
                )

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x[row, 4] * 1e10
                    m2 = x[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            ######################
            # Predictions        #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise output dynamics
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (batch.x[:, t, :4] + x, static_features), dim=-1
            )
            # Save predictions
            y_hat[t, :, :] = predicted_graph[:, :]
            y_target[t, :, :] = batch.x[:, t + 1, :]

        ######################
        # Future             #
        ######################

        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            x = predicted_graph

            if self.min_dist is not None:
                # Edge indices of close nodes
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to ensure connected graphs
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=1, batch=batch.batch
                )
                # Remove duplicates
                edge_index = torch_geometric.utils.coalesce(
                    torch.cat((edge_index, nn_edge_index), dim=1)
                )

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x[row, 4] * 1e10
                    m2 = x[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            ######################
            # Predictions        #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise deltas
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (predicted_graph[:, :4] + x, static_features), dim=-1
            )  # [n_nodes, n_features]

            # Save prediction alongside true value (next time step state)
            y_hat[t, :, :] = predicted_graph[:, :]
            y_target[t, :, :] = batch.x[:, t + 1, :]

        return y_hat, y_target

    def test_step(self, batch: Batch, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class SequentialModule(pl.LightningModule):
    def __init__(
        self,
        model_type: Union[None, str],
        model_dict: Union[None, dict],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        noise: Union[None, float] = None,
        teacher_forcing_ratio: float = 0.3,
        min_dist: int = 0,
        n_neighbours: int = 30,
        fully_connected: bool = True,
        edge_weight: bool = False,
        edge_type: str = "distance",
        self_loop: bool = True,
        undirected: bool = False,
        out_features: int = 6,
        node_features: int = 9,
        edge_features: int = 1,
        training_horizon: int = 90,
        grav_attraction: bool = False,
    ):
        super().__init__()

        # Setup metrics
        self.train_ade_loss = torchmetrics.MeanSquaredError()
        self.train_fde_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.test_ade_loss = torchmetrics.MeanSquaredError()
        self.test_fde_loss = torchmetrics.MeanSquaredError()
        self.test_vel_loss = torchmetrics.MeanSquaredError()

        # Instantiate model
        self.model_type = model_type
        self.model = eval(model_type)(**model_dict)

        # Learning parameters
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.training_horizon = training_horizon

        # Model parameters
        self.rnn_type = (
            model_dict["rnn_type"] if "rnn_type" in model_dict.keys() else None
        )
        self.out_features = out_features
        self.edge_features = edge_features
        self.node_features = node_features

        # Graph parameters
        self.min_dist = min_dist
        self.edge_weight = edge_weight
        self.undirected = undirected
        self.self_loop = self_loop
        self.grav_attraction = grav_attraction
        self.edge_type = edge_type
        self.fully_connected = fully_connected
        self.n_neighbours = n_neighbours

        if self.fully_connected:
            self.edge_type = "knn"
            self.n_neighbours = 100

        self.save_hyperparameters()

    def training_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Extract data from batch
        n_nodes = batch.num_nodes
        static_features = batch.x[:, 10, 4].unsqueeze(1)
        edge_attr = None

        # Ignore data after training horizon
        batch.x = batch.x[:, : (self.training_horizon + 1)]

        # Allocate prediction tensor
        y_predictions = torch.zeros((n_nodes, self.training_horizon, self.out_features))
        y_predictions = y_predictions.type_as(batch.x)
        # Define target tensor
        y_target = batch.x[:, 1 : (self.training_horizon + 1), : self.out_features]
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
        elif self.rnn_type == "LSTM":
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
        else:
            h_node, h_edge, c_node, c_edge = None, None, None, None

        ######################
        # History            #
        ######################

        for t in range(11):

            # Extract current input
            x_t = batch.x[:, t, :]
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
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            #######################
            # Training 1/2        #
            #######################

            # Obtain predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            elif self.rnn_type == "LSTM":  # LSTM
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )
            else:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )

            # Compute updated positions
            vel = delta_x[:, [0, 1]]
            pos = batch.x[:, t][:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[:, t, :] = x_t[:, : self.out_features]

        # If using teacher_forcing, draw sample and accept <teach_forcing_ratio*100> % of the time. Else, deny.
        use_groundtruth = random.random() < self.teacher_forcing_ratio

        ######################
        # Future             #
        ######################

        for t in range(11, self.training_horizon):
            # Use groundtruth 'teacher_forcing_ratio' % of the time
            if use_groundtruth:
                x_t = batch.x[:, t, :].clone()
            x_prev = x_t.clone()

            ######################
            # Graph construction #
            ######################

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            #######################
            # Training 2/2        #
            #######################

            # Obtain predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            elif self.rnn_type == "LSTM":  # LSTM
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )
            else:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )

            # Compute updated positions
            vel = delta_x[:, [0, 1]]
            pos = x_prev[:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[:, t, :] = x_t[:, : self.out_features]

        # Compute and log loss
        fde_loss = self.train_fde_loss(y_predictions[:, -1, :2], y_target[:, -1, :2])
        ade_loss = self.train_ade_loss(y_predictions[:, :, :2], y_target[:, :, :2])
        vel_loss = self.train_vel_loss(
            y_predictions[:, :, [2, 3]], y_target[:, :, [2, 3]]
        )

        self.log("train_fde_loss", fde_loss, on_step=True, on_epoch=True, batch_size=n_nodes)
        self.log("train_ade_loss", ade_loss, on_step=True, on_epoch=True, batch_size=n_nodes)
        self.log("train_vel_loss", vel_loss, on_step=True, on_epoch=True, batch_size=n_nodes)
        loss = ade_loss
        self.log("train_total_loss", loss, on_step=True, on_epoch=True, batch_size=n_nodes)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Extract data from batch
        n_nodes = batch.num_nodes
        static_features = batch.x[:, 10, 4].unsqueeze(1)
        edge_attr = None

        # Ignore data after training horizon
        batch.x = batch.x[:, : (self.training_horizon + 1)]

        # Allocate prediction tensor
        y_predictions = torch.zeros((n_nodes, self.training_horizon, self.out_features))
        y_predictions = y_predictions.type_as(batch.x)
        # Define target tensor
        y_target = batch.x[:, 1 : (self.training_horizon + 1), : self.out_features]
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
        elif self.rnn_type == "LSTM":
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
        else:
            h_node, h_edge, c_node, c_edge = None, None, None, None

        ######################
        # History            #
        ######################

        for t in range(11):

            # Extract current input
            x_t = batch.x[:, t, :]
            x_t = x_t.type_as(batch.x)

            ######################
            # Graph construction #
            ######################

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            #######################
            # Training 1/2        #
            #######################

            # Obtain predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            elif self.rnn_type == "LSTM":  # LSTM
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )
            else:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )

            # Compute updated positions
            vel = delta_x[:, [0, 1]]
            pos = batch.x[:, t][:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[:, t, :] = x_t[:, : self.out_features]

        ######################
        # Future             #
        ######################

        for t in range(11, self.training_horizon):

            x_prev = x_t.clone()

            ######################
            # Graph construction #
            ######################

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            #######################
            # Training 2/2        #
            #######################

            # Obtain predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            elif self.rnn_type == "LSTM":  # LSTM
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )
            else:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )

            # Compute updated positions
            vel = delta_x[:, [0, 1]]
            pos = x_prev[:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[:, t, :] = x_t[:, : self.out_features]

        # Compute and log loss
        fde_loss = self.val_fde_loss(y_predictions[:, -1, :2], y_target[:, -1, :2])
        ade_loss = self.val_ade_loss(y_predictions[:, 11:, :2], y_target[:, 11:, :2])
        vel_loss = self.val_vel_loss(
            y_predictions[:, 11:, [2, 3]], y_target[:, 11:, [2, 3]]
        )

        self.log("val_fde_loss", fde_loss, batch_size=n_nodes)
        self.log("val_ade_loss", ade_loss, batch_size=n_nodes)
        self.log("val_vel_loss", vel_loss, batch_size=n_nodes)
        loss = ade_loss

        self.log("val_total_loss", loss, batch_size=n_nodes)

        return loss

    def predict_step(self, batch: Batch, batch_idx=None, prediction_horizon: int = 90):

        ######################
        # Initialisation     #
        ######################

        # Extract data from batch
        n_nodes = batch.num_nodes
        static_features = batch.x[:, 10, 4].unsqueeze(1)
        edge_attr = None

        # Ignore data after training horizon
        batch.x = batch.x[:, : (self.training_horizon + 1)]

        # Allocate prediction tensor
        y_predictions = torch.zeros((n_nodes, prediction_horizon, self.node_features))
        y_predictions = y_predictions.type_as(batch.x)
        # Define target tensor
        y_target = batch.x[:, 1 : (self.training_horizon + 1)]
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
        elif self.rnn_type == "LSTM":
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
        else:
            h_node, h_edge, c_node, c_edge = None, None, None, None

        ######################
        # History            #
        ######################

        for t in range(11):

            # Extract current input
            x_t = batch.x[:, t, :]
            x_t = x_t.type_as(batch.x)

            ######################
            # Graph construction #
            ######################

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            #######################
            # Training 1/2        #
            #######################

            # Obtain predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            elif self.rnn_type == "LSTM":  # LSTM
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )
            else:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )

            # Compute updated positions
            vel = delta_x[:, [0, 1]]
            pos = batch.x[:, t][:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[:, t, :] = x_t

        ######################
        # Future             #
        ######################

        for t in range(11, prediction_horizon):

            x_prev = x_t.clone()

            ######################
            # Graph construction #
            ######################

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            #######################
            # Training 2/2        #
            #######################

            # Obtain predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            elif self.rnn_type == "LSTM":  # LSTM
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )
            else:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )

            # Compute updated positions
            vel = delta_x[:, [0, 1]]
            pos = x_prev[:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[:, t, :] = x_t

        return y_predictions, y_target

    def test_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Extract data from batch
        n_nodes = batch.num_nodes
        static_features = batch.x[:, 10, 4].unsqueeze(1)
        edge_attr = None

        # Ignore data after training horizon
        batch.x = batch.x[:, : (self.training_horizon + 1)]

        # Allocate prediction tensor
        y_predictions = torch.zeros((n_nodes, self.training_horizon, self.out_features))
        y_predictions = y_predictions.type_as(batch.x)
        # Define target tensor
        y_target = batch.x[:, 1 : (self.training_horizon + 1), : self.out_features]
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
        elif self.rnn_type == "LSTM":
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
        else:
            h_node, h_edge, c_node, c_edge = None, None, None, None

        ######################
        # History            #
        ######################

        for t in range(11):

            # Extract current input
            x_t = batch.x[:, t, :]
            x_t = x_t.type_as(batch.x)

            ######################
            # Graph construction #
            ######################

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            #######################
            # Training 1/2        #
            #######################

            # Obtain predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            elif self.rnn_type == "LSTM":  # LSTM
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )
            else:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )

            # Compute updated positions
            vel = delta_x[:, [0, 1]]
            pos = batch.x[:, t][:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[:, t, :] = x_t[:, : self.out_features]

        ######################
        # Future             #
        ######################

        for t in range(11, self.training_horizon):

            x_prev = x_t.clone()

            ######################
            # Graph construction #
            ######################

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            #######################
            # Training 2/2        #
            #######################

            # Obtain predicted delta dynamics
            if self.rnn_type == "GRU":
                delta_x, (h_node, h_edge) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=(h_node, h_edge),
                )
            elif self.rnn_type == "LSTM":  # LSTM
                delta_x, ((h_node, c_node), (h_edge, c_edge)) = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=((h_node, c_node), (h_edge, c_edge)),
                )
            else:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )

            # Compute updated positions
            vel = delta_x[:, [0, 1]]
            pos = x_prev[:, [0, 1]] + 0.1 * vel
            x_t = torch.cat([pos, vel, static_features], dim=-1)
            x_t = x_t.type_as(batch.x)

            # Save deltas for loss computation
            y_predictions[:, t, :] = x_t[:, : self.out_features]

        # Compute and log loss
        fde_loss = self.test_fde_loss(y_predictions[:, -1, :2], y_target[:, -1, :2])
        ade_loss = self.test_ade_loss(y_predictions[:, 11:, :2], y_target[:, 11:, :2])
        vel_loss = self.test_vel_loss(
            y_predictions[:, 11:, [2, 3]], y_target[:, 11:, [2, 3]]
        )

        self.log("test_fde_loss", fde_loss)
        self.log("test_ade_loss", ade_loss)
        self.log("test_vel_loss", vel_loss)
        loss = ade_loss

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class ConstantPhysicalBaselineModule(pl.LightningModule):
    def __init__(self, prediction_horizon: int = 90, out_features: int = 4, **kwargs):
        super().__init__()
        # Setup metrics
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_total_loss = torchmetrics.MeanSquaredError()

        self.prediction_horizon = prediction_horizon
        self.out_features = out_features
        self.save_hyperparameters()

    def training_step(self, batch: Batch, batch_idx: int):
        pass

    def validation_step(self, batch: Batch, batch_idx: int):
        # Validate on sequential dataset. First 11 observations are used to prime the model.
        # Loss is computed on remaining 80 samples using rollout.

        # Setup target and allocate prediction tensors
        y_target = batch.x[:, 11 : (self.prediction_horizon + 1), : self.out_features]
        y_hat = torch.zeros_like(y_target)
        static_features = batch.x[:, 0, 4].unsqueeze(1)

        # Extract last observed positions/velocities
        last_pos = batch.x[:, 10, :2]
        last_vel = batch.x[:, 10, 2:4]
        # Compute delta change
        delta_pos = last_vel * 0.1
        # First predicted positions
        predicted_pos = last_pos + delta_pos
        predicted_graph = torch.cat((predicted_pos, last_vel, static_features), dim=-1)
        # Save first prediction and target
        y_hat[:, 0, :] = predicted_graph[:, :4]

        # 1 prediction done, 79 remaining
        for t in range(11, self.prediction_horizon):
            predicted_pos += delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_vel, static_features), dim=-1
            )
            y_hat[:, t - 10, :] = predicted_graph[:, :4]

        # Compute and log loss
        fde_loss = self.val_fde_loss(y_hat[:, -1, :2], y_target[:, -1, :2])
        ade_loss = self.val_ade_loss(y_hat[:, 11:, :2], y_target[:, 11:, :2])
        vel_loss = self.val_vel_loss(y_hat[:, 11:, 2:], y_target[:, 11:, 2:])

        self.log("val_ade_loss", ade_loss)
        self.log("val_fde_loss", fde_loss)
        self.log("val_vel_loss", vel_loss)
        self.log("val_total_loss", ade_loss)

        return ade_loss

    def predict_step(self, batch: Batch, prediction_horizon: int = 51):
        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        n_features = 5
        y_hat = torch.zeros((prediction_horizon, n_nodes, n_features))
        y_target = torch.zeros((prediction_horizon, n_nodes, n_features))

        # Fill in targets
        for t in range(0, prediction_horizon):
            y_target[t, :, :] = batch.x[:, t + 1, :]

        static_features = batch.x[:, 0, 4].unsqueeze(1)
        for t in range(11):
            last_pos = batch.x[:, t, :2]
            last_vel = batch.x[:, t, 2:4]
            delta_pos = last_vel * 0.1
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_vel, static_features), dim=1
            )
            y_hat[t, :, :] = predicted_graph

        for t in range(11, prediction_horizon):
            last_pos = predicted_pos
            # velocity no longer changing
            # delta_pos no longer changing
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_vel, static_features), dim=1
            )
            y_hat[t, :, :] = predicted_graph

        y_hat = torch.permute(y_hat, (1, 0, 2))
        y_target = torch.permute(y_target, (1, 0, 2))
        return y_hat, y_target

    def test_step(self, batch: Batch, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


@hydra.main(config_path="../../configs/nbody/", config_name="config")
def main(config):
    # Print configuration for online monitoring
    print(OmegaConf.to_yaml(config))
    # Save complete yaml file for logging and reproducibility
    log_dir = f"logs/nbody/{config.logger.project}/{config.logger.version}"
    os.makedirs(log_dir, exist_ok=True)
    yaml_path = f"{log_dir}/{config.logger.version}.yaml"
    OmegaConf.save(config, f=yaml_path)

    # Seed for reproducibility
    seed_everything(config["misc"]["seed"], workers=True)
    # Load data, model, and regressor
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])
    # Define model
    if config["misc"]["model_type"] != "ConstantModel":
        model_dict = dict(config["model"])
        model_type = config["misc"]["model_type"]
    else:
        model_dict, model_type = None, None
        config["misc"]["regressor_type"] = "ConstantPhysicalBaselineModule"
        config["misc"]["train"] = False

    # Define LightningModule
    regressor = eval(config["misc"]["regressor_type"])(
        model_type=model_type, model_dict=model_dict, **config["regressor"]
    )

    # Setup logging (using saved yaml file)
    wandb_logger = WandbLogger(
        entity="petergroth",
        config=OmegaConf.to_container(config, resolve=True),
        **config["logger"],
    )
    wandb_logger.watch(regressor, log_freq=config["misc"]["log_freq"], log_graph=False)

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

    trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    main()
