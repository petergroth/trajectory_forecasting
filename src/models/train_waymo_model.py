import argparse
import pytorch_lightning as pl
import torch
import torch_geometric.nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule
import torchmetrics
from torch_geometric.data import Batch
from src.models.model import *
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Union
from pytorch_lightning.callbacks import RichProgressBar
import math

class OneStepModule(pl.LightningModule):
    def __init__(
        self,
        model_type: Union[None, str],
        model_dict: Union[None, dict],
        noise: Union[None, float] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        edge_type: str = "knn",
        min_dist: int = 0,
        n_neighbours: int = 30,
        fully_connected: bool = True,
        edge_weight: bool = False,
        self_loop: bool = True,
        undirected: bool = False,
        out_features: int = 6,
        normalise: bool = True,
        node_features: int = 9,
        edge_features: int = 1,
    ):
        super().__init__()
        # Verify inputs
        assert edge_type in ["knn", "distance"]
        if edge_type == "distance":
            assert min_dist > 0.0

        # Instantiate model
        self.model_type = model_type
        self.model = eval(model_type)(**model_dict)

        # Setup metrics
        self.train_pos_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        self.train_yaw_loss = torchmetrics.MeanSquaredError()
        self.train_difference_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_yaw_loss = torchmetrics.MeanSquaredError()
        self.val_fde_ttp_loss = torchmetrics.MeanSquaredError()
        self.val_ade_ttp_loss = torchmetrics.MeanSquaredError()

        # Learning parameters
        self.normalise = normalise
        self.global_scale = 8.025897979736328
        # self.global_scale = 1
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay

        # Model parameters
        self.out_features = out_features
        self.edge_features = edge_features
        self.node_features = node_features

        # Graph parameters
        self.edge_type = edge_type
        self.min_dist = min_dist
        self.fully_connected = fully_connected
        self.n_neighbours = 128 if fully_connected else n_neighbours
        self.edge_weight = edge_weight
        self.self_loop = self_loop
        self.undirected = undirected

        self.save_hyperparameters()

    def training_step(self, batch: Batch, batch_idx: int):
        # CARS
        type_mask = batch.x[:, 11] == 1
        batch.x = batch.x[type_mask]
        batch.y = batch.y[type_mask]
        batch.batch = batch.batch[type_mask]
        # Remove type from data
        batch.x = batch.x[:, :10]

        # Extract node features
        x = batch.x
        edge_attr = None

        ######################
        # Graph construction #
        ######################

        # Construct edges
        if self.edge_type == "knn":
            # Neighbour-based graph
            edge_index = torch_geometric.nn.knn_graph(
                x=x[:, :2], k=self.n_neighbours, batch=batch.batch, loop=self.self_loop
            )
        else:
            # Distance-based graph
            edge_index = torch_geometric.nn.radius_graph(
                x=x[:, :2],
                r=self.min_dist,
                batch=batch.batch,
                loop=self.self_loop,
                max_num_neighbors=self.n_neighbours,
                flow="source_to_target",
            )

        if self.undirected:
            edge_index, _ = torch_geometric.utils.to_undirected(edge_index)

        # Remove duplicates and sort
        edge_index = torch_geometric.utils.coalesce(edge_index)
        self.log("train_edges_per_node", edge_index.shape[1] / x.shape[0])

        # Determine whether to add random noise to dynamic states
        if self.noise is not None:
            x[:, : self.out_features] += self.noise * torch.randn_like(
                x[:, : self.out_features]
            )

        # Create edge_attr if specified
        if self.edge_weight:
            # Encode distance between nodes as edge_attr
            row, col = edge_index
            edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
            edge_attr = edge_attr.type_as(batch.x)

        ######################
        # Training 1/1       #
        ######################

        # Obtain target delta dynamic nodes
        y_target = batch.y[:, : self.out_features] - x[:, : self.out_features]
        y_target = y_target.type_as(batch.x)

        if self.normalise:
            if edge_attr is None:
                # Center node positions
                x[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                # Scale all features (except yaws) with global scaler
                x[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
            else:
                # Center node positions
                x[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                # Scale all features (except yaws) with global scaler
                x[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                # Scale edge attributes
                edge_attr /= self.global_scale

        # Obtain predicted delta dynamics
        delta_x = self.model(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
        )

        # Process predicted yaw values
        yaw_pred = torch.tanh(delta_x[:, 5:9])
        yaw_targ = torch.vstack(
            [
                torch.sin(y_target[:, 5]),
                torch.cos(y_target[:, 5]),
                torch.sin(y_target[:, 6]),
                torch.cos(y_target[:, 6]),
            ]
        ).T

        # Compute new positions using old velocities (in normalised space)
        # pos_expected = batch.x[:, [0, 1]] + 0.1 * batch.x[:, [3, 4]]
        # # Compute new positions by updating old position with new (normalised) delta dynamics
        # pos_new = delta_x[:, [0, 1]] / self.global_scale + batch.x[:, [0, 1]]

        # Compute and log loss
        pos_loss = self.train_pos_loss(delta_x[:, :3], y_target[:, :3])
        vel_loss = self.train_vel_loss(delta_x[:, 3:5], y_target[:, 3:5])
        yaw_loss = self.train_yaw_loss(yaw_pred, yaw_targ)
        # pos_diff = self.train_difference_loss(pos_new, pos_expected)

        self.log("train_pos_loss", pos_loss, on_step=True, on_epoch=True)
        self.log("train_vel_loss", vel_loss, on_step=True, on_epoch=True)
        self.log("train_yaw_loss", yaw_loss, on_step=True, on_epoch=True)
        # self.log("position_difference", pos_diff, on_step=True, on_epoch=True)

        loss = pos_loss + vel_loss + yaw_loss  # +  pos_diff

        self.log("train_total_loss", loss, on_step=True, on_epoch=True)

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

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, self.out_features))
        y_target = torch.zeros((80, n_nodes, self.out_features))
        # Ensure device placement
        y_hat = y_hat.type_as(batch.x)
        y_target = y_target.type_as(batch.x)

        # Discard mask from features and extract static features
        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = torch.cat(
            [batch.x[:, 10, self.out_features:]], dim=1
        )
        static_features = static_features.type_as(batch.x)
        edge_attr = None

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :].clone()
            batch_t = batch.batch[mask_t]

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch_t,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch_t,
                    loop=self.self_loop,
                    max_num_neighbors=self.n_neighbours,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, _ = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)
            self.log(
                "val_history_edges_per_node",
                edge_index.shape[1] / x_t.shape[0],
                on_step=True,
                on_epoch=True,
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
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain predicted delta dynamics
            delta_x = self.model(
                x=x_t, edge_index=edge_index, edge_attr=edge_attr, batch=batch_t
            )

            # Transform yaw
            bbox_yaw = torch.atan2(
                torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
            ).unsqueeze(1)
            vel_yaw = torch.atan2(
                torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
            ).unsqueeze(1)
            tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
            delta_x = tmp

            # Add deltas to input graph
            predicted_graph = torch.cat(
                (
                    batch.x[mask_t, t, : self.out_features] + delta_x,
                    static_features[mask_t],
                ),
                dim=-1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Process yaw values to ensure [-pi, pi] interval
            yaws = predicted_graph[:, [5, 6]]
            yaws[yaws > 0] = (
                    torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                    - math.pi
            )
            yaws[yaws < 0] = (
                    torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                    + math.pi
            )
            predicted_graph[:, [5, 6]] = yaws


        # Save first prediction and target
        y_hat[0, mask_t, :] = predicted_graph[:, : self.out_features]
        y_target[0, mask_t, :] = batch.x[mask_t, 11, : self.out_features]

        ######################
        # Future             #
        ######################

        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            # Latest prediction as input
            x_t = predicted_graph.clone()

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
                edge_index, _ = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)
            self.log(
                "val_future_edges_per_node",
                edge_index.shape[1] / x_t.shape[0],
                on_step=True,
                on_epoch=True,
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
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain predicted delta dynamics

            delta_x = self.model(
                x=x_t, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )

            # Transform yaw
            bbox_yaw = torch.atan2(
                torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
            ).unsqueeze(1)
            vel_yaw = torch.atan2(
                torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
            ).unsqueeze(1)
            tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
            delta_x = tmp

            # Add deltas to input graph
            predicted_graph = torch.cat(
                [
                    predicted_graph[:, : self.out_features] + delta_x,
                    predicted_graph[:, self.out_features :],
                ],
                dim=1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Process yaw values to ensure [-pi, pi] interval
            yaws = predicted_graph[:, [5, 6]]
            yaws[yaws > 0] = (
                    torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                    - math.pi
            )
            yaws[yaws < 0] = (
                    torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                    + math.pi
            )
            predicted_graph[:, [5, 6]] = yaws

            # Save prediction alongside true value (next time step state)
            y_hat[t - 10, :, :] = predicted_graph[:, : self.out_features]
            y_target[t - 10, :, :] = batch.x[:, t + 1, : self.out_features]

        fde_mask = mask[:, -1]
        val_mask = mask[:, 11:].permute(1, 0)

        # Compute and log loss
        fde_loss = self.val_fde_loss(
            y_hat[-1, fde_mask, :3], y_target[-1, fde_mask, :3]
        )
        ade_loss = self.val_ade_loss(
            y_hat[:, :, 0:3][val_mask], y_target[:, :, 0:3][val_mask]
        )
        vel_loss = self.val_vel_loss(
            y_hat[:, :, 3:5][val_mask], y_target[:, :, 3:5][val_mask]
        )
        yaw_loss = self.val_yaw_loss(
            y_hat[:, :, 5:7][val_mask], y_target[:, :, 5:7][val_mask]
        )

        # Compute losses on "tracks_to_predict"
        fde_ttp_mask = torch.logical_and(fde_mask, batch.tracks_to_predict)
        fde_ttp_loss = self.val_fde_ttp_loss(
            y_hat[-1, fde_ttp_mask, :3], y_target[-1, fde_ttp_mask, :3]
        )
        ade_ttp_mask = torch.logical_and(
            val_mask, batch.tracks_to_predict.expand((80, mask.size(0)))
        )
        ade_ttp_loss = self.val_ade_loss(
            y_hat[:, :, 0:3][ade_ttp_mask], y_target[:, :, 0:3][ade_ttp_mask]
        )

        ######################
        # Logging            #
        ######################

        self.log("val_ade_loss", ade_loss)
        self.log("val_fde_loss", fde_loss)
        self.log("val_vel_loss", vel_loss)
        self.log("val_yaw_loss", yaw_loss)
        self.log("val_total_loss", (ade_loss + vel_loss + yaw_loss) / 3)
        self.log("val_fde_ttp_loss", fde_ttp_loss)
        self.log("val_ade_ttp_loss", ade_ttp_loss)

        return (ade_loss + vel_loss + yaw_loss) / 3

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

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((90, n_nodes, self.node_features))
        y_target = torch.zeros((90, n_nodes, self.node_features))
        # Ensure device placement
        y_hat = y_hat.type_as(batch.x)
        y_target = y_target.type_as(batch.x)

        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = torch.cat(
            [batch.x[:, 10, self.out_features:]], dim=1
        )
        edge_attr = None

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :].clone()
            batch_t = batch.batch[mask_t]

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch_t,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch_t,
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

            ######################
            # Prediction 1/2     #
            ######################

            # Normalise input graph
            if self.normalise:
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain predicted delta dynamics
            delta_x = self.model(
                x=x_t, edge_index=edge_index, edge_attr=edge_attr, batch=batch_t
            )

            # Transform yaw
            bbox_yaw = torch.atan2(
                torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
            ).unsqueeze(1)
            vel_yaw = torch.atan2(
                torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
            ).unsqueeze(1)
            tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
            delta_x = tmp

            # Add deltas to input graph
            predicted_graph = torch.cat(
                (
                    batch.x[mask_t, t, : self.out_features] + delta_x,
                    static_features[mask_t],
                ),
                dim=-1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Process yaw values to ensure [-pi, pi] interval
            yaws = predicted_graph[:, [5, 6]]
            yaws[yaws > 0] = (
                    torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                    - math.pi
            )
            yaws[yaws < 0] = (
                    torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                    + math.pi
            )
            predicted_graph[:, [5, 6]] = yaws

            # Save first prediction and target
            y_hat[t, mask_t, :] = predicted_graph
            # y_target[t, mask_t, :] = torch.cat(
            #     [batch.x[mask_t, t + 1, :], batch.type[mask_t]], dim=-1
            # )

            y_target[t, mask_t, :] = batch.x[mask_t, t + 1, :].clone()

        ######################
        # Future             #
        ######################

        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            # Latest prediction as input
            x_t = predicted_graph.clone()

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

            ######################
            # Prediction 2/2     #
            ######################

            # Normalise input graph
            if self.normalise:
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain predicted delta dynamics
            delta_x = self.model(
                x=x_t, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )

            # Transform yaw
            bbox_yaw = torch.atan2(
                torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
            ).unsqueeze(1)
            vel_yaw = torch.atan2(
                torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
            ).unsqueeze(1)
            tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
            delta_x = tmp

            # Add deltas to input graph
            predicted_graph = torch.cat(
                [
                    predicted_graph[:, : self.out_features] + delta_x,
                    predicted_graph[:, self.out_features :],
                ],
                dim=1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Process yaw values to ensure [-pi, pi] interval
            yaws = predicted_graph[:, [5, 6]]
            yaws[yaws > 0] = (
                    torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                    - math.pi
            )
            yaws[yaws < 0] = (
                    torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                    + math.pi
            )
            predicted_graph[:, [5, 6]] = yaws

            # Save prediction alongside true value (next time step state)
            y_hat[t, :, :] = predicted_graph
            # y_target[t, :, :] = torch.cat([batch.x[:, t + 1], batch.type], dim=-1)
            y_target[t, :, :] = batch.x[:, t + 1].clone()

        return y_hat, y_target, mask

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
        teacher_forcing: bool = False,
        teacher_forcing_ratio: float = 0.3,
        min_dist: int = 0,
        n_neighbours: int = 30,
        fully_connected: bool = True,
        edge_weight: bool = False,
        edge_type: str = "knn",
        self_loop: bool = True,
        undirected: bool = False,
        rnn_type: str = "GRU",
        out_features: int = 6,
        node_features: int = 9,
        edge_features: int = 1,
        normalise: bool = True,
        training_horizon: int = 90
    ):
        super().__init__()
        # Verify inputs
        assert edge_type in ["knn", "distance"]
        if edge_type == "distance":
            assert min_dist > 0.0

        # Set up metrics
        self.train_ade_loss = torchmetrics.MeanSquaredError()
        self.train_fde_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        self.train_yaw_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_yaw_loss = torchmetrics.MeanSquaredError()
        self.val_fde_ttp_loss = torchmetrics.MeanSquaredError()
        self.val_ade_ttp_loss = torchmetrics.MeanSquaredError()

        # Instantiate model
        self.model_type = model_type
        self.model = eval(model_type)(**model_dict)

        # Learning parameters
        self.normalise = normalise
        self.global_scale = 8.025897979736328
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay
        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.training_horizon = training_horizon

        # Model parameters
        self.rnn_type = rnn_type
        self.out_features = out_features
        self.edge_features = edge_features
        self.node_features = node_features

        # Graph parameters
        self.edge_type = edge_type
        self.min_dist = min_dist
        self.fully_connected = fully_connected
        self.n_neighbours = 128 if fully_connected else n_neighbours
        self.edge_weight = edge_weight
        self.self_loop = self_loop
        self.undirected = undirected

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
        batch.x = batch.x[:, :(self.training_horizon+1)]

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Discard masks and extract static features
        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = batch.x[:, 10, self.out_features:].clone()
        static_features = static_features.type_as(batch.x)
        edge_attr = None
        # Extract dimensions and allocate predictions
        n_nodes = batch.num_nodes
        y_predictions = torch.zeros((n_nodes, self.training_horizon, self.out_features+2))
        y_predictions = y_predictions.type_as(batch.x)

        # Obtain target delta dynamic nodes
        # Use torch.roll to compute differences between x_t and x_{t+1}.
        # Ignore final difference (between t_0 and t_{-1})
        y_target = (
            torch.roll(batch.x[:, :, : self.out_features], (0, -1, 0), (0, 1, 2))
            - batch.x[:, :, : self.out_features]
        )[:, :-1, :]

        # Transform yaw values to their sines/cosines
        sines = torch.sin(y_target[:, :, [5, 6]])
        cosines = torch.cos(y_target[:, :, [5, 6]])

        # Replace yaws with sines/cosines
        y_target = torch.cat([
            y_target[:, :, [0, 1, 2, 3, 4]],
            sines[:, :, 0].unsqueeze(2),
            cosines[:, :, 0].unsqueeze(2),
            sines[:, :, 1].unsqueeze(2),
            cosines[:, :, 1].unsqueeze(2)
        ], dim=-1)
        y_target = y_target.type_as(batch.x)
        assert y_target.shape == y_predictions.shape

        # Initial hidden state
        if self.rnn_type == "GRU":
            h = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h = h.type_as(batch.x)
        elif self.rnn_type == "LSTM":
            raise NotImplementedError
            h = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
            c = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
            h = (h, c)
        else:
            h = None

        ######################
        # History            #
        ######################

        for t in range(11):

            # Extract current input and target
            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :].clone()
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
                    batch=batch.batch[mask_t],
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch[mask_t],
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

            #######################
            # Training 1/2        #
            #######################

            # Normalise input graph
            if self.normalise:
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain predicted delta dynamics
            if h is None:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                )
            else:
                # Add zero rows for new columns
                assert self.rnn_type == "GRU"
                delta_x, h_t = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=h[:, mask_t],
                )
                # Update hidden states
                h[:, mask_t] = h_t

            # Process predicted yaw values
            yaw_pred = torch.tanh(delta_x[:, 5:9])
            delta_x[:, [5, 6, 7, 8]] = yaw_pred

            # Save deltas for loss computation
            y_predictions[mask_t, t, :] = delta_x

            if t == 10:
                # Transform yaw values for next graph
                bbox_yaw = torch.atan2(
                    torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
                ).unsqueeze(1)
                vel_yaw = torch.atan2(
                    torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
                ).unsqueeze(1)
                tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
                delta_x = tmp

                # Add deltas to input graph
                x_t = torch.cat(
                    (
                        batch.x[mask_t, t, : self.out_features] + delta_x,
                        static_features[mask_t],
                    ),
                    dim=-1,
                )

                # Process yaw values of latest graph to ensure [-pi, pi] interval
                yaws = x_t[:, [5, 6]]
                yaws[yaws > 0] = (
                        torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                        - math.pi
                )
                yaws[yaws < 0] = (
                        torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                        + math.pi
                )
                x_t[:, [5, 6]] = yaws

        # If using teacher_forcing, draw sample and accept <teach_forcing_ratio*100> % of the time. Else, deny.
        use_groundtruth = (
            (
                torch.distributions.uniform.Uniform(0, 1).sample()
                < self.teacher_forcing_ratio
            )
            if self.teacher_forcing
            else False
        )

        ######################
        # Future             #
        ######################

        for t in range(11, self.training_horizon):
            # Use groundtruth 'teacher_forcing_ratio' % of the time
            if use_groundtruth:
                # x_t = torch.cat([batch.x[:, t, :], batch.type], dim=1)
                x_t = batch.x[:, t, :]
            x_prev = x_t

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

            #######################
            # Training 2/2        #
            #######################

            # Normalise input graph
            if self.normalise:
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if h is None:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )
            else:
                delta_x, h = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=h,
                )

            # Process predicted yaw values
            yaw_pred = torch.tanh(delta_x[:, 5:9])
            delta_x[:, [5, 6, 7, 8]] = yaw_pred

            # Save deltas for loss computation
            y_predictions[mask_t, t, :] = delta_x

            # Transform yaw
            bbox_yaw = torch.atan2(
                torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
            ).unsqueeze(1)
            vel_yaw = torch.atan2(
                torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
            ).unsqueeze(1)
            tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
            delta_x = tmp

            # Add deltas to input graph. Input for next timestep
            x_t = torch.cat(
                (
                    x_prev[:, : self.out_features] + delta_x,
                    x_prev[:, self.out_features :],
                ),
                dim=-1,
            )

            # Process yaw values to ensure [-pi, pi] interval
            yaws = x_t[:, [5, 6]]
            yaws[yaws > 0] = (
                    torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                    - math.pi
            )
            yaws[yaws < 0] = (
                    torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                    + math.pi
            )
            x_t[:, [5, 6]] = yaws

        # Determine valid inputs
        loss_mask_0 = mask[:, :self.training_horizon]
        # Determine valid targets
        loss_mask_1 = mask[:, 1:(self.training_horizon+1)]
        # Loss is computed at intersection
        loss_mask = torch.logical_and(loss_mask_0, loss_mask_1)
        fde_mask = mask[:, -2]

        # Compute and log loss
        fde_loss = self.train_fde_loss(
            y_predictions[fde_mask, -1, :3], y_target[fde_mask, -1, :3]
        )
        ade_loss = self.train_ade_loss(
            y_predictions[:, :self.training_horizon, :3][loss_mask],
            y_target[:, :self.training_horizon, :3][loss_mask]
        )
        vel_loss = self.train_vel_loss(
            y_predictions[:, :self.training_horizon, 3:5][loss_mask],
            y_target[:, :self.training_horizon, 3:5][loss_mask]
        )
        yaw_loss = self.train_yaw_loss(
            y_predictions[:, :self.training_horizon, [5, 6, 7, 8]][loss_mask],
            y_target[:, :self.training_horizon, [5, 6, 7, 8]][loss_mask]
        )

        self.log("train_fde_loss", fde_loss, on_step=True, on_epoch=True, batch_size=fde_mask.sum().item())
        self.log("train_ade_loss", ade_loss, on_step=True, on_epoch=True, batch_size=loss_mask.sum().item())
        self.log("train_vel_loss", vel_loss, on_step=True, on_epoch=True, batch_size=loss_mask.sum().item())
        self.log("train_yaw_loss", yaw_loss, on_step=True, on_epoch=True, batch_size=loss_mask.sum().item())
        loss = ade_loss + vel_loss + yaw_loss
        self.log(
            "train_total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=loss_mask.sum().item()
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

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, self.out_features))
        y_hat = y_hat.type_as(batch.x)
        y_target = torch.zeros((80, n_nodes, self.out_features))
        y_target = y_target.type_as(batch.x)
        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = batch.x[:, 10, self.out_features:].clone()
        static_features = static_features.type_as(batch.x)
        edge_attr = None

        # Initial hidden state
        if self.rnn_type == "GRU":
            h = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h = h.type_as(batch.x)
        elif self.rnn_type == "LSTM":
            h = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
            c = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
            h = (h, c)
        else:
            h = None

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :].clone()

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch[mask_t],
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch[mask_t],
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

            ######################
            # Validation 1/2     #
            ######################

            # Normalise input graph
            if self.normalise:
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if h is None:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                )
            else:
                delta_x, h_t = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=h[:, mask_t],
                )
                # Update hidden state
                h[:, mask_t] = h_t

            if t == 10:
                # Transform yaw
                bbox_yaw = torch.atan2(
                    torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
                ).unsqueeze(1)
                vel_yaw = torch.atan2(
                    torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
                ).unsqueeze(1)
                tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
                delta_x = tmp

                # Add deltas to input graph
                predicted_graph = torch.cat(
                    (
                        batch.x[mask_t, t, : self.out_features] + delta_x,
                        static_features[mask_t],
                    ),
                    dim=-1,
                )
                predicted_graph = predicted_graph.type_as(batch.x)

                # Process yaw values to ensure [-pi, pi] interval
                yaws = predicted_graph[:, [5, 6]]
                yaws[yaws > 0] = (
                        torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                        - math.pi
                )
                yaws[yaws < 0] = (
                        torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                        + math.pi
                )
                predicted_graph[:, [5, 6]] = yaws

        # Save first prediction and target
        y_hat[0, mask_t, :] = predicted_graph[:, : self.out_features]
        y_target[0, mask_t, :] = batch.x[mask_t, 11, : self.out_features]

        ######################
        # Future             #
        ######################

        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            # Latest prediction as input
            x_t = predicted_graph.clone()

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

            ######################
            # Validation 2/2     #
            ######################

            # Normalise input graph
            if self.normalise:
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if h is None:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )
            else:
                delta_x, h = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=h,
                )

            # Transform yaw
            bbox_yaw = torch.atan2(
                torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
            ).unsqueeze(1)
            vel_yaw = torch.atan2(
                torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
            ).unsqueeze(1)
            tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
            delta_x = tmp

            # Add deltas to input graph
            predicted_graph = torch.cat(
                (
                    predicted_graph[:, : self.out_features] + delta_x,
                    predicted_graph[:, self.out_features :],
                ),
                dim=-1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Process yaw values to ensure [-pi, pi] interval
            yaws = predicted_graph[:, [5, 6]]
            yaws[yaws > 0] = (
                    torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                    - math.pi
            )
            yaws[yaws < 0] = (
                    torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                    + math.pi
            )
            predicted_graph[:, [5, 6]] = yaws

            # Save prediction alongside true value (next time step state)
            y_hat[t - 10, :, :] = predicted_graph[:, : self.out_features]
            y_target[t - 10, :, :] = batch.x[:, t + 1, : self.out_features]

        fde_mask = mask[:, -1]
        val_mask = mask[:, 11:].permute(1, 0)

        # Compute and log loss
        fde_loss = self.val_fde_loss(
            y_hat[-1, fde_mask, :3], y_target[-1, fde_mask, :3]
        )
        ade_loss = self.val_ade_loss(
            y_hat[:, :, 0:3][val_mask], y_target[:, :, 0:3][val_mask]
        )
        vel_loss = self.val_vel_loss(
            y_hat[:, :, 3:5][val_mask], y_target[:, :, 3:5][val_mask]
        )
        yaw_loss = self.val_yaw_loss(
            y_hat[:, :, 5:7][val_mask], y_target[:, :, 5:7][val_mask]
        )

        # Compute losses on "tracks_to_predict"
        fde_ttp_mask = torch.logical_and(fde_mask, batch.tracks_to_predict)
        fde_ttp_loss = self.val_fde_ttp_loss(
            y_hat[-1, fde_ttp_mask, :3], y_target[-1, fde_ttp_mask, :3]
        )
        ade_ttp_mask = torch.logical_and(
            val_mask, batch.tracks_to_predict.expand((80, mask.size(0)))
        )
        ade_ttp_loss = self.val_ade_loss(
            y_hat[:, :, 0:3][ade_ttp_mask], y_target[:, :, 0:3][ade_ttp_mask]
        )

        ######################
        # Logging            #
        ######################

        self.log("val_ade_loss", ade_loss, batch_size=val_mask.sum().item())
        self.log("val_fde_loss", fde_loss, batch_size=fde_mask.sum().item())
        self.log("val_vel_loss", vel_loss, batch_size=val_mask.sum().item())
        self.log("val_yaw_loss", yaw_loss, batch_size=val_mask.sum().item())
        self.log("val_total_loss", (ade_loss + vel_loss + yaw_loss) / 3, batch_size=val_mask.sum().item())
        self.log("val_fde_ttp_loss", fde_ttp_loss, batch_size=fde_ttp_mask.sum().item())
        self.log("val_ade_ttp_loss", ade_ttp_loss, batch_size=ade_ttp_mask.sum().item())

        return (ade_loss + vel_loss + yaw_loss) / 3

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

        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((90, n_nodes, self.node_features))
        y_target = torch.zeros((90, n_nodes, self.node_features))
        # Ensure device placement
        y_hat = y_hat.type_as(batch.x)
        y_target = y_target.type_as(batch.x)

        batch.x = batch.x[:, :, :-1]
        # static_features = torch.cat(
        #     [batch.x[:, 10, self.out_features :], batch.type], dim=1
        # )
        static_features = batch.x[:, 10, self.out_features :].clone()
        edge_attr = None

        # Initial hidden state
        if self.rnn_type == "GRU":
            h = torch.zeros((self.model.num_layers, n_nodes, self.model.rnn_size))
            h = h.type_as(batch.x)
        elif self.rnn_type == "LSTM":
            h = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
            c = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
            h = (h, c)
        else:
            h = None

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            mask_t = mask[:, t]
            # x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            x_t = batch.x[mask_t, t, :].clone()
            batch_t = batch.batch[mask_t]

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2],
                    k=self.n_neighbours,
                    batch=batch_t,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch_t,
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
                edge_attr = (x_t[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Predictions 1/2    #
            ######################

            # Normalise input graph
            if self.normalise:
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x[:, [0, 1, 2]] -= batch.loc[batch.batch][mask_t][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if h is None:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                )
            else:
                delta_x, h_t = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=h[:, mask_t],
                )
                h[:, mask_t] = h_t

            # Process yaw values
            bbox_yaw = torch.atan2(
                torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
            ).unsqueeze(1)
            vel_yaw = torch.atan2(
                torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
            ).unsqueeze(1)
            tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
            delta_x = tmp

            # Add deltas to input graph
            predicted_graph = torch.cat(
                (
                    batch.x[mask_t, t, : self.out_features] + delta_x,
                    static_features[mask_t],
                ),
                dim=-1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Process yaw values to ensure [-pi, pi] interval
            yaws = predicted_graph[:, [5, 6]]
            yaws[yaws > 0] = (
                    torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                    - math.pi
            )
            yaws[yaws < 0] = (
                    torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                    + math.pi
            )
            predicted_graph[:, [5, 6]] = yaws

            # Save predictions and targets
            y_hat[t, mask_t, :] = predicted_graph
            # y_target[t, mask_t, :] = torch.cat(
            #     [batch.x[mask_t, t + 1, :], batch.type[mask_t]], dim=1
            # )
            y_target[t, mask_t, :] = batch.x[mask_t, t + 1, :].clone()

        ######################
        # Future             #
        ######################

        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            x_t = predicted_graph.clone()

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

            ######################
            # Predictions 2/2    #
            ######################

            # Normalise input graph
            if self.normalise:
                if edge_attr is None:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                else:
                    # Center node positions
                    x_t[:, [0, 1, 2]] -= batch.loc[batch.batch][:, [0, 1, 2]]
                    # Scale all features (except yaws) with global scaler
                    x_t[:, [0, 1, 2, 3, 4, 7, 8, 9]] /= self.global_scale
                    # Scale edge attributes
                    edge_attr /= self.global_scale

            # Obtain normalised predicted delta dynamics
            if h is None:
                delta_x = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )
            else:
                delta_x, h = self.model(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=h,
                )

            # Transform yaw
            bbox_yaw = torch.atan2(
                torch.tanh(delta_x[:, 5]), torch.tanh(delta_x[:, 6])
            ).unsqueeze(1)
            vel_yaw = torch.atan2(
                torch.tanh(delta_x[:, 7]), torch.tanh(delta_x[:, 8])
            ).unsqueeze(1)
            tmp = torch.cat([delta_x[:, 0:5], bbox_yaw, vel_yaw], dim=1)
            delta_x = tmp

            predicted_graph = torch.cat(
                (
                    predicted_graph[:, : self.out_features] + delta_x,
                    predicted_graph[:, self.out_features :],
                ),
                dim=-1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Process yaw values to ensure [-pi, pi] interval
            yaws = predicted_graph[:, [5, 6]]
            yaws[yaws > 0] = (
                    torch.fmod(yaws[yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                    - math.pi
            )
            yaws[yaws < 0] = (
                    torch.fmod(yaws[yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                    + math.pi
            )
            predicted_graph[:, [5, 6]] = yaws

            # Save prediction alongside true value (next time step state)
            y_hat[t, :, :] = predicted_graph
            # y_target[t, :, :] = torch.cat([batch.x[:, t + 1, :], batch.type], dim=1)
            y_target[t, :, :] = batch.x[:, t + 1, :].clone()

        return y_hat, y_target, mask

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class ConstantPhysicalBaselineModule(pl.LightningModule):
    def __init__(self, model=None, out_features: int = 6, **kwargs):
        super().__init__()
        assert model is None
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_yaw_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_fde_ttp_loss = torchmetrics.MeanSquaredError()
        self.val_ade_ttp_loss = torchmetrics.MeanSquaredError()

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
        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, self.out_features))
        y_target = torch.zeros((80, n_nodes, self.out_features))
        # Remove valid flag from features
        batch.x = batch.x[:, :, :-1]
        # Extract static features
        static_features = torch.cat(
            [batch.x[:, 10, self.out_features :], batch.type], dim=1
        )
        # Find valid agents at time t=11
        initial_mask = mask[:, 10]
        # Extract final dynamic states to use for predictions
        last_pos = batch.x[initial_mask, 10, 0:2]
        last_z = batch.x[initial_mask, 10, 2]
        last_vel = batch.x[initial_mask, 10, 3:5]
        last_yaw = batch.x[initial_mask, 10, 5:7]
        # Constant change in positions
        delta_pos = last_vel * 0.1
        # First updated position
        predicted_pos = last_pos + delta_pos
        predicted_graph = torch.cat(
            (predicted_pos, last_z.unsqueeze(1), last_vel, last_yaw, static_features),
            dim=1,
        )
        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph[:, : self.out_features]
        y_target[0, :, :] = batch.x[:, 11, : self.out_features]

        for t in range(11, 90):
            predicted_pos += delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_z.unsqueeze(1), last_vel, static_features), dim=1
            )
            y_hat[t - 10, :, :] = predicted_graph[:, : self.out_features]
            y_target[t - 10, :, :] = batch.x[:, t + 1, : self.out_features]

        fde_mask = mask[:, -1]
        val_mask = mask[:, 11:].permute(1, 0)

        # Compute and log loss
        fde_loss = self.val_fde_loss(
            y_hat[-1, fde_mask, :3], y_target[-1, fde_mask, :3]
        )
        ade_loss = self.val_ade_loss(
            y_hat[:, :, 0:3][val_mask], y_target[:, :, 0:3][val_mask]
        )
        vel_loss = self.val_vel_loss(
            y_hat[:, :, 3:5][val_mask], y_target[:, :, 3:5][val_mask]
        )
        yaw_loss = self.val_yaw_loss(
            y_hat[:, :, 5:7][val_mask], y_target[:, :, 5:7][val_mask]
        )

        # Compute losses on "tracks_to_predict"
        fde_ttp_mask = torch.logical_and(fde_mask, batch.tracks_to_predict)
        fde_ttp_loss = self.val_fde_ttp_loss(
            y_hat[-1, fde_ttp_mask, :3], y_target[-1, fde_ttp_mask, :3]
        )
        ade_ttp_mask = torch.logical_and(
            val_mask, batch.tracks_to_predict.expand((80, mask.size(0)))
        )
        ade_ttp_loss = self.val_ade_loss(
            y_hat[:, :, 0:3][ade_ttp_mask], y_target[:, :, 0:3][ade_ttp_mask]
        )

        ######################
        # Logging            #
        ######################

        self.log("val_ade_loss", ade_loss)
        self.log("val_fde_loss", fde_loss)
        self.log("val_vel_loss", vel_loss)
        self.log("val_yaw_loss", yaw_loss)
        self.log("val_total_loss", (ade_loss + vel_loss + yaw_loss) / 3)
        self.log("val_fde_ttp_loss", fde_ttp_loss)
        self.log("val_ade_ttp_loss", ade_ttp_loss)

        return (ade_loss + vel_loss + yaw_loss) / 3

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
        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((90, n_nodes, 15))
        y_target = torch.zeros((90, n_nodes, 15))
        # Remove valid flag from features
        batch.x = batch.x[:, :, :-1]
        # Extract static features
        static_features = torch.cat(
            [batch.x[:, 10, self.out_features :], batch.type], dim=1
        )

        # Fill in targets
        for t in range(0, 90):
            y_target[t, :, :] = torch.cat([batch.x[:, t + 1, :], batch.type], dim=1)

        for t in range(11):
            mask_t = mask[:, t]

            last_pos = batch.x[mask_t, t, 0:2]
            last_z = batch.x[mask_t, t, 2]
            last_vel = batch.x[mask_t, t, 3:5]
            last_yaw = batch.x[mask_t, t, 5:7]

            delta_pos = last_vel * 0.1
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat(
                (
                    predicted_pos,
                    last_z.unsqueeze(1),
                    last_vel,
                    last_yaw,
                    static_features[mask_t],
                ),
                dim=1,
            )
            y_hat[t, mask_t, :] = predicted_graph

        for t in range(11, 90):
            last_pos = predicted_pos
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat(
                (
                    predicted_pos,
                    last_z.unsqueeze(1),
                    last_vel,
                    last_yaw,
                    static_features,
                ),
                dim=1,
            )
            y_hat[t, :, :] = predicted_graph

        return y_hat, y_target, mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


@hydra.main(config_path="../../configs/waymo/", config_name="config")
def main(config):
    # Print configuration file
    print(OmegaConf.to_yaml(config))
    # Seed for reproducibility
    seed_everything(config["misc"]["seed"], workers=True)
    # Load data, model, and regressor
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])
    # Define model
    if config["misc"]["model_type"] != "ConstantModel":
        model_dict = config["model"]
        model_type = config["misc"]["model_type"]
    else:
        model_dict = None
        model_type = None

    # Define LightningModule
    regressor = eval(config["misc"]["regressor_type"])(model_type=model_type, model_dict=dict(model_dict),
                                                       **config["regressor"])

    # Setup logging
    wandb_logger = WandbLogger(
        entity="petergroth", config=dict(config), **config["logger"]
    )
    wandb_logger.watch(regressor, log_freq=config["misc"]["log_freq"], log_graph=False)
    # Add default dir for logs

    # Setup trainer
    if config["misc"]["checkpoint"]:
        # Setup callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_total_loss",
            dirpath="checkpoints",
            filename=config["logger"]["version"],
        )
        # Create trainer, fit, and validate
        trainer = pl.Trainer(
            logger=wandb_logger, **config["trainer"], callbacks=[checkpoint_callback]
        )
    else:
        # Create trainer, fit, and validate
        trainer = pl.Trainer(
            logger=wandb_logger, **config["trainer"] #, callbacks=[RichProgressBar()]
        )

    if config["misc"]["train"]:
        trainer.fit(model=regressor, datamodule=datamodule)

    trainer.validate(regressor, datamodule=datamodule)


if __name__ == "__main__":
    main()
