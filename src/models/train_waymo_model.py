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
import yaml


class OneStepModule(pl.LightningModule):
    def __init__(
            self,
            model_type,
            noise,
            lr: float,
            weight_decay: float,
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
            centered_edges: bool = False,
    ):
        super().__init__()
        # Verify inputs
        assert edge_type in ["knn", "distance"]
        if edge_type == "distance":
            assert min_dist > 0.0

        # Setup metrics
        self.train_pos_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        self.train_yaw_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_yaw_loss = torchmetrics.MeanSquaredError()
        self.val_fde_ttp_loss = torchmetrics.MeanSquaredError()
        self.val_ade_ttp_loss = torchmetrics.MeanSquaredError()

        # Save parameters
        self.normalise = normalise
        self.model = model_type
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay
        self.edge_type = edge_type
        self.min_dist = min_dist
        self.fully_connected = fully_connected
        self.n_neighbours = 128 if fully_connected else n_neighbours
        self.edge_weight = edge_weight
        self.self_loop = self_loop
        self.undirected = undirected
        self.out_features = out_features
        self.edge_features = edge_features
        self.centered_edges = centered_edges
        self.node_features = node_features

        self.save_hyperparameters()
        node_features -= 5
        # Normalisation parameters
        self.register_buffer("node_counter", torch.zeros(1))
        self.register_buffer("node_in_sum", torch.zeros(node_features))
        self.register_buffer("node_in_squaresum", torch.zeros(node_features))
        self.register_buffer("node_in_std", torch.ones(node_features))
        self.register_buffer("node_in_mean", torch.zeros(node_features))
        # Output normalisation parameters
        self.register_buffer("node_out_sum", torch.zeros(out_features))
        self.register_buffer("node_out_squaresum", torch.zeros(out_features))
        self.register_buffer("node_out_std", torch.ones(out_features))
        self.register_buffer("node_out_mean", torch.zeros(out_features))
        # Edge normalisation parameters
        self.register_buffer("edge_counter", torch.zeros(1))
        self.register_buffer("edge_in_sum", torch.zeros(edge_features))
        self.register_buffer("edge_in_squaresum", torch.zeros(edge_features))
        self.register_buffer("edge_in_std", torch.ones(edge_features))
        self.register_buffer("edge_in_mean", torch.zeros(edge_features))
        self.register_buffer("edge_in_mean", torch.zeros(edge_features))

    def training_step(self, batch: Batch, batch_idx: int):
        # Extract node features
        x = batch.x
        # One-hot encode type and concatenate with feature matrix
        type = one_hot(batch.type, num_classes=5)
        type = type.type_as(batch.x)
        x = torch.cat([x, type], dim=1)
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
                max_num_neighbors=128,
                flow="source_to_target",
            )

        if self.undirected:
            edge_index, _ = torch_geometric.utils.to_undirected(edge_index)

        # Remove duplicates and sort
        edge_index = torch_geometric.utils.coalesce(edge_index)

        # Determine whether to add random noise to dynamic states
        if self.noise is not None:
            x[:, : self.out_features] += self.noise * torch.randn_like(
                x[:, : self.out_features].detach()
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

        # Update normalisation state and normalise
        if edge_attr is None:
            self.update_in_normalisation(x.detach().clone())
        else:
            self.update_in_normalisation(x.detach().clone(), edge_attr.detach().clone())
        self.update_out_normalisation(y_target.detach().clone())

        # Obtain normalised input graph # and normalised target nodes
        x_nrm, edge_attr_nrm = self.in_normalise(x.detach(), edge_attr.detach())
        y_target_nrm = self.out_normalise(y_target.detach())

        # Obtain normalised predicted delta dynamics
        y_hat = self.model(
            x=x_nrm, edge_index=edge_index, edge_attr=edge_attr_nrm, batch=batch.batch
        )

        # Compute and log loss
        pos_loss = self.train_pos_loss(y_hat[:, :3], y_target_nrm[:, :3])
        vel_loss = self.train_pos_loss(y_hat[:, 3:5], y_target_nrm[:, 3:5])
        yaw_loss = self.train_pos_loss(y_hat[:, 5:7], y_target_nrm[:, 5:7])

        # Compute new positions using old velocities
        pos_expected = x_nrm[:, :2] + 0.1 * x_nrm[:, 3:5]
        pos_new = y_hat[:, :2] + x_nrm[:, :2]
        pos_diff = torch.linalg.norm(pos_expected - pos_new)

        self.log("train_pos_loss", pos_loss, on_step=True, on_epoch=True)
        self.log("train_vel_loss", vel_loss, on_step=True, on_epoch=True)
        self.log("train_yaw_loss", yaw_loss, on_step=True, on_epoch=True)
        self.log("position_difference", pos_diff, on_step=True, on_epoch=True)

        loss = pos_loss + vel_loss + yaw_loss + pos_diff
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
        batch.type = one_hot(batch.type[valid_mask], num_classes=5)
        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, self.out_features))
        y_hat = y_hat.type_as(batch.x)
        y_target = torch.zeros((80, n_nodes, self.out_features))
        y_target = y_target.type_as(batch.x)

        # Discard mask from features and extract static features
        batch.x = batch.x[:, :, :-1]
        static_features = torch.cat(
            [batch.x[:, 10, self.out_features:], batch.type], dim=1
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
            x = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            batch_t = batch.batch[mask_t]

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=self.n_neighbours, batch=batch_t, loop=self.self_loop
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch_t,
                    loop=self.self_loop,
                    max_num_neighbors=128,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, _ = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Validation 1/2     #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_t
            )
            # Renormalise output dynamics
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (batch.x[mask_t, t, : self.out_features] + x, static_features[mask_t]),
                dim=-1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

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
            x = predicted_graph

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=128,
                    flow="source_to_target",
                )

            if self.undirected:
                edge_index, _ = torch_geometric.utils.to_undirected(edge_index)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Create edge_attr if specified
            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

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
                (predicted_graph[:, : self.out_features] + x, static_features), dim=-1
            )
            predicted_graph = predicted_graph.type_as(batch.x)

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
        # types = batch.type[valid_mask]
        batch.type = one_hot(batch.type[valid_mask], num_classes=5)
        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((90, n_nodes, self.node_features))
        y_hat = y_hat.type_as(batch.x)
        y_target = torch.zeros((90, n_nodes, self.node_features))
        y_target = y_target.type_as(batch.x)

        batch.x = batch.x[:, :, :-1]
        static_features = torch.cat(
            [batch.x[:, 10, self.out_features:], batch.type], dim=1
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
            x = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            batch_t = batch.batch[mask_t]

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=self.n_neighbours, batch=batch_t, loop=self.self_loop
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch_t,
                    loop=self.self_loop,
                    max_num_neighbors=128,
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
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Prediction 1/2     #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_t
            )
            # Renormalise output dynamics
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (batch.x[mask_t, t, : self.out_features] + x, static_features[mask_t]),
                dim=-1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Save first prediction and target
            y_hat[t, mask_t, :] = predicted_graph
            y_target[t, mask_t, :] = torch.cat([batch.x[mask_t, t + 1, :], batch.type[mask_t]], dim=-1)

        ######################
        # Future             #
        ######################

        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            # Latest prediction as input
            x = predicted_graph

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=128,
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
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Prediction 2/2     #
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
                (predicted_graph[:, : self.out_features] + x, static_features), dim=-1
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Save prediction alongside true value (next time step state)
            y_hat[t, :, :] = predicted_graph
            y_target[t, :, :] = torch.cat([batch.x[:, t + 1], batch.type], dim=-1)

        return y_hat, y_target, mask

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def update_in_normalisation(self, x, edge_attr=None):
        if self.normalise:
            x = x[:, : (self.node_features - 5)]
            # Node normalisation
            tmp = torch.sum(x, dim=0)
            tmp = tmp.type_as(x)
            self.node_in_sum += tmp
            self.node_in_squaresum += tmp * tmp
            self.node_counter += x.size(0)
            self.register_buffer("node_in_mean", self.node_in_sum / self.node_counter)
            self.register_buffer(
                "node_in_std",
                torch.sqrt(
                    (
                            (self.node_in_squaresum / self.node_counter)
                            - (self.node_in_sum / self.node_counter) ** 2
                    )
                ),
            )

            # Edge normalisation
            if edge_attr is not None:
                tmp = torch.sum(edge_attr, dim=0)
                tmp = tmp.type_as(x)
                self.edge_in_sum += tmp
                self.edge_in_squaresum += tmp * tmp
                self.edge_counter += edge_attr.size(0)
                self.register_buffer(
                    "edge_in_mean", self.edge_in_sum / self.edge_counter
                )
                self.register_buffer(
                    "edge_in_std",
                    torch.sqrt(
                        (
                                (self.edge_in_squaresum / self.edge_counter)
                                - (self.edge_in_sum / self.edge_counter) ** 2
                        )
                    ),
                )

    def in_normalise(self, x, edge_attr=None):
        if self.normalise:
            type = x[:, (self.node_features - 5):]
            x = x[:, : (self.node_features - 5)]

            x = torch.sub(x, self.node_in_mean)
            x = torch.div(x, self.node_in_std)

            # Edge normalisation
            if edge_attr is not None:
                if self.centered_edges:
                    edge_attr = torch.sub(edge_attr, self.edge_in_mean)
                edge_attr = torch.div(edge_attr, self.edge_in_std)
            x = torch.cat([x, type], dim=1)
        return x, edge_attr

    def update_out_normalisation(self, x):
        if self.normalise:
            tmp = torch.sum(x, dim=0)
            self.node_out_sum += tmp
            self.node_out_squaresum += tmp * tmp
            self.register_buffer("node_out_mean", self.node_out_sum / self.node_counter)
            self.register_buffer(
                "node_out_std",
                torch.sqrt(
                    (
                            (self.node_out_squaresum / self.node_counter)
                            - (self.node_out_sum / self.node_counter) ** 2
                    )
                ),
            )

    def out_normalise(self, x):
        if self.normalise:
            x = torch.sub(x, self.node_out_mean)
            x = torch.div(x, self.node_out_std)
        return x

    def out_renormalise(self, x):
        if self.normalise:
            type = x[:, (self.node_features - 5):]
            x = x[:, : (self.node_features - 5)]
            x = torch.mul(self.node_out_std, x)
            x = torch.add(x, self.node_out_mean)
            x = torch.cat([x, type], dim=1)
        return x


class SequentialModule(pl.LightningModule):
    def __init__(
            self,
            model_type,
            lr: float,
            weight_decay: float,
            noise=None,
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
            centered_edges: bool = False,
            normalise: bool = True,
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

        # Save parameters
        self.normalise = normalise
        self.model = model_type
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay
        self.edge_type = edge_type
        self.min_dist = min_dist
        self.fully_connected = fully_connected
        self.n_neighbours = 128 if fully_connected else n_neighbours
        self.edge_weight = edge_weight
        self.self_loop = self_loop
        self.undirected = undirected
        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.rnn_type = rnn_type
        self.out_features = out_features
        self.edge_features = edge_features
        self.centered_edges = centered_edges
        self.node_features = node_features

        self.save_hyperparameters()
        node_features -= 5
        # Normalisation parameters
        self.register_buffer("node_counter", torch.zeros(1))
        self.register_buffer("node_in_sum", torch.zeros(node_features))
        self.register_buffer("node_in_squaresum", torch.zeros(node_features))
        self.register_buffer("node_in_std", torch.ones(node_features))
        self.register_buffer("node_in_mean", torch.zeros(node_features))
        # Output normalisation parameters
        self.register_buffer("node_out_sum", torch.zeros(out_features))
        self.register_buffer("node_out_squaresum", torch.zeros(out_features))
        self.register_buffer("node_out_std", torch.ones(out_features))
        self.register_buffer("node_out_mean", torch.zeros(out_features))
        # Edge normalisation parameters
        self.register_buffer("edge_counter", torch.zeros(1))
        self.register_buffer("edge_in_sum", torch.zeros(edge_features))
        self.register_buffer("edge_in_squaresum", torch.zeros(edge_features))
        self.register_buffer("edge_in_std", torch.ones(edge_features))
        self.register_buffer("edge_in_mean", torch.zeros(edge_features))

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
        batch.type = one_hot(batch.type[valid_mask], num_classes=5)
        # Update mask
        mask = batch.x[:, :, -1].bool()
        batch.x = batch.x[:, :, :-1]
        static_features = torch.cat(
            [batch.x[:, 10, self.out_features:], batch.type], dim=1
        )
        edge_attr = None

        # Extract dimensions
        n_nodes = batch.num_nodes
        y_predictions = torch.zeros((n_nodes, 90, self.out_features))
        y_predictions = y_predictions.type_as(batch.x)

        # Obtain target delta dynamic nodes
        # Use torch.roll to compute differences between x_t and x_{t+1}.
        # Ignore final difference (between t_0 and t_{-1})
        y_target_abs = (
                               torch.roll(batch.x[:, :, : self.out_features], (0, -1, 0), (0, 1, 2))
                               - batch.x[:, :, : self.out_features]
                       )[:, :-1, :]
        y_target_abs = y_target_abs.type_as(batch.x)
        y_target_nrm = torch.zeros_like(y_target_abs)
        y_target_nrm = y_target_nrm.type_as(batch.x)

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
            # Extract current input and target
            mask_t = mask[:, t]
            x_t = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            y_t = y_target_abs[:, t, :]
            x_t = x_t.type_as(batch.x)
            y_t = y_t.type_as(batch.x)
            # Add noise if specified
            if self.noise is not None:
                x_t[:, : self.out_features] += self.noise * torch.rand_like(
                    x_t[:, : self.out_features].detach()
                )

            ######################
            # Graph construction #
            ######################

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2], k=self.n_neighbours, batch=batch.batch[mask_t], loop=self.self_loop
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch[mask_t],
                    loop=self.self_loop,
                    max_num_neighbors=128,
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

            if edge_attr is None:
                self.update_in_normalisation(x_t.detach().clone())
            else:
                self.update_in_normalisation(x_t.detach().clone(), edge_attr.detach().clone())
            self.update_out_normalisation(y_t.detach().clone())

            # Obtain normalised input graph and normalised target nodes
            x_t_nrm, edge_attr_nrm = self.in_normalise(x_t, edge_attr)
            y_t_nrm = self.out_normalise(y_t)
            # x_t_nrm, edge_attr_nrm, y_t_nrm = x_t, edge_attr, y_t
            y_target_nrm[:, t, :] = y_t_nrm
            # Obtain normalised predicted delta dynamics
            if h is None:
                y_pred_t = self.model(
                    x=x_t_nrm,
                    edge_index=edge_index,
                    edge_attr=edge_attr_nrm,
                    batch=batch.batch[mask_t],
                )
            else:
                # Add zero rows for new columns
                assert self.rnn_type == 'GRU'
                y_pred_t, h_t = self.model(
                    x=x_t_nrm,
                    edge_index=edge_index,
                    edge_attr=edge_attr_nrm,
                    batch=batch.batch[mask_t],
                    hidden=h[:, mask_t],
                )
                h[:, mask_t] = h_t

            # Save deltas for loss computation
            y_predictions[mask_t, t, :] = y_pred_t
            # Renormalise output dynamics
            y_pred_abs = self.out_renormalise(y_pred_t)
            # Add deltas to input graph
            x_t = torch.cat(
                (batch.x[mask_t, t, : self.out_features] + y_pred_abs, static_features[mask_t]), dim=-1
            )

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

        for t in range(11, 90):
            # Use groundtruth 'teacher_forcing_ratio' % of the time
            if use_groundtruth:
                x_t = torch.cat([batch.x[:, t, :], batch.type], dim=1)
            x_prev = x_t
            y_t = y_target_abs[:, t, :]

            ######################
            # Graph construction #
            ######################

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2], k=self.n_neighbours, batch=batch.batch, loop=self.self_loop
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=128,
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

            if edge_attr is None:
                self.update_in_normalisation(x_t.detach().clone())
            else:
                self.update_in_normalisation(x_t.detach().clone(), edge_attr.detach().clone())
            self.update_out_normalisation(y_t.detach().clone())

            # Obtain normalised input graph and normalised target nodes
            x_t_nrm, edge_attr_nrm = self.in_normalise(x_t, edge_attr)
            y_t_nrm = self.out_normalise(y_t)
            y_target_nrm[:, t, :] = y_t_nrm
            # Obtain normalised predicted delta dynamics
            if h is None:
                y_pred_t = self.model(
                    x=x_t_nrm,
                    edge_index=edge_index,
                    edge_attr=edge_attr_nrm,
                    batch=batch.batch,
                )
            else:
                y_pred_t, h = self.model(
                    x=x_t_nrm,
                    edge_index=edge_index,
                    edge_attr=edge_attr_nrm,
                    batch=batch.batch,
                    hidden=h,
                )

            # Save deltas for loss computation
            y_predictions[:, t, :] = y_pred_t
            # Renormalise output dynamics
            y_pred_abs = self.out_renormalise(y_pred_t)
            # Add deltas to input graph. Input for next timestep
            x_t = torch.cat(
                (x_prev[:, : self.out_features] + y_pred_abs, static_features), dim=-1
            )

        loss_mask = mask[:, :-1]
        fde_mask = mask[:, -2]

        # Compute and log loss
        fde_loss = self.train_fde_loss(
            y_predictions[fde_mask, -1, :3], y_target_nrm[fde_mask, -1, :3]
        )
        ade_loss = self.train_ade_loss(y_predictions[:, :, :3][loss_mask], y_target_nrm[:, :, :3][loss_mask])
        vel_loss = self.train_vel_loss(
            y_predictions[:, :, 3:5][loss_mask], y_target_nrm[:, :, 3:5][loss_mask]
        )
        yaw_loss = self.train_yaw_loss(
            y_predictions[:, :, 5:7][loss_mask], y_target_nrm[:, :, 5:7][loss_mask]
        )

        self.log("train_fde_loss", fde_loss, on_step=True, on_epoch=True)
        self.log("train_ade_loss", ade_loss, on_step=True, on_epoch=True)
        self.log("train_vel_loss", vel_loss, on_step=True, on_epoch=True)
        self.log("train_yaw_loss", yaw_loss, on_step=True, on_epoch=True)
        loss = ade_loss + vel_loss + yaw_loss
        self.log(
            "train_total_loss",
            loss,
            on_step=True,
            on_epoch=True,
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
        batch.type = one_hot(batch.type[valid_mask], num_classes=5)
        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, self.out_features))
        y_hat = y_hat.type_as(batch.x)
        y_target = torch.zeros((80, n_nodes, self.out_features))
        y_target = y_target.type_as(batch.x)
        batch.x = batch.x[:, :, :-1]
        static_features = torch.cat(
            [batch.x[:, 10, self.out_features:], batch.type], dim=1
        )
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
            x = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=self.n_neighbours, batch=batch.batch[mask_t], loop=self.self_loop
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch[mask_t],
                    loop=self.self_loop,
                    max_num_neighbors=128,
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
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Validation 1/2     #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            if h is None:
                x = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                )
            else:
                x, h_t = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=h[:, mask_t],
                )
                h[:, mask_t] = h_t

            # Renormalise output dynamics
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (batch.x[mask_t, t, : self.out_features] + x, static_features[mask_t]),
                dim=-1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)

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
            x = predicted_graph

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=128,
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
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Validation 2/2     #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            if h is None:
                x = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )
            else:
                x, h = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=h,
                )
            # Renormalise deltas
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (predicted_graph[:, : self.out_features] + x, static_features), dim=-1
            )
            predicted_graph = predicted_graph.type_as(batch.x)

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
        raise NotImplementedError
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
        batch.type = one_hot(batch.type[valid_mask], num_classes=5)
        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((90, n_nodes, self.node_features))
        y_hat = y_hat.type_as(batch.x)
        y_target = torch.zeros((90, n_nodes, self.node_features))
        y_target = y_target.type_as(batch.x)
        batch.x = batch.x[:, :, :-1]
        static_features = torch.cat(
            [batch.x[:, 10, self.out_features:], batch.type], dim=1
        )
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
            x = torch.cat([batch.x[mask_t, t, :], batch.type[mask_t]], dim=1)
            batch_t = batch.batch[mask_t]

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=self.n_neighbours, batch=batch_t, loop=self.self_loop
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch_t,
                    loop=self.self_loop,
                    max_num_neighbors=128,
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
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Predictions 1/2    #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            if h is None:
                x = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                )
            else:
                x, h_t = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch[mask_t],
                    hidden=h[:, mask_t],
                )
                h[:, mask_t] = h_t

            # Renormalise output dynamics
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (batch.x[mask_t, t, : self.out_features] + x, static_features[mask_t]),
                dim=-1,
            )
            predicted_graph = predicted_graph.type_as(batch.x)
            # Save predictions and targets
            y_hat[t, mask_t, :] = predicted_graph
            y_target[t, mask_t, :] = torch.cat([batch.x[mask_t, t + 1, :], batch.type[mask_t]], dim=1)

        ######################
        # Future             #
        ######################

        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            x = predicted_graph

            # Construct edges
            if self.edge_type == "knn":
                # Neighbour-based graph
                edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2],
                    k=self.n_neighbours,
                    batch=batch.batch,
                    loop=self.self_loop,
                )
            else:
                # Distance-based graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=128,
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
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)
                edge_attr = edge_attr.type_as(batch.x)

            ######################
            # Predictions 2/2    #
            ######################

            # Normalise input graph
            x, edge_attr = self.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            if h is None:
                x = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                )
            else:
                x, h = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch.batch,
                    hidden=h,
                )
            # Renormalise deltas
            x = self.out_renormalise(x)
            predicted_graph = torch.cat(
                (predicted_graph[:, : self.out_features] + x, static_features), dim=-1
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Save prediction alongside true value (next time step state)
            y_hat[t, :, :] = predicted_graph
            y_target[t, :, :] = torch.cat([batch.x[:, t + 1, :], batch.type], dim=1)

        return y_hat, y_target, mask

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def update_in_normalisation(self, x, edge_attr=None):
        if self.normalise:
            x = x[:, : (self.node_features - 5)]
            # Node normalisation
            tmp = torch.sum(x, dim=0)
            tmp = tmp.type_as(x)
            self.node_in_sum += tmp
            self.node_in_squaresum += tmp * tmp
            self.node_counter += x.size(0)
            self.register_buffer("node_in_mean", self.node_in_sum / self.node_counter)
            self.register_buffer(
                "node_in_std",
                torch.sqrt(
                    (
                            (self.node_in_squaresum / self.node_counter)
                            - (self.node_in_sum / self.node_counter) ** 2
                    )
                ),
            )

            # Edge normalisation
            if edge_attr is not None:
                tmp = torch.sum(edge_attr, dim=0)
                tmp = tmp.type_as(x)
                self.edge_in_sum += tmp
                self.edge_in_squaresum += tmp * tmp
                self.edge_counter += edge_attr.size(0)
                self.register_buffer(
                    "edge_in_mean", self.edge_in_sum / self.edge_counter
                )
                self.register_buffer(
                    "edge_in_std",
                    torch.sqrt(
                        (
                                (self.edge_in_squaresum / self.edge_counter)
                                - (self.edge_in_sum / self.edge_counter) ** 2
                        )
                    ),
                )

    def in_normalise(self, x, edge_attr=None):
        if self.normalise:
            type = x[:, (self.node_features - 5):]
            x = x[:, : (self.node_features - 5)]

            x = torch.sub(x, self.node_in_mean)
            x = torch.div(x, self.node_in_std)

            # Edge normalisation
            if edge_attr is not None:
                if self.centered_edges:
                    edge_attr = torch.sub(edge_attr, self.edge_in_mean)
                edge_attr = torch.div(edge_attr, self.edge_in_std)
            x = torch.cat([x, type], dim=1)
        return x, edge_attr

    def update_out_normalisation(self, x):
        if self.normalise:
            tmp = torch.sum(x, dim=0)
            self.node_out_sum += tmp
            self.node_out_squaresum += tmp * tmp
            self.register_buffer("node_out_mean", self.node_out_sum / self.node_counter)
            self.register_buffer(
                "node_out_std",
                torch.sqrt(
                    (
                            (self.node_out_squaresum / self.node_counter)
                            - (self.node_out_sum / self.node_counter) ** 2
                    )
                ),
            )

    def out_normalise(self, x):
        if self.normalise:
            x = torch.sub(x, self.node_out_mean)
            x = torch.div(x, self.node_out_std)
        return x

    def out_renormalise(self, x):
        if self.normalise:
            type = x[:, (self.node_features - 5):]
            x = x[:, : (self.node_features - 5)]
            x = torch.mul(self.node_out_std, x)
            x = torch.add(x, self.node_out_mean)
            x = torch.cat([x, type], dim=1)
        return x


class ConstantBaselineModule(pl.LightningModule):
    def __init__(self, out_features: int = 6, **kwargs):
        super().__init__()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_yaw_loss = torchmetrics.MeanSquaredError()
        self.out_features = out_features
        self.save_hyperparameters()

    def training_step(self, batch: Batch, batch_idx: int):
        pass

    def validation_step(self, batch: Batch, batch_idx: int):
        # Validate on sequential dataset. First 11 observations are used to prime the model.
        # Loss is computed on remaining 80 samples using rollout.

        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, self.out_features))
        y_target = torch.zeros((80, n_nodes, self.out_features))

        last_input = batch.x[:, 10, : self.out_features]
        last_delta = (
                batch.x[:, 10, : self.out_features] - batch.x[:, 9, : self.out_features]
        )
        predicted_graph = last_input + last_delta
        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph
        y_target[0, :, :] = batch.x[:, 11, : self.out_features]

        # 1 prediction done, 79 remaining
        for t in range(11, 90):
            predicted_graph += last_delta
            y_hat[t - 10, :, :] = predicted_graph
            print(predicted_graph[0, :])
            y_target[t - 10, :, :] = batch.x[:, t + 1, :4]

        # Compute and log loss
        fde_loss = self.val_fde_loss(y_hat[-1, :, :], y_target[-1, :, :])
        ade_loss = self.val_ade_loss(y_hat, y_target)

        self.log("val_ade_loss", ade_loss)
        self.log("val_fde_loss", fde_loss)

        return ade_loss

    def predict_step(self, batch, batch_idx=None):
        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        n_features = 5
        y_hat = torch.zeros((90, n_nodes, n_features))
        y_target = torch.zeros((90, n_nodes, n_features))

        # Fill in targets
        for t in range(0, 90):
            y_target[t, :, :] = batch.x[:, t + 1, :]

        # First prediction is same as first input due to lack of history
        y_hat[0, :, :] = batch.x[:, 0, :]
        # Fill in next 10 values with previous inputs
        for t in range(1, 11):
            y_hat[t, :, :] = batch.x[:, t, :] + (
                    batch.x[:, t, :] - batch.x[:, t - 1, :]
            )

        # Constant change
        constant_delta = batch.x[:, 10, :] - batch.x[:, 9, :]
        predicted_graph = batch.x[:, 10, :] + constant_delta
        y_hat[11, :, :] = predicted_graph

        for t in range(12, 90):
            predicted_graph += constant_delta
            y_hat[t, :, :] = predicted_graph

        return y_hat, y_target

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
        batch.type = one_hot(batch.type[valid_mask], num_classes=5)
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
            [batch.x[:, 10, self.out_features:], batch.type], dim=1
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
            (predicted_pos, last_z.unsqueeze(1), last_vel, last_yaw, static_features), dim=1
        )
        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph[:, : self.out_features]
        y_target[0, :, :] = batch.x[:, 11, : self.out_features]

        # 1 prediction done, 79 remaining
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
        batch.type = one_hot(batch.type[valid_mask], num_classes=5)
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
            [batch.x[:, 10, self.out_features:], batch.type], dim=1
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
                (predicted_pos, last_z.unsqueeze(1), last_vel, last_yaw, static_features[mask_t]), dim=1
            )
            y_hat[t, mask_t, :] = predicted_graph

        for t in range(11, 90):
            last_pos = predicted_pos
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_z.unsqueeze(1), last_vel, last_yaw, static_features), dim=1
            )
            y_hat[t, :, :] = predicted_graph

        return y_hat, y_target, mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    # torch.autograd.set_detect_anomaly(True)
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # Seed for reproducibility
    seed_everything(config["misc"]["seed"], workers=True)
    # Load data, model, and regressor
    datamodule = eval(config["misc"]["dm_type"])(**config["datamodule"])

    # Define model
    if config["misc"]["model_type"] != "ConstantModel":
        model = eval(config["misc"]["model_type"])(**config["model"])
    else:
        model = None

    # Define LightningModule
    regressor = eval(config["misc"]["regressor_type"])(model, **config["regressor"])

    # Setup logging
    wandb_logger = WandbLogger(entity="petergroth", config=config, **config["logger"])
    wandb_logger.watch(regressor, log_freq=100)
    # Add default dir for logs
    config["trainer"]["default_root_dir"] = "logs"

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
        trainer = pl.Trainer(logger=wandb_logger, **config["trainer"])

    if config["misc"]["continue_training"] is not None:
        raise NotImplementedError
        # Setup callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_ade_loss",
            dirpath="checkpoints",
            filename=config["logger"]["version"],
        )
        trainer = pl.Trainer(
            logger=wandb_logger,
            **config["trainer"],
            resume_from_checkpoint=config["misc"]["continue_training"],
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model=regressor, datamodule=datamodule)

    elif config["misc"]["train"]:
        trainer.fit(model=regressor, datamodule=datamodule)

    trainer.validate(regressor, datamodule=datamodule)

    # loader = datamodule.train_dataloader()
    # batch = next(iter(loader))
    # y_hat, y_target, mask = regressor.predict_step(batch)
    #
    # with torch.no_grad():
    #     for t in range(90):
    #         print(y_target[t, 2, :2])
    #         print(y_hat[t, 2, :2])