import argparse
import pytorch_lightning as pl
import torch
import torch_geometric.nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from src.data.dataset import OneStepWaymoDataModule, SequentialWaymoDataModule
import torchmetrics
from torch_geometric.data import Batch, Data
from src.models.model import *
# from src.utils import generate_fully_connected_edges
import yaml

class OneStepModule(pl.LightningModule):
    def __init__(
        self,
        model_type,
        noise,
        lr: float,
        weight_decay: float,
        edge_type: str = 'knn',
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
        if edge_type == 'distance':
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
        self.n_neighbours = 127 if fully_connected else n_neighbours
        self.edge_weight = edge_weight
        self.self_loop = self_loop
        self.undirected = undirected
        self.out_features = out_features
        self.edge_features = edge_features
        self.centered_edges = centered_edges

        self.save_hyperparameters()

        # Normalisation parameters
        self.register_buffer('node_counter', torch.zeros(1))
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

        # self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

        if self.undirected:
            edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index)

        # Remove duplicates and sort
        edge_index = torch_geometric.utils.coalesce(edge_index)

        # Determine whether to add random noise to dynamic states
        if self.noise is not None:
            x[:, :self.out_features] += self.noise*torch.rand_like(x[:, :self.out_features])

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
        y_target = batch.y[:, :self.out_features] - x[:, :self.out_features]
        y_target = y_target.type_as(batch.x)

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
        vel_loss = self.train_pos_loss(y_hat[:, 2:4], y_target_nrm[:, 2:4])
        yaw_loss = self.train_pos_loss(y_hat[:, 4:6], y_target_nrm[:, 4:6])

        self.log("train_pos_loss", pos_loss, on_step=True, on_epoch=True)
        self.log("train_vel_loss", vel_loss, on_step=True, on_epoch=True)
        self.log("train_yaw_loss", vel_loss, on_step=True, on_epoch=True)
        self.log(
            "train_total_loss", (pos_loss + vel_loss + yaw_loss) / 3, on_step=True, on_epoch=True
        )

        return pos_loss + vel_loss + yaw_loss

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
        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, self.out_features))
        y_hat = y_hat.type_as(batch.x)
        y_target = torch.zeros((80, n_nodes, self.out_features))
        y_target = y_target.type_as(batch.x)

        batch.x = batch.x[:, :, :-1]
        static_features = batch.x[:, 0, self.out_features:]
        edge_attr = None

        ######################
        # History            #
        ######################

        for t in range(11):

            ######################
            # Graph construction #
            ######################

            mask_t = mask[:, t]
            x = batch.x[mask_t, t, :]
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
                (batch.x[mask_t, t, :self.out_features] + x, static_features[mask_t]), dim=-1
            )
            predicted_graph = predicted_graph.type_as(batch.x)

        # Save first prediction and target
        y_hat[0, mask_t, :] = predicted_graph[:, :self.out_features]
        y_target[0, mask_t, :] = batch.x[mask_t, 11, :self.out_features]

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
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise deltas
            x = self.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (predicted_graph[:, :self.out_features] + x, static_features), dim=-1
            )
            predicted_graph = predicted_graph.type_as(batch.x)

            # Save prediction alongside true value (next time step state)
            y_hat[t - 10, :, :] = predicted_graph[:, :self.out_features]
            y_target[t - 10, :, :] = batch.x[:, t + 1, :self.out_features]

        fde_mask = mask[:, -1]
        val_mask = mask[:, 11:].permute(1, 0)

        # Compute and log loss
        fde_loss = self.val_fde_loss(y_hat[-1, fde_mask, :2], y_target[-1, fde_mask, :2])
        ade_loss = self.val_ade_loss(y_hat[:, :, 0:2][val_mask], y_target[:, :, 0:2][val_mask])
        vel_loss = self.val_vel_loss(y_hat[:, :, 2:4][val_mask], y_target[:, :, 2:4][val_mask])
        yaw_loss = self.val_yaw_loss(y_hat[:, :, 4:6][val_mask], y_target[:, :, 4:6][val_mask])

        # Compute losses on "tracks_to_predict"
        fde_ttp_mask = torch.logical_and(fde_mask, batch.tracks_to_predict)
        fde_ttp_loss = self.val_fde_ttp_loss(y_hat[-1, fde_ttp_mask, :2], y_target[-1, fde_ttp_mask, :2])
        ade_ttp_mask = torch.logical_and(val_mask, batch.tracks_to_predict.expand((80, mask.size(0))))
        ade_ttp_loss = self.val_ade_loss(y_hat[:, :, 0:2][ade_ttp_mask], y_target[:, :, 0:2][ade_ttp_mask])

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
            x, edge_attr = self.model.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise output dynamics
            x = self.model.out_renormalise(x)
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
            x, edge_attr = self.model.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise deltas
            x = self.model.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (predicted_graph[:, :4] + x, static_features), dim=-1
            )  # [n_nodes, n_features]

            # Save prediction alongside true value (next time step state)
            y_hat[t, :, :] = predicted_graph[:, :]
            y_target[t, :, :] = batch.x[:, t + 1, :]

        return y_hat, y_target

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def update_in_normalisation(self, x, edge_attr=None):
        if self.normalise:
            # Node normalisation
            tmp = torch.sum(x, dim=0)
            tmp = tmp.type_as(x)
            self.node_in_sum += tmp
            self.node_in_squaresum += tmp * tmp
            self.node_counter += x.size(0)
            self.register_buffer("node_in_mean", self.node_in_sum / self.node_counter)
            self.register_buffer("node_in_std", torch.sqrt(
                                                            (
                                                                (self.node_in_squaresum / self.node_counter)
                                                                - (self.node_in_sum / self.node_counter) ** 2
                                                            )
                                                           )
                                )

            # Edge normalisation
            if edge_attr is not None:
                tmp = torch.sum(edge_attr, dim=0)
                tmp = tmp.type_as(x)
                self.edge_in_sum += tmp
                self.edge_in_squaresum += tmp * tmp
                self.edge_counter += edge_attr.size(0)
                self.register_buffer("edge_in_mean", self.edge_in_sum / self.edge_counter)
                self.register_buffer("edge_in_std", torch.sqrt(
                    (
                        (self.edge_in_squaresum / self.edge_counter)
                        - (self.edge_in_sum / self.edge_counter) ** 2
                    )
                ))

    def in_normalise(self, x, edge_attr=None):
        if self.normalise:
            x = torch.sub(x, self.node_in_mean)
            x = torch.div(x, self.node_in_std)

            # Edge normalisation
            if edge_attr is not None:
                if self.centered_edges:
                    edge_attr = torch.sub(edge_attr, self.edge_in_mean)
                edge_attr = torch.div(edge_attr, self.edge_in_std)

        return x, edge_attr

    def update_out_normalisation(self, x):
        if self.normalise:
            tmp = torch.sum(x, dim=0)
            self.node_out_sum += tmp
            self.node_out_squaresum += tmp * tmp
            self.register_buffer("node_out_mean", self.node_out_sum / self.node_counter)
            self.register_buffer("node_out_std", torch.sqrt(
                (
                    (self.node_out_squaresum / self.node_counter)
                    - (self.node_out_sum / self.node_counter) ** 2
                )))


    def out_normalise(self, x):
        if self.normalise:
            x = torch.sub(x, self.node_out_mean)
            x = torch.div(x, self.node_out_std)
        return x

    def out_renormalise(self, x):
        if self.normalise:
            x = torch.mul(self.node_out_std, x)
            x = torch.add(x, self.node_out_mean)
        return x



class SequentialModule(pl.LightningModule):
    def __init__(
        self,
        model_type,
        lr: float,
        weight_decay: float,
        log_norm: bool,
        noise=None,
        teacher_forcing: bool = False,
        teacher_forcing_ratio: float = 0.3,
        min_dist: int = 0,
        fully_connected: bool = True,
        edge_weight: bool = False,
        self_loop: bool = True,
        undirected: bool = False,
        rnn_type: str = "GRU",
        out_features: int = 6
    ):
        super().__init__()
        # Set up metrics
        self.train_ade_loss = torchmetrics.MeanSquaredError()
        self.train_fde_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        self.train_yaw_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.val_yaw_loss = torchmetrics.MeanSquaredError()

        # Save parameters
        self.model = model_type
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_norm = log_norm
        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.min_dist = min_dist
        self.edge_weight = edge_weight
        self.undirected = undirected
        self.self_loop = self_loop
        self.rnn_type = rnn_type
        self.out_features = out_features
        self.save_hyperparameters()

    def training_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Extract dimensions
        n_nodes = batch.num_nodes
        y_predictions = torch.zeros((n_nodes, 90, self.out_features))
        edge_index = batch.edge_index
        mask = batch.x[:, :, -1]
        batch.x = batch.x[:, :, :-1]
        static_features = batch.x[:, 0, self.out_features:]
        edge_attr = None
        x = batch.x

        # Obtain target delta dynamic nodes
        # Use torch.roll to compute differences between x_t and x_{t+1}.
        # Ignore final difference (between t_0 and t_{-1})
        y_target_abs = (torch.roll(x[:, :, :self.out_features], (0, -1, 0), (0, 1, 2))
                        - x[:, :, :self.out_features])[:, :-1, :]
        y_target_nrm = torch.zeros_like(y_target_abs)

        # Initial hidden state
        if self.rnn_type == "GRU":
            h = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
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
            x_t = x[:, t, :]
            y_t = y_target_abs[:, t, :]

            # Add noise if specified
            if self.noise is not None:
                x_t[:, :self.out_features] += self.noise * torch.rand_like(x_t[:, :self.out_features])

            ######################
            # Graph construction #
            ######################

            # Determine edge construction
            if self.fully_connected:
                # Generate fully connected graph
                edge_index = torch.LongTensor(generate_fully_connected_edges(n_nodes))
                if not self.self_loop:
                    edge_index = torch_geometric.utils.remove_self_loops(edge_index)
            else:
                assert self.min_dist > 0.0
                # Create radius graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to ensure each node has a neighbour
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2], k=1, batch=batch.batch
                )
                # Combine
                edge_index = torch.cat((edge_index, nn_edge_index), dim=1)
                self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            #######################
            # Training 1/2        #
            #######################

            if edge_attr is None:
                self.model.update_in_normalisation(x_t.clone())
            else:
                self.model.update_in_normalisation(
                    x_t.clone(), edge_attr.clone()
                )
            self.model.update_out_normalisation(y_t.detach().clone())

            # Obtain normalised input graph and normalised target nodes
            x_t_nrm, edge_attr_nrm = self.model.in_normalise(x_t, edge_attr)
            y_t_nrm = self.model.out_normalise(y_t)

            y_target_nrm[:, t, :] = y_t_nrm
            # Obtain normalised predicted delta dynamics
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
            y_pred_abs = self.model.out_renormalise(y_pred_t)
            # Add deltas to input graph
            x_t = torch.cat((x[:, t, :self.out_features] + y_pred_abs, static_features), dim=-1)

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
                x_t = x[:, t, :]
            x_prev = x_t
            y_t = y_target_abs[:, t, :]

            ######################
            # Graph construction #
            ######################

            # Determine edge construction
            if self.fully_connected:
                # Generate fully connected graph
                edge_index = torch.LongTensor(generate_fully_connected_edges(n_nodes))
                if not self.self_loop:
                    edge_index = torch_geometric.utils.remove_self_loops(edge_index)
            else:
                assert self.min_dist > 0.0
                # Create radius graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to ensure each node has a neighbour
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2], k=1, batch=batch.batch
                )
                # Combine
                edge_index = torch.cat((edge_index, nn_edge_index), dim=1)
                self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            #######################
            # Training 2/2        #
            #######################

            if edge_attr is None:
                self.model.update_in_normalisation(x_t.clone())
            else:
                self.model.update_in_normalisation(
                    x_t.clone(), edge_attr.clone()
                )
            self.model.update_out_normalisation(y_t.clone())

            # Obtain normalised input graph and normalised target nodes
            x_t_nrm, edge_attr_nrm = self.model.in_normalise(x_t, edge_attr)
            y_t_nrm = self.model.out_normalise(y_t)
            y_target_nrm[:, t, :] = y_t_nrm
            # Obtain normalised predicted delta dynamics
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
            y_pred_abs = self.model.out_renormalise(y_pred_t)
            # Add deltas to input graph. Input for next timestep
            x_t = torch.cat((x_prev[:, :self.out_features] + y_pred_abs, static_features), dim=-1)

        # Compute and log loss
        fde_loss = self.train_fde_loss(
            y_predictions[:, -1, :2], y_target_nrm[:, -1, :2]
        )
        ade_loss = self.train_ade_loss(y_predictions[:, :, :2], y_target_nrm[:, :, :2])
        vel_loss = self.train_vel_loss(y_predictions[:, :, 2:4], y_target_nrm[:, :, 2:4])
        yaw_loss = self.train_yaw_loss(y_predictions[:, :, 4:6], y_target_nrm[:, :, 4:6])

        self.log("train_fde_loss", fde_loss, on_step=True, on_epoch=True)
        self.log("train_ade_loss", ade_loss, on_step=True, on_epoch=True)
        self.log("train_vel_loss", vel_loss, on_step=True, on_epoch=True)
        self.log("train_yaw_loss", yaw_loss, on_step=True, on_epoch=True)
        self.log(
            "train_total_loss", (vel_loss + ade_loss + yaw_loss) / 3, on_step=True, on_epoch=True
        )

        return ade_loss + vel_loss + yaw_loss

    def validation_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Validate on sequential dataset. First 11 observations are used to prime the model.
        # Loss is computed on remaining 80 samples using rollout.

        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, self.out_features))
        y_target = torch.zeros((80, n_nodes, self.out_features))
        mask = batch.x[:, :, -1]
        batch.x = batch.x[:, :, :-1]
        static_features = batch.x[:, 0, self.out_features:]
        edge_attr = None

        # Initial hidden state
        if self.rnn_type == "GRU":
            h = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
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

            x = batch.x[:, t, :]

            # Determine edge construction
            if self.fully_connected:
                # Generate fully connected graph
                edge_index = torch.LongTensor(generate_fully_connected_edges(n_nodes))
                if not self.self_loop:
                    edge_index = torch_geometric.utils.remove_self_loops(edge_index)
            else:
                assert self.min_dist > 0.0
                # Create radius graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to ensure each node has a neighbour
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=1, batch=batch.batch
                )
                # Combine
                edge_index = torch.cat((edge_index, nn_edge_index), dim=1)
                self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            ######################
            # Validation 1/2     #
            ######################

            # Normalise input graph
            x, edge_attr = self.model.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch.batch,
                hidden=h,
            )
            # Renormalise output dynamics
            x = self.model.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (batch.x[:, t, :self.out_features] + x, static_features), dim=-1
            )

        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph[:, :self.out_features]
        y_target[0, :, :] = batch.x[:, 11, :self.out_features]

        # Predict future values
        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            x = predicted_graph

            # Determine edge construction
            if self.fully_connected:
                # Generate fully connected graph
                edge_index = torch.LongTensor(generate_fully_connected_edges(n_nodes))
                if not self.self_loop:
                    edge_index = torch_geometric.utils.remove_self_loops(edge_index)
            else:
                assert self.min_dist > 0.0
                # Create radius graph
                edge_index = torch_geometric.nn.radius_graph(
                    x=x[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to ensure each node has a neighbour
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x[:, :2], k=1, batch=batch.batch
                )
                # Combine
                edge_index = torch.cat((edge_index, nn_edge_index), dim=1)
                self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

            # Remove duplicates and sort
            edge_index = torch_geometric.utils.coalesce(edge_index)

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x[row, :2] - x[col, :2]).norm(dim=-1).unsqueeze(1)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            ######################
            # Validation 2/2     #
            ######################

            # Normalise input graph
            x, edge_attr = self.model.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch.batch,
                hidden=h,
            )
            # Renormalise deltas
            x = self.model.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (predicted_graph[:, :self.out_features] + x, static_features), dim=-1
            )  # [n_nodes, n_features]

            # Save prediction alongside true value (next time step state)
            y_hat[t - 10, :, :] = predicted_graph[:, :self.out_features]
            y_target[t - 10, :, :] = batch.x[:, t + 1, :self.out_features]

        # Compute and log loss
        fde_loss = self.val_fde_loss(y_hat[-1, :, :2], y_target[-1, :, :2])
        ade_loss = self.val_ade_loss(y_hat[:, :, :2], y_target[:, :, :2])
        vel_loss = self.val_vel_loss(y_hat[:, :, 2:4], y_target[:, :, 2:4])
        yaw_loss = self.val_vel_loss(y_hat[:, :, 4:6], y_target[:, :, 4:6])


        ######################
        # Logging            #
        ######################

        self.log("val_ade_loss", ade_loss)
        self.log("val_fde_loss", fde_loss)
        self.log("val_vel_loss", vel_loss)
        self.log("val_yaw_loss", yaw_loss)
        self.log("val_total_loss", (vel_loss + ade_loss + yaw_loss) / 3)

        return (ade_loss + vel_loss + yaw_loss)/3

    def predict_step(self, batch, batch_idx=None):
        raise NotImplementedError
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
        # Initial hidden state
        if self.rnn_type == "GRU":
            h = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
            # torch.zeros((self.model.num_layers, 1, self.model.rnn_size)))
        else:
            h = None
        # h = torch.zeros_like(batch.x[:, 0, :])
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
            # Predictions 1/2    #
            ######################

            # Normalise input graph
            x, edge_attr = self.model.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch.batch,
                hidden=h,
            )
            # Renormalise output dynamics
            x = self.model.out_renormalise(x)  # [n_nodes, 4]
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (batch.x[:, t, :4] + x, static_features), dim=-1
            )  # [n_nodes, n_features]
            # Save predictions and targets
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
            # Predictions 2/2    #
            ######################

            # Normalise input graph
            x, edge_attr = self.model.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch.batch,
                hidden=h,
            )
            # Renormalise deltas
            x = self.model.out_renormalise(x)
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (predicted_graph[:, :4] + x, static_features), dim=-1
            )  # [n_nodes, n_features]

            # Save prediction alongside true value (next time step state)
            y_hat[t, :, :] = predicted_graph[:, :]
            y_target[t, :, :] = batch.x[:, t + 1, :]

        return y_hat, y_target

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


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

        last_input = batch.x[:, 10, :self.out_features]
        last_delta = batch.x[:, 10, :self.out_features] - batch.x[:, 9, :self.out_features]
        predicted_graph = last_input + last_delta
        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph
        y_target[0, :, :] = batch.x[:, 11, :self.out_features]

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
        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, self.out_features))
        y_target = torch.zeros((80, n_nodes, self.out_features))
        # Remove valid flag from features
        batch.x = batch.x[:, :, :-1]
        # Extract static features
        static_features = batch.x[:, 0, self.out_features:]
        # Find valid agents at time t=11
        initial_mask = mask[:, 10]
        # Extract final dynamic states to use for predictions
        last_pos = batch.x[initial_mask, 10, :2]
        last_vel = batch.x[initial_mask, 10, 2:4]
        last_yaw = batch.x[initial_mask, 10, 4:6]
        # Constant change in positions
        delta_pos = last_vel * 0.1
        # First updated position
        predicted_pos = last_pos + delta_pos
        predicted_graph = torch.cat(
            (predicted_pos, last_vel, last_yaw, static_features), dim=1
        )
        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph[:, :self.out_features]
        y_target[0, :, :] = batch.x[:, 11, :self.out_features]

        # 1 prediction done, 79 remaining
        for t in range(11, 90):
            predicted_pos += delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_vel, static_features), dim=1
            )
            y_hat[t - 10, :, :] = predicted_graph[:, :self.out_features]
            y_target[t - 10, :, :] = batch.x[:, t + 1, :self.out_features]

        # Masks for loss computation
        fde_mask = mask[:, -1]
        val_mask = mask[:, 11:].permute(1, 0)

        # Compute and log loss
        fde_loss = self.val_fde_loss(y_hat[-1, fde_mask, :2], y_target[-1, fde_mask, :2])
        ade_loss = self.val_ade_loss(y_hat[:, :, 0:2][val_mask], y_target[:, :, 0:2][val_mask])
        vel_loss = self.val_vel_loss(y_hat[:, :, 2:4][val_mask], y_target[:, :, 2:4][val_mask])
        yaw_loss = self.val_yaw_loss(y_hat[:, :, 4:6][val_mask], y_target[:, :, 4:6][val_mask])

        # Compute losses on "tracks_to_predict"
        fde_ttp_mask = torch.logical_and(fde_mask, batch.tracks_to_predict)
        fde_ttp_loss = self.val_fde_ttp_loss(y_hat[-1, fde_ttp_mask, :2], y_target[-1, fde_ttp_mask, :2])
        ade_ttp_mask = torch.logical_and(val_mask, batch.tracks_to_predict.expand((80, mask.size(0))))
        ade_ttp_loss = self.val_ade_loss(y_hat[:, :, 0:2][ade_ttp_mask], y_target[:, :, 0:2][ade_ttp_mask])

        ######################
        # Logging            #
        ######################
        self.log("val_ade_loss", ade_loss)
        self.log("val_fde_loss", fde_loss)
        self.log("val_vel_loss", vel_loss)
        self.log("val_yaw_loss", yaw_loss)
        self.log("val_total_loss", (ade_loss + yaw_loss + vel_loss)/3)
        self.log("val_fde_ttp_loss", fde_ttp_loss)
        self.log("val_ade_ttp_loss", ade_ttp_loss)

        return (ade_loss + yaw_loss + vel_loss)/3

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
        # Update mask
        mask = batch.x[:, :, -1].bool()

        # Allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((90, n_nodes, 9))
        y_target = torch.zeros((90, n_nodes, 9))
        # Remove valid flag from features
        batch.x = batch.x[:, :, :-1]
        # Extract static features
        static_features = batch.x[:, 0, self.out_features:]

        # Fill in targets
        for t in range(0, 90):
            y_target[t, :, :] = batch.x[:, t + 1, :]

        for t in range(11):
            mask_t = mask[:, t]

            last_pos = batch.x[mask_t, t, 0:2]
            last_vel = batch.x[mask_t, t, 2:4]
            last_yaw = batch.x[mask_t, t, 4:6]

            delta_pos = last_vel * 0.1
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_vel, last_yaw, static_features[mask_t]), dim=1
            )
            y_hat[t, mask_t, :] = predicted_graph

        for t in range(11, 90):
            last_pos = predicted_pos
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_vel, last_yaw, static_features), dim=1
            )
            y_hat[t, :, :] = predicted_graph

        return y_hat, y_target, mask

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-4
        )


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
            filename=config["logger"]["version"] + "-{epoch}-{loss:.2f}",
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
            filename=config["logger"]["version"] + "-{epoch}-{val_ade_loss:.2f}",
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
