import argparse
import pytorch_lightning as pl
import torch
import torch_geometric.nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from data.dataset import OneStepNBodyDataModule, SequentialNBodyDataModule
import torchmetrics
from torch_geometric.data import Batch, Data
from models.model import *
import yaml


class OneStepModule(pl.LightningModule):
    def __init__(
        self,
        model_type,
        noise,
        lr: float,
        weight_decay: float,
        log_norm: bool,
        min_dist=None,
        edge_weight: bool = False,
        self_loop: bool = True,
        undirected: bool = False,
        grav_attraction: bool = False,
    ):
        super().__init__()
        self.train_pos_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.model = model_type
        self.save_hyperparameters()
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_norm = log_norm
        self.min_dist = min_dist
        self.edge_weight = edge_weight
        self.self_loop = self_loop
        self.undirected = undirected
        self.grav_attraction = grav_attraction

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
            x += self.noise * torch.rand_like(x)
            # # Add noise to dynamic states
            # x_d = x[:, :4] + self.noise*torch.rand_like(x[:, :4])
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
            self.model.NormBlock.update_in_normalisation(x.clone())
        else:
            self.model.NormBlock.update_in_normalisation(x.clone(), edge_attr.clone())
        self.model.NormBlock.update_out_normalisation(y_target.clone())

        # Obtain normalised input graph and normalised target nodes
        x_nrm, edge_attr_nrm = self.model.NormBlock.in_normalise(x, edge_attr)
        y_target_nrm = self.model.NormBlock.out_normalise(y_target)

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
        y_hat = torch.zeros((80, n_nodes, 4), device=self.device)
        y_target = torch.zeros((80, n_nodes, 4), device=self.device)
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
            x, edge_attr = self.model.NormBlock.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise output dynamics
            x = self.model.NormBlock.out_renormalise(x)
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
            x, edge_attr = self.model.NormBlock.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise deltas
            x = self.model.NormBlock.out_renormalise(x)
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
                {f"in_std_{i}": std for i, std in enumerate(self.model.in_std)},
            )
            self.log(
                "in_mean",
                {f"in_mean_{i}": mean for i, mean in enumerate(self.model.in_mean)},
            )
            self.log(
                "out_std",
                {f"out_std_{i}": std for i, std in enumerate(self.model.out_std)},
            )
            self.log(
                "out_mean",
                {f"out_mean_{i}": mean for i, mean in enumerate(self.model.out_mean)},
            )

        return (ade_loss + vel_loss) / 2

    def predict_step(self, batch, batch_idx=None):

        ######################
        # Initialisation     #
        ######################

        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        n_features = 5
        y_hat = torch.zeros((90, n_nodes, n_features), device=self.device)
        y_target = torch.zeros((90, n_nodes, n_features), device=self.device)
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
            x, edge_attr = self.model.NormBlock.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise output dynamics
            x = self.model.NormBlock.out_renormalise(x)
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
            x, edge_attr = self.model.NormBlock.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch
            )
            # Renormalise deltas
            x = self.model.NormBlock.out_renormalise(x)
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
        min_dist=None,
        edge_weight: bool = False,
        self_loop: bool = True,
        undirected: bool = False,
        rnn_type: str = "GRU",
        grav_attraction: bool = False,
    ):
        super().__init__()
        self.train_ade_loss = torchmetrics.MeanSquaredError()
        self.train_fde_loss = torchmetrics.MeanSquaredError()
        self.train_vel_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.model = model_type
        self.save_hyperparameters()
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
        self.grav_attraction = grav_attraction

    def training_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Extract dimensions
        n_nodes = batch.num_nodes
        y_predictions = torch.zeros((n_nodes, 90, 4), device=self.device)
        edge_index = batch.edge_index
        static_features = batch.x[:, 0, 4:]
        x = batch.x
        edge_attr = None

        # # Add noise if specified
        # if self.noise is not None:
        #     # Add noise to dynamic states
        #     x[:, :, :4] += self.noise*torch.rand_like(x[:, :, :4])

        # Obtain target delta dynamic nodes
        # Use torch.roll to compute differences between x_t and x_{t+1}.
        # Ignore final difference (between t_0 and t_{-1})
        y_target_abs = (torch.roll(x[:, :, :4], (0, -1, 0), (0, 1, 2)) - x[:, :, :4])[
            :, :-1, :
        ]
        y_target_nrm = torch.zeros_like(y_target_abs)

        # Update normalisation states
        # for t in range(90):
        #     self.model.update_in_normalisation(x[:, t, :])
        #     self.model.NormBlock.update_out_normalisation(y_target_abs[:, t, :])

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
                # Add noise to dynamic states
                x_t += self.noise * torch.rand_like(x_t)

            ######################
            # Graph construction #
            ######################

            # Determine whether to compute edge_index as a function of distances
            if self.min_dist is not None:
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to ensure connected graphs
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2], k=1, batch=batch.batch
                )
                # Remove duplicates
                edge_index = torch_geometric.utils.coalesce(
                    torch.cat((edge_index, nn_edge_index), dim=1)
                )
                self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            #######################
            # Training 1/2        #
            #######################

            if edge_attr is None:
                self.model.NormBlock.update_in_normalisation(x_t.detach().clone())
            else:
                self.model.NormBlock.update_in_normalisation(
                    x_t.detach().clone(), edge_attr.detach().clone()
                )
            self.model.NormBlock.update_out_normalisation(y_t.detach().clone())

            # Obtain normalised input graph and normalised target nodes
            x_t_nrm, edge_attr_nrm = self.model.NormBlock.in_normalise(x_t, edge_attr)
            y_t_nrm = self.model.NormBlock.out_normalise(y_t)

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
            y_pred_abs = self.model.NormBlock.out_renormalise(y_pred_t)
            # Add deltas to input graph
            x_t = torch.cat((x[:, t, :4] + y_pred_abs, static_features), dim=-1)

        # If using teacher_forcing, draw sample and accept teach_forcing_ratio % of the time. Else, deny.
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

            # Add noise if specified
            if self.noise is not None:
                # Add noise to dynamic states
                x_t += self.noise * torch.rand_like(x_t)

            ######################
            # Graph construction #
            ######################

            # Determine whether to compute edge_index as a function of distances
            if self.min_dist is not None:
                edge_index = torch_geometric.nn.radius_graph(
                    x=x_t[:, :2],
                    r=self.min_dist,
                    batch=batch.batch,
                    loop=self.self_loop,
                    max_num_neighbors=30,
                    flow="source_to_target",
                )
                # 1 nearest neighbour to ensure connected graphs
                nn_edge_index = torch_geometric.nn.knn_graph(
                    x=x_t[:, :2], k=1, batch=batch.batch
                )
                # Remove duplicates
                edge_index = torch_geometric.utils.coalesce(
                    torch.cat((edge_index, nn_edge_index), dim=1)
                )
                self.log("average_num_edges", edge_index.shape[1] / batch.num_graphs)

            if self.edge_weight:
                # Encode distance between nodes as edge_attr
                row, col = edge_index
                edge_attr = (x_t[row, :2] - x_t[col, :2]).norm(dim=-1).unsqueeze(1)
                if self.grav_attraction:
                    # Compute gravitational attraction between all nodes
                    m1 = x_t[row, 4] * 1e10
                    m2 = x_t[col, 4] * 1e10
                    attraction = m1 * m2 / (edge_attr.squeeze() ** 2) * 6.674e-11
                    edge_attr = torch.hstack([edge_attr, attraction.unsqueeze(1)])
                    # Replace inf values with 0
                    edge_attr = torch.nan_to_num(edge_attr, posinf=0)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr
                )

            #######################
            # Training 2/2        #
            #######################

            if edge_attr is None:
                self.model.NormBlock.update_in_normalisation(x_t.detach().clone())
            else:
                self.model.NormBlock.update_in_normalisation(
                    x_t.detach().clone(), edge_attr.detach().clone()
                )
            self.model.NormBlock.update_out_normalisation(y_t.detach().clone())

            # Obtain normalised input graph and normalised target nodes
            x_t_nrm, edge_attr_nrm = self.model.NormBlock.in_normalise(x_t, edge_attr)
            y_t_nrm = self.model.NormBlock.out_normalise(y_t)
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
            y_pred_abs = self.model.NormBlock.out_renormalise(y_pred_t)
            # Add deltas to input graph. Input for next timestep
            x_t = torch.cat((x_prev[:, :4] + y_pred_abs, static_features), dim=-1)

        # Compute and log loss
        fde_loss = self.train_fde_loss(
            y_predictions[:, -1, :2], y_target_nrm[:, -1, :2]
        )
        ade_loss = self.train_ade_loss(y_predictions[:, :, :2], y_target_nrm[:, :, :2])
        vel_loss = self.train_vel_loss(y_predictions[:, :, 2:], y_target_nrm[:, :, 2:])

        self.log("train_fde_loss", fde_loss, on_step=True, on_epoch=True)
        self.log("train_ade_loss", ade_loss, on_step=True, on_epoch=True)
        self.log("train_vel_loss", vel_loss, on_step=True, on_epoch=True)
        self.log(
            "train_total_loss", (vel_loss + ade_loss) / 2, on_step=True, on_epoch=True
        )

        return ade_loss + vel_loss

    def validation_step(self, batch: Batch, batch_idx: int):

        ######################
        # Initialisation     #
        ######################

        # Validate on sequential dataset. First 11 observations are used to prime the model.
        # Loss is computed on remaining 80 samples using rollout.

        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, 4), device=self.device)
        y_target = torch.zeros((80, n_nodes, 4), device=self.device)
        edge_index = batch.edge_index
        static_features = batch.x[:, 0, 4:]
        # Initial hidden state
        if self.rnn_type == "GRU":
            h = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
        elif self.rnn_type == "LSTM":
            h = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
            c = torch.zeros((self.model.num_layers, 1, self.model.rnn_size))
            h = (h, c)
        else:
            h = None
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
            x, edge_attr = self.model.NormBlock.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch.batch,
                hidden=h,
            )
            # Renormalise output dynamics
            x = self.model.NormBlock.out_renormalise(x)  # [n_nodes, 4]
            # Add deltas to input graph
            predicted_graph = torch.cat(
                (batch.x[:, t, :4] + x, static_features), dim=-1
            )  # [n_nodes, n_features]

        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph[:, :4]
        y_target[0, :, :] = batch.x[:, 11, :4]

        # Predict future values
        for t in range(11, 90):

            ######################
            # Graph construction #
            ######################

            x = predicted_graph

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
            # Validation 2/2     #
            ######################

            # Normalise input graph
            x, edge_attr = self.model.NormBlock.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch.batch,
                hidden=h,
            )
            # Renormalise deltas
            x = self.model.NormBlock.out_renormalise(x)
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
        self.log("val_total_loss", (vel_loss + ade_loss) / 2)

        # Log normalisation states
        if self.log_norm:
            self.log(
                "in_std",
                {f"in_std_{i}": std for i, std in enumerate(self.model.in_std)},
            )
            self.log(
                "in_mean",
                {f"in_mean_{i}": mean for i, mean in enumerate(self.model.in_mean)},
            )
            self.log(
                "out_std",
                {f"out_std_{i}": std for i, std in enumerate(self.model.out_std)},
            )
            self.log(
                "out_mean",
                {f"out_mean_{i}": mean for i, mean in enumerate(self.model.out_mean)},
            )

        return ade_loss + vel_loss

    def predict_step(self, batch, batch_idx=None):

        ######################
        # Initialisation     #
        ######################

        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        n_features = 5
        y_hat = torch.zeros((90, n_nodes, n_features), device=self.device)
        y_target = torch.zeros((90, n_nodes, n_features), device=self.device)
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
            x, edge_attr = self.model.NormBlock.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch.batch,
                hidden=h,
            )
            # Renormalise output dynamics
            x = self.model.NormBlock.out_renormalise(x)  # [n_nodes, 4]
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
            x, edge_attr = self.model.NormBlock.in_normalise(x, edge_attr)
            # Obtain normalised predicted delta dynamics
            x, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch.batch,
                hidden=h,
            )
            # Renormalise deltas
            x = self.model.NormBlock.out_renormalise(x)
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
    def __init__(self, model_type, noise, lr, weight_decay, log_norm):
        super().__init__()
        self.train_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.model = model_type
        self.save_hyperparameters()
        self.noise = noise
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_norm = log_norm

    def training_step(self, batch: Batch, batch_idx: int):
        pass

    def validation_step(self, batch: Batch, batch_idx: int):
        # Validate on sequential dataset. First 11 observations are used to prime the model.
        # Loss is computed on remaining 80 samples using rollout.

        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, 4), device=self.device)
        y_target = torch.zeros((80, n_nodes, 4), device=self.device)

        last_input = batch.x[:, 10, :4]
        last_delta = batch.x[:, 10, :4] - batch.x[:, 9, :4]
        predicted_graph = last_input + last_delta
        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph
        y_target[0, :, :] = batch.x[:, 11, :4]

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
        y_hat = torch.zeros((90, n_nodes, n_features), device=self.device)
        y_target = torch.zeros((90, n_nodes, n_features), device=self.device)

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
    def __init__(self, model, **kwargs):
        super().__init__()
        assert model is None
        self.train_loss = torchmetrics.MeanSquaredError()
        self.val_ade_loss = torchmetrics.MeanSquaredError()
        self.val_fde_loss = torchmetrics.MeanSquaredError()
        self.val_vel_loss = torchmetrics.MeanSquaredError()
        self.save_hyperparameters()

    def training_step(self, batch: Batch, batch_idx: int):
        pass

    def validation_step(self, batch: Batch, batch_idx: int):
        # Validate on sequential dataset. First 11 observations are used to prime the model.
        # Loss is computed on remaining 80 samples using rollout.

        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        y_hat = torch.zeros((80, n_nodes, 4), device=self.device)
        y_target = torch.zeros((80, n_nodes, 4), device=self.device)
        static_features = batch.x[:, 0, 4]
        last_pos = batch.x[:, 10, :2]
        last_vel = batch.x[:, 10, 2:4]
        delta_pos = last_vel * 0.1
        predicted_pos = last_pos + delta_pos
        predicted_graph = torch.cat(
            (predicted_pos, last_vel, static_features.unsqueeze(1)), dim=1
        )
        # Save first prediction and target
        y_hat[0, :, :] = predicted_graph[:, :4]
        y_target[0, :, :] = batch.x[:, 11, :4]

        # 1 prediction done, 79 remaining
        for t in range(11, 90):
            predicted_pos += delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_vel, static_features.unsqueeze(1)), dim=1
            )
            y_hat[t - 10, :, :] = predicted_graph[:, :4]
            y_target[t - 10, :, :] = batch.x[:, t + 1, :4]

        # Compute and log loss
        fde_loss = self.val_fde_loss(y_hat[-1, :, :2], y_target[-1, :, :2])
        ade_loss = self.val_ade_loss(y_hat[:, :, :2], y_target[:, :, :2])
        vel_loss = self.val_vel_loss(y_hat[:, :, 2:], y_target[:, :, 2:])

        self.log("val_ade_loss", ade_loss)
        self.log("val_fde_loss", fde_loss)
        self.log("val_vel_loss", vel_loss)
        self.log("val_total_loss", (ade_loss+vel_loss)/2)

        return ade_loss+vel_loss

    def predict_step(self, batch, batch_idx=None):
        # Extract dimensions and allocate target/prediction tensors
        n_nodes = batch.num_nodes
        n_features = 5
        y_hat = torch.zeros((90, n_nodes, n_features), device=self.device)
        y_target = torch.zeros((90, n_nodes, n_features), device=self.device)

        # Fill in targets
        for t in range(0, 90):
            y_target[t, :, :] = batch.x[:, t + 1, :]

        static_features = batch.x[:, 0, 4]
        for t in range(11):
            last_pos = batch.x[:, t, :2]
            last_vel = batch.x[:, t, 2:4]
            delta_pos = last_vel * 0.1
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_vel, static_features.unsqueeze(1)), dim=1
            )
            y_hat[t, :, :] = predicted_graph

        for t in range(11, 90):
            last_pos = predicted_pos
            # velocity no longer changing
            # delta_pos no longer chaning
            predicted_pos = last_pos + delta_pos
            predicted_graph = torch.cat(
                (predicted_pos, last_vel, static_features.unsqueeze(1)), dim=1
            )
            y_hat[t, :, :] = predicted_graph

        return y_hat, y_target

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
        # model = AbstractGNN(node_model=config["misc"]["model_type"], **config["model"])
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
            monitor="val_ade_loss",
            dirpath="checkpoints",
            filename=config["logger"]["version"] + "-{epoch}-{val_ade_loss:.2f}",
        )
        # Create trainer, fit, and validate
        trainer = pl.Trainer(
            logger=wandb_logger, **config["trainer"], callbacks=[checkpoint_callback]
        )
    else:
        # Create trainer, fit, and validate
        trainer = pl.Trainer(logger=wandb_logger, **config["trainer"])

    if config["misc"]["continue_training"] is not None:
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
