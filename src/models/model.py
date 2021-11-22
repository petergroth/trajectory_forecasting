from typing import Optional, Tuple

import torch
import torch_geometric.utils
from torch import nn, Tensor
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    Sequential,
    GatedGraphConv,
    MessagePassing,
)
from torch_geometric.nn.meta import MetaLayer
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add

# from src.data.dataset import SequentialNBodyDataModule, OneStepNBodyDataModule
from src.models.node_edge_blocks import *
from torch_geometric.utils import dropout_adj

# from torch_geometric_temporal.nn import GConvLSTM, GCLSTM, TGCN


class ConstantModel(nn.Module):
    def __init__(self):
        super(ConstantModel, self).__init__()

    def forward(self, x, edge_index):
        # Identity
        return torch.zeros_like(x[:, :4])


class mlp_forward_model(nn.Module):
    # Forward model without edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        skip: bool = True,
        aggregate: bool = False,
        **kwargs
    ):
        super(mlp_forward_model, self).__init__()
        self.aggregate = aggregate
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.GN1 = GraphNetworkBlock(
            node_model=node_mlp_1(
                hidden_size=hidden_size,
                node_features=node_features,
                dropout=dropout,
                edge_features=0,
            )
        )
        GN2_node_input = hidden_size + node_features if skip else hidden_size

        self.GN2 = GraphNetworkBlock(
            node_model=node_mlp_out(
                hidden_size=hidden_size,
                node_features=GN2_node_input,
                dropout=dropout,
                edge_features=0,
            )
        )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        # Normalisation is applied in regressor module

        # First block
        x_1, _, _ = self.GN1(
            x=x, edge_index=edge_index, edge_attr=None, u=u, batch=batch
        )
        # concatenation of node and edge attributes
        if self.aggregate:
            x_1 = scatter_add(x_1, batch, dim=0)
            x_1 = torch.cat([x, x_1[batch]], dim=1)
        else:
            x_1 = torch.cat([x, x_1], dim=1)
        # edge_attr_1 = torch.cat([edge_attr, edge_attr_1], dim=1)
        # Second block
        out, _, _ = self.GN2(
            x=x_1, edge_index=edge_index, edge_attr=None, u=u, batch=batch
        )
        return out


class mlp_full_forward_model(nn.Module):
    # Forward model with edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
        latent_edge_features: int = 0,
        skip: bool = True,
        aggregate: bool = False,
        out_features: int = 4,
    ):
        super(mlp_full_forward_model, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.edge_features = edge_features
        self.latent_edge_features = latent_edge_features
        self.aggregate = aggregate

        self.GN1 = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=node_features,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_mlp_1(
                hidden_size=hidden_size,
                node_features=node_features,
                dropout=dropout,
                edge_features=latent_edge_features,
            ),
        )
        GN2_node_input = node_features + hidden_size if skip else hidden_size
        GN2_edge_input = (
            edge_features + latent_edge_features if skip else latent_edge_features
        )

        self.GN2 = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=GN2_node_input,
                edge_features=GN2_edge_input,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_mlp_out(
                hidden_size=hidden_size,
                node_features=GN2_node_input,
                dropout=dropout,
                edge_features=latent_edge_features,
                out_features=out_features,
            ),
        )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        # First block
        x_1, edge_attr_1, _ = self.GN1(
            x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch
        )
        # Concatenation of node and edge attributes
        if self.aggregate:
            x_1 = scatter_add(x_1, batch, dim=0)
            x_1 = torch.cat([x, x_1[batch]], dim=1)
        else:
            x_1 = torch.cat([x, x_1], dim=1)

        edge_attr_1 = torch.cat([edge_attr, edge_attr_1], dim=1)
        # Second block
        out, _, _ = self.GN2(
            x=x_1, edge_index=edge_index, edge_attr=edge_attr_1, u=u, batch=batch
        )
        return out


class mpnn_forward_model(nn.Module):
    # Message passing neural network. Inspired from https://arxiv.org/abs/2002.09405v2
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
        latent_edge_features: int = 0,
        rounds: int = 1,
        shared_params: bool = True,
        out_features: int = 7,
        **kwargs
    ):
        super(mpnn_forward_model, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.edge_features = edge_features
        self.latent_edge_features = latent_edge_features
        self.rounds = rounds
        self.shared_params = shared_params

        # Encoder block to create latent edge/node representations
        self.encoder = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=node_features,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_mlp_encoder(
                hidden_size=hidden_size,
                node_features=node_features,
                dropout=dropout,
                edge_features=latent_edge_features,
            ),
        )

        if shared_params:
            # If parameter sharing, create single GN block
            self.processor = GraphNetworkBlock(
                edge_model=edge_mlp_latent(
                    node_features=hidden_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    latent_edge_features=latent_edge_features,
                ),
                node_model=node_mlp_latent(
                    hidden_size=hidden_size,
                    node_features=hidden_size,
                    dropout=dropout,
                    edge_features=latent_edge_features,
                ),
            )
        else:
            # Else, create module list of separate blocks
            self.processors = nn.ModuleList(
                [
                    GraphNetworkBlock(
                        edge_model=edge_mlp_latent(
                            node_features=hidden_size,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            latent_edge_features=latent_edge_features,
                        ),
                        node_model=node_mlp_latent(
                            hidden_size=hidden_size,
                            node_features=hidden_size,
                            dropout=dropout,
                            edge_features=latent_edge_features,
                        ),
                    )
                    for _ in range(rounds)
                ]
            )

        # Decoder block to produce output dynamics
        self.decoder = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=hidden_size,
                edge_features=latent_edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_mlp_out(
                hidden_size=hidden_size,
                node_features=hidden_size,
                dropout=dropout,
                edge_features=latent_edge_features,
                out_features=out_features,
            ),
        )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        # Encoding to latent representation
        x_latent_prev, edge_attr_latent_prev, _ = self.encoder(
            x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch
        )
        # Perform message passing
        if self.shared_params:
            # Use same block if sharing parameters
            for _ in range(self.rounds):
                x_latent, edge_attr_latent, _ = self.processor(
                    x=x_latent_prev,
                    edge_index=edge_index,
                    edge_attr=edge_attr_latent_prev,
                    u=u,
                    batch=batch,
                )
                x_latent_prev += x_latent
                edge_attr_latent_prev += edge_attr_latent
        else:
            # Iterate through blocks if not sharing
            for mp_block in self.processors:
                x_latent, edge_attr_latent, _ = mp_block(
                    x=x_latent_prev,
                    edge_index=edge_index,
                    edge_attr=edge_attr_latent_prev,
                    u=u,
                    batch=batch,
                )
                x_latent_prev += x_latent
                edge_attr_latent_prev += edge_attr_latent

        # Decode output to produce delta dynamics
        out, _, _ = self.decoder(
            x=x_latent_prev,
            edge_index=edge_index,
            edge_attr=edge_attr_latent_prev,
            u=u,
            batch=batch,
        )

        return out


class rnn_full_forward_model(nn.Module):
    # Forward model with edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
        latent_edge_features: int = 0,
        skip: bool = True,
        normalise: bool = True,
        num_layers: int = 1,
        rnn_size: int = 20,
    ):
        super(rnn_full_forward_model, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.edge_features = edge_features
        self.rnn_size = rnn_size
        self.num_layers = num_layers

        self.GN1 = GraphNetworkBlock(
            edge_model=edge_rnn_1(
                node_features=node_features,
                edge_features=edge_features,
                rnn_size=rnn_size,
                dropout=dropout,
                num_layers=num_layers,
            ),
            node_model=node_rnn_1(
                node_features=node_features,
                edge_features=edge_features,
                rnn_size=rnn_size,
                dropout=dropout,
                num_layers=num_layers,
            ),
        )
        GN2_node_input = rnn_size + node_features if skip else rnn_size
        GN2_edge_input = rnn_size + edge_features if skip else rnn_size

        self.GN2 = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=GN2_node_input,
                edge_features=0,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=GN2_edge_input,
            ),
            node_model=node_mlp_out(
                hidden_size=hidden_size,
                node_features=GN2_node_input,
                dropout=dropout,
                edge_features=GN2_edge_input,
            ),
        )

        # self.NormBlock = NormalisationBlock(
        #     normalise=normalise,
        #     node_features=node_features,
        #     edge_features=edge_features,
        # )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None, hidden=None):
        # Normalisation is applied in regressor module

        # First block
        x_1, edge_attr_1, _, hidden = self.GN1(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u,
            batch=batch,
            hidden=hidden,
        )
        # Concatenation of node and edge attributes
        x_1 = torch.cat([x, x_1], dim=1)  # [rnn_dim + node_features]
        edge_attr_1 = torch.cat(
            [edge_attr, edge_attr_1], dim=1
        )  # [rnn_dim + edge_features]
        # Second block
        out, _, _ = self.GN2(
            x=x_1, edge_index=edge_index, edge_attr=edge_attr_1, u=u, batch=batch
        )
        return out, hidden


class rnn_node_forward_model(nn.Module):
    # Forward model with edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
        latent_edge_features: int = 0,
        skip: bool = True,
        num_layers: int = 1,
        rnn_size: int = 20,
        out_features: int = 7,
        **kwargs,
    ):
        super(rnn_node_forward_model, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.edge_features = edge_features
        self.latent_edge_features = latent_edge_features
        self.rnn_size = rnn_size
        self.num_layers = num_layers

        self.GN1 = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=node_features,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_rnn_1(
                node_features=node_features,
                edge_features=latent_edge_features,
                rnn_size=rnn_size,
                dropout=dropout,
                num_layers=num_layers,
                out_features=out_features,
                hidden_size=hidden_size,
            ),
        )
        GN2_node_input = rnn_size + node_features if skip else rnn_size
        GN2_edge_input = (
            latent_edge_features + edge_features if skip else latent_edge_features
        )

        self.GN2 = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=GN2_node_input,
                edge_features=GN2_edge_input,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_mlp_out(
                hidden_size=hidden_size,
                node_features=GN2_node_input,
                dropout=dropout,
                edge_features=latent_edge_features,
                out_features=out_features,
            ),
        )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None, hidden=None):
        # Normalisation is applied in regressor module
        # First block
        x_1, edge_attr_1, _, hidden = self.GN1(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u,
            batch=batch,
            hidden=hidden,
        )
        # Concatenation of node and edge attributes
        x_1 = torch.cat([x, x_1], dim=1)
        edge_attr_1 = torch.cat([edge_attr, edge_attr_1], dim=1)
        # Second block
        out, _, _ = self.GN2(
            x=x_1, edge_index=edge_index, edge_attr=edge_attr_1, u=u, batch=batch
        )
        return out, hidden


class rnn_forward_model_v2(nn.Module):
    # Forward model with edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
        latent_edge_features: int = 0,
        skip: bool = True,
        num_layers: int = 1,
        rnn_size: int = 20,
        out_features: int = 7,
        **kwargs,
    ):
        super(rnn_forward_model_v2, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.edge_features = edge_features
        self.latent_edge_features = latent_edge_features
        self.rnn_size = rnn_size
        self.num_layers = num_layers

        self.GN1 = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=node_features,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_rnn_2(
                node_features=node_features,
                edge_features=latent_edge_features,
                rnn_size=rnn_size,
                dropout=dropout,
                num_layers=num_layers,
                out_features=out_features,
                hidden_size=hidden_size,
            ),
        )
        GN2_node_input = hidden_size + node_features if skip else hidden_size
        GN2_edge_input = (
            latent_edge_features + edge_features if skip else latent_edge_features
        )

        self.GN2 = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=GN2_node_input,
                edge_features=GN2_edge_input,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_mlp_out(
                hidden_size=hidden_size,
                node_features=GN2_node_input,
                dropout=dropout,
                edge_features=latent_edge_features,
                out_features=out_features,
            ),
        )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None, hidden=None):
        # Normalisation is applied in regressor module
        # First block
        x_1, edge_attr_1, _, hidden = self.GN1(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u,
            batch=batch,
            hidden=hidden,
        )
        # Concatenation of node and edge attributes
        x_1 = torch.cat([x, x_1], dim=1)
        edge_attr_1 = torch.cat([edge_attr, edge_attr_1], dim=1)
        # Second block
        out, _, _ = self.GN2(
            x=x_1, edge_index=edge_index, edge_attr=edge_attr_1, u=u, batch=batch
        )
        return out, hidden


class conv_model(nn.Module):
    # Forward model without edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        skip: bool = True,
        normalise: bool = True,
        edge_features: int = 1,
        **kwargs
    ):
        super(conv_model, self).__init__()
        self.edge_features = 1
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.out_features = 4

        self.GN1 = GraphNetworkBlock(
            node_model=GCNConv(in_channels=node_features, out_channels=hidden_size)
        )

        GN2_node_input = hidden_size + node_features if skip else hidden_size

        self.GN2 = GraphNetworkBlock(
            node_model=GCNConv(
                in_channels=GN2_node_input, out_channels=self.out_features
            )
        )

        # self.NormBlock = NormalisationBlock(
        #     normalise=normalise,
        #     node_features=node_features,
        #     edge_features=edge_features,
        # )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        # Normalisation is applied in regressor module

        # First block
        x_1, edge_attr_1, _ = self.GN1(
            x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch
        )
        # concatenation of node and edge attributes
        x_1 = torch.cat([x, x_1], dim=1)
        # edge_attr_1 = torch.cat([edge_attr, edge_attr_1], dim=1)
        # Second block
        out, _, _ = self.GN2(
            x=x_1, edge_index=edge_index, edge_attr=edge_attr_1, u=u, batch=batch
        )
        return out


class GraphNetworkBlock(MetaLayer):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        hidden=None,
    ):
        """"""
        # Hidden is a tuple of (edge_hidden, node_hidden)
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            btch = batch if batch is None else batch[row]
            # if hidden is None:
            edge_attr = self.edge_model(
                src=x[row],
                dest=x[col],
                edge_attr=edge_attr,
                u=u,
                batch=btch,
                edge_index=edge_index,
            )

        if self.node_model is not None:
            if hidden is None:
                x = self.node_model(x, edge_index, edge_attr, u, batch)
            else:
                x, hidden = self.node_model(
                    x, edge_index, edge_attr, u, batch, hidden=hidden
                )

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        if hidden is None:
            return x, edge_attr, u
        else:
            return x, edge_attr, u, hidden

    def __repr__(self):
        return (
            "{}(\n"
            "    edge_model={},\n"
            "    node_model={},\n"
            "    global_model={}\n"
            ")"
        ).format(
            self.__class__.__name__, self.edge_model, self.node_model, self.global_model
        )


class mlp_baseline(nn.Module):
    # Forward model without edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        n_nodes: int = 10,
        normalise: bool = True,
        dropout: float = 0.0,
        permute: bool = False,
        **kwargs
    ):
        super(mlp_baseline, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.out_features = 4

        self.mlp_1 = nn.Sequential(
            nn.Linear(in_features=node_features * n_nodes, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(
                in_features=node_features * n_nodes + hidden_size,
                out_features=hidden_size,
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_size, out_features=self.out_features * self.n_nodes
            ),
        )
        #
        # self.NormBlock = NormalisationBlock(
        #     normalise=normalise, node_features=node_features, edge_features=0
        # )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        n_graphs = torch.max(batch) + 1
        # If permuting, change order of inputs
        # Reshape input from [n_nodes_in_batch, n_features] to [n_graphs, n_nodes*n_features]
        x = torch.reshape(x, (n_graphs, self.n_nodes * self.node_features))
        # Forward pass
        x_latent = self.mlp_1(x)
        # Skip connection
        x_latent = torch.cat([x, x_latent], dim=1)
        # Second pass
        out = self.mlp_2(x_latent)
        return torch.reshape(out, (self.n_nodes * n_graphs, self.out_features))


class mlp_node_baseline(nn.Module):
    # Forward model without edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        n_nodes: int = 10,
        dropout: float = 0.0,
        out_features: int = 7,
        **kwargs
    ):
        super(mlp_node_baseline, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.out_features = out_features

        self.mlp_1 = nn.Sequential(
            nn.Linear(in_features=node_features, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(
                in_features=node_features + hidden_size,
                out_features=hidden_size,
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.out_features),
        )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        # Forward pass
        x_latent = self.mlp_1(x)
        # Skip connection
        x_latent = torch.cat([x, x_latent], dim=1)
        # Second pass
        out = self.mlp_2(x_latent)
        return out


class rnn_baseline(nn.Module):
    # Forward model without edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        n_nodes: int = 10,
        normalise: bool = True,
        dropout: float = 0.0,
        rnn_size: int = 20,
        num_layers: int = 1,
        **kwargs
    ):
        super(rnn_baseline, self).__init__()
        self.normalise = normalise
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.rnn_size = rnn_size
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.out_features = 4
        self.num_layers = num_layers

        self.rnn_1 = nn.LSTM(
            input_size=node_features,
            hidden_size=rnn_size,
            num_layers=1,
            batch_first=True,
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(in_features=rnn_size + node_features, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.out_features),
        )

        # self.NormBlock = NormalisationBlock(
        #     normalise=normalise, node_features=node_features, edge_features=0
        # )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None, hidden=None):
        h, c = hidden
        if h.shape[1] != x.shape[0]:
            h = torch.repeat_interleave(input=h, repeats=x.shape[0], dim=1)
            c = torch.repeat_interleave(input=c, repeats=x.shape[0], dim=1)
            hidden = (h, c)
        # Forward pass
        x_latent, hidden = self.rnn_1(x.unsqueeze(1), hidden)
        x_latent = torch.cat([x, x_latent.squeeze()], dim=1)
        # Second pass
        out = self.mlp_2(x_latent)
        return out, hidden


class rnn_graph_baseline(nn.Module):
    # Forward model without edge update function
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        n_nodes: int = 10,
        normalise: bool = True,
        dropout: float = 0.0,
        rnn_size: int = 20,
        num_layers: int = 1,
        **kwargs
    ):
        super(rnn_graph_baseline, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.rnn_size = rnn_size
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.out_features = 4
        self.num_layers = num_layers

        self.rnn_1 = nn.LSTM(
            input_size=node_features * n_nodes,
            hidden_size=rnn_size,
            num_layers=1,
            batch_first=True,
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(
                in_features=rnn_size + n_nodes * node_features, out_features=hidden_size
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_size, out_features=self.n_nodes * self.out_features
            ),
        )

        # self.NormBlock = NormalisationBlock(
        #     normalise=normalise, node_features=node_features, edge_features=0
        # )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None, hidden=None):
        # Reshape input
        n_graphs = torch.max(batch) + 1
        x = torch.reshape(x, (n_graphs, self.n_nodes * self.node_features))
        # Repeat hidden first step
        h, c = hidden
        if h.shape[1] != x.shape[0]:
            h = torch.repeat_interleave(input=h, repeats=x.shape[0], dim=1)
            c = torch.repeat_interleave(input=c, repeats=x.shape[0], dim=1)
            hidden = (h, c)
        # Forward pass
        x_latent, hidden = self.rnn_1(x.unsqueeze(1), hidden)
        x_latent = torch.cat([x, x_latent.squeeze()], dim=1)
        # Second pass
        out = self.mlp_2(x_latent)
        out = out.squeeze()
        return torch.reshape(out, (self.n_nodes * n_graphs, self.out_features)), hidden


class NormalisationBlock:
    def __init__(
        self,
        normalise: bool = True,
        node_features: int = 5,
        edge_features: int = 2,
        subtract_edge_mean: bool = True,
        out_features: int = 4,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.normalise = normalise
        self.node_features = node_features
        self.edge_features = edge_features
        self.subtract_edge_mean = subtract_edge_mean

        # Node normalisation parameters
        # self.node_counter = torch.zeros(1).to(self.device)
        self.register_buffer("node_counter", torch.zeros(1))
        # self.node_in_sum = torch.zeros(node_features).to(self.device)
        self.register_buffer("node_in_sum", torch.zeros(node_features))
        # self.node_in_squaresum = torch.zeros(node_features).to(self.device)
        self.register_buffer("node_in_squaresum", torch.zeros(node_features))
        # self.node_in_std = torch.ones(node_features).to(self.device)
        self.register_buffer("node_in_std", torch.ones(node_features))
        # self.node_in_mean = torch.zeros(node_features).to(self.device)
        # self.register_buffer("node_in_mean", torch.zeros(node_features))

        # Output normalisation parameters
        self.node_out_sum = torch.zeros(out_features).to(self.device)
        self.node_out_squaresum = torch.zeros(out_features).to(self.device)
        self.node_out_std = torch.ones(out_features).to(self.device)
        self.node_out_mean = torch.zeros(out_features).to(self.device)

        # Edge normalisation parameters
        self.edge_counter = torch.zeros(1).to(self.device)
        self.edge_in_sum = torch.zeros(edge_features).to(self.device)
        self.edge_in_squaresum = torch.zeros(edge_features).to(self.device)
        self.edge_in_std = torch.ones(edge_features).to(self.device)
        self.edge_in_mean = torch.zeros(edge_features).to(self.device)
        self.edge_in_mean = torch.zeros(edge_features).to(self.device)

    def update_in_normalisation(self, x, edge_attr=None):
        if self.normalise:
            # Node normalisation
            tmp = torch.sum(x, dim=0)
            self.node_in_sum += tmp
            self.node_in_squaresum += tmp * tmp
            self.node_counter += torch.Tensor([x.shape[0]])
            self.node_in_mean = self.node_in_sum / self.node_counter
            self.node_in_std = torch.sqrt(
                (
                    (self.node_in_squaresum / self.node_counter)
                    - (self.node_in_sum / self.node_counter) ** 2
                )
            )

            # Edge normalisation
            if edge_attr is not None:
                tmp = torch.sum(edge_attr, dim=0)
                self.edge_in_sum += tmp
                self.edge_in_squaresum += tmp * tmp
                self.edge_counter += torch.Tensor([edge_attr.shape[0]])
                self.edge_in_mean = self.edge_in_sum / self.edge_counter
                self.edge_in_std = torch.sqrt(
                    (
                        (self.edge_in_squaresum / self.edge_counter)
                        - (self.edge_in_sum / self.edge_counter) ** 2
                    )
                )

    def in_normalise(self, x, edge_attr=None):
        if self.normalise:
            # x = x.detach()
            x = torch.sub(x, self.node_in_mean)
            x = torch.div(x, self.node_in_std)

            # Edge normalisation
            if edge_attr is not None:
                if self.subtract_edge_mean:
                    edge_attr = torch.sub(edge_attr, self.edge_in_mean)
                edge_attr = torch.div(edge_attr, self.edge_in_std)

        return x, edge_attr

    def update_out_normalisation(self, x):
        if self.normalise:
            tmp = torch.sum(x, dim=0)
            self.node_out_sum += tmp
            self.node_out_squaresum += tmp * tmp
            self.node_out_mean = self.node_out_sum / self.node_counter
            self.node_out_std = torch.sqrt(
                (
                    (self.node_out_squaresum / self.node_counter)
                    - (self.node_out_sum / self.node_counter) ** 2
                )
            )

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
