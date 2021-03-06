from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_geometric.utils
from torch import Tensor, nn
from torch_geometric.nn import (GATConv, GatedGraphConv, GCNConv,
                                MessagePassing, Sequential)
from torch_geometric.nn.meta import MetaLayer
from torch_geometric.utils import dropout_adj
from torch_scatter import scatter_add, scatter_mean

from src.models.node_edge_blocks import *


class ConstantModel(nn.Module):
    # Model that simply returns zeros.
    def __init__(self):
        super(ConstantModel, self).__init__()

    def forward(self, x, edge_index):
        # Identity
        return torch.zeros_like(x[:, :4])


class mlp_forward_model(nn.Module):
    #
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        skip: bool = True,
        aggregate: bool = False,
        **kwargs,
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


class mpnn_forward_model(nn.Module):
    # Message passing neural network. Inspired from <https://arxiv.org/abs/2002.09405v2>
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
        **kwargs,
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


class rnn_mp_hetero(nn.Module):
    # Recurrent message-passing GNN with different agent classes.
    # Not finalised.
    def __init__(
        self,
        hidden_size: int = 64,
        dropout: float = 0.0,
        node_features: int = 5,
        edge_features: int = 0,
        num_layers: int = 1,
        rnn_size: int = 20,
        rnn_edge_size: int = 8,
        out_features: int = 4,
        rnn_type: str = "LSTM",
        latent_edge_features: int = 32,
    ):
        super(rnn_mp_hetero, self).__init__()
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.rnn_edge_size = rnn_edge_size
        self.dropout = dropout
        self.out_features = out_features

        # Node history encoder
        self.node_history_encoder = node_rnn_simple(
            node_features=node_features,
            edge_features=0,
            rnn_size=rnn_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )
        # Edge history encoder
        self.edge_history_encoder = edge_rnn_1(
            edge_features=edge_features,
            rnn_size=rnn_edge_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )
        # Message-passing GN block
        self.edge_encoding = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=rnn_size + rnn_edge_size,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
        )

        node_in = rnn_size + rnn_edge_size + latent_edge_features

        self.node_car_module = nn.Sequential(
            nn.Linear(in_features=node_in, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=out_features),
        )

        self.node_pedestrian_module = nn.Sequential(
            nn.Linear(in_features=node_in, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=out_features),
        )

        self.node_bike_module = nn.Sequential(
            nn.Linear(in_features=node_in, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=out_features),
        )

    def forward(self, x, edge_index, edge_attr, hidden: tuple, u=None, batch=None):
        types = x[:, -5:].bool()
        x = x[:, :-5]

        # Unpack hidden states
        h_node, h_edge = hidden

        # Aggregate and encode edge histories. Shape [n_nodes, rnn_edge_size]
        edge_attr_encoded, h_edge = self.edge_history_encoder(
            edge_attr=edge_attr, hidden=h_edge, edge_index=edge_index, x_size=x.size(0)
        )
        # Encode node histories. Shape [n_nodes, rnn_size]
        x_encoded, h_node = self.node_history_encoder(
            x=x,
            edge_index=edge_index,
            edge_attr=None,
            u=None,
            batch=None,
            hidden=h_node,
        )

        # Concatenate. Shape [n_nodes, rnn_size+rnn_edge_size
        x_concat = torch.cat([x_encoded, edge_attr_encoded], dim=-1)

        # Compute message for each edge
        _, edge_attr, _ = self.edge_encoding(
            x=x_concat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=None,
            batch=None,
            hidden=None,
        )

        # Aggregate messages per node
        row, col = edge_index
        edge_attr = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
        x_agg = torch.cat([x_concat, edge_attr], dim=1)

        # Agent class indexing
        car_idx = (types[:, 1] == 1) + (types[:, 0] == 1) + (types[:, 4] == 1)
        pedestrian_idx = types[:, 2] == 1
        bike_idx = types[:, 3] == 1

        x_car = self.node_car_module(x_agg[car_idx])
        x_pedestrian = self.node_car_module(x_agg[pedestrian_idx])
        x_bike = self.node_car_module(x_agg[bike_idx])

        out = torch.zeros((x.size(0), self.out_features))
        out = out.type_as(x)
        out[car_idx] = x_car
        out[pedestrian_idx] = x_pedestrian
        out[bike_idx] = x_bike

        return out, (h_node, h_edge)


class rnn_gat(nn.Module):
    # Recurrent attention-based GNN
    def __init__(
        self,
        dropout: float = 0.0,
        node_features: int = 5,
        edge_features: int = 0,
        num_layers: int = 1,
        rnn_size: int = 20,
        rnn_edge_size: int = 8,
        out_features: int = 4,
        rnn_type: str = "LSTM",
        heads: int = 4,
    ):
        super(rnn_gat, self).__init__()
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.rnn_edge_size = rnn_edge_size
        self.dropout = dropout

        # Encoding module. Encodes node and edge history separately.
        self.node_history_encoder = node_rnn_simple(
            node_features=node_features,
            edge_features=0,
            rnn_size=rnn_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )
        self.edge_history_encoder = edge_rnn_1(
            edge_features=edge_features,
            rnn_size=rnn_edge_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        self.gat_out = GATConv(
            in_channels=rnn_size + rnn_edge_size,
            out_channels=out_features,
            heads=heads,
            dropout=dropout,
            concat=False,
        )

    def forward(self, x, edge_index, edge_attr, hidden, batch=None, u=None):
        # Unpack hidden states
        h_node, h_edge = hidden
        # Aggregate and encode edge histories. Shape [n_nodes, rnn_edge_size]
        edge_attr_encoded, h_edge = self.edge_history_encoder(
            edge_attr=edge_attr, hidden=h_edge, edge_index=edge_index, x_size=x.size(0)
        )
        # Encode node histories. Shape [n_nodes, rnn_size]
        x_encoded, h_node = self.node_history_encoder(
            x=x,
            edge_index=edge_index,
            edge_attr=None,
            u=None,
            batch=None,
            hidden=h_node,
        )

        # Concatenate. Shape [n_nodes, rnn_size+rnn_edge_size]
        out = torch.cat([x_encoded, edge_attr_encoded], dim=-1)
        # Attention based mechanism to obtain outputs
        out = self.gat_out(out, edge_index=edge_index)
        return out, (h_node, h_edge)


class rnn_mp_gat(nn.Module):
    # Recurrent message-passing GAT network
    def __init__(
        self,
        hidden_size: int = 64,
        dropout: float = 0.0,
        node_features: int = 5,
        edge_features: int = 0,
        num_layers: int = 1,
        rnn_size: int = 20,
        rnn_edge_size: int = 8,
        out_features: int = 4,
        rnn_type: str = "LSTM",
        heads: int = 4,
        latent_edge_features: int = 32,
    ):
        super(rnn_mp_gat, self).__init__()
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.rnn_edge_size = rnn_edge_size

        # Encoding module. Encodes node and edge history separately.
        self.node_history_encoder = node_rnn_simple(
            node_features=node_features,
            edge_features=0,
            rnn_size=rnn_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )
        self.edge_history_encoder = edge_rnn_1(
            edge_features=edge_features,
            rnn_size=rnn_edge_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        self.gat_in = GATConv(
            in_channels=rnn_size + rnn_edge_size,
            out_channels=hidden_size,
            heads=heads,
            dropout=dropout,
            concat=False,
        )

        self.GN = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=hidden_size,
                edge_features=edge_features,
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

    def forward(self, x, edge_index, edge_attr, hidden: tuple, batch=None, u=None):
        # Unpack hidden states
        h_node, h_edge = hidden
        # Aggregate and encode edge histories. Shape [n_nodes, rnn_edge_size]
        edge_attr_encoded, h_edge = self.edge_history_encoder(
            edge_attr=edge_attr, hidden=h_edge, edge_index=edge_index, x_size=x.size(0)
        )
        # Encode node histories. Shape [n_nodes, rnn_size]
        x_encoded, h_node = self.node_history_encoder(
            x=x,
            edge_index=edge_index,
            edge_attr=None,
            u=None,
            batch=None,
            hidden=h_node,
        )

        # Concatenate. Shape [n_nodes, rnn_size+rnn_edge_size]
        x_encoded = torch.cat([x_encoded, edge_attr_encoded], dim=-1)

        # Attention module. Shape [n_nodes, rnn_size+rnn_edge_size]
        x_encoded = self.gat_in(x=x_encoded, edge_index=edge_index)

        # Message passing
        out, _, _ = self.GN(x=x_encoded, edge_index=edge_index, edge_attr=edge_attr)

        return out, (h_node, h_edge)


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
        self.skip = skip

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
        if self.skip:
            x_1 = torch.cat([x, x_1], dim=1)
            edge_attr_1 = torch.cat([edge_attr, edge_attr_1], dim=1)
        # Second block
        out, _, _ = self.GN2(
            x=x_1, edge_index=edge_index, edge_attr=edge_attr_1, u=u, batch=batch
        )
        return out, hidden


class rnn_forward_model_v3(nn.Module):
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
        rnn_type: str = "GRU",
        aggregate: bool = False,
        **kwargs,
    ):
        super(rnn_forward_model_v3, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.edge_features = edge_features
        self.latent_edge_features = latent_edge_features
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.skip = skip
        self.rnn_type = rnn_type
        self.aggregate = aggregate

        self.GN1 = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=node_features,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_rnn_simple(
                node_features=node_features,
                edge_features=latent_edge_features,
                rnn_size=rnn_size,
                dropout=dropout,
                num_layers=num_layers,
                out_features=out_features,
                rnn_type=rnn_type,
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
        if self.skip:
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
        return out, hidden


class rnn_forward_model_v4(nn.Module):
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
        rnn_type: str = "GRU",
        aggregate: bool = False,
        **kwargs,
    ):
        super(rnn_forward_model_v4, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.edge_features = edge_features
        self.latent_edge_features = latent_edge_features
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.skip = skip
        self.rnn_type = rnn_type
        self.aggregate = aggregate

        self.GN1 = GraphNetworkBlock(
            node_model=node_rnn_simple(
                node_features=node_features,
                edge_features=0,
                rnn_size=rnn_size,
                dropout=dropout,
                num_layers=num_layers,
                out_features=out_features,
                rnn_type=rnn_type,
            ),
        )
        GN2_node_input = rnn_size + node_features if skip else rnn_size
        GN2_edge_input = edge_features

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
        # Encode node wise history
        x_1, _, _, hidden = self.GN1(
            x=x,
            edge_index=edge_index,
            edge_attr=None,
            u=u,
            batch=batch,
            hidden=hidden,
        )
        # Concatenation of node and edge attributes
        if self.skip:
            if self.aggregate:
                x_1 = scatter_add(x_1, batch, dim=0)
                x_1 = torch.cat([x, x_1[batch]], dim=1)
            else:
                x_1 = torch.cat([x, x_1], dim=1)
            # edge_attr_1 = torch.cat([edge_attr, edge_attr_1], dim=1)
        # Second block
        out, _, _ = self.GN2(
            x=x_1, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch
        )
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
        **kwargs,
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
        **kwargs,
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


class rnn_message_passing_global_v3(nn.Module):
    # Recurrent message-passing GNN
    def __init__(
        self,
        hidden_size: int = 64,
        dropout: float = 0.0,
        node_features: int = 5,
        edge_features: int = 0,
        num_layers: int = 1,
        rnn_size: int = 20,
        rnn_edge_size: int = 8,
        out_features: int = 4,
        rnn_type: str = "LSTM",
        latent_edge_features: int = 32,
        map_encoding_size: int = 32,
    ):
        super(rnn_message_passing_global_v3, self).__init__()
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.rnn_edge_size = rnn_edge_size
        self.dropout = dropout

        # Node history encoder.
        # Computes a node-wise representation which incorporates the nodes' respective histories.
        self.node_history_encoder = node_rnn_simple(
            node_features=node_features + map_encoding_size,
            edge_features=0,
            rnn_size=rnn_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        # GN-block to compute messages/interactions between nodes
        self.message_gn = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=node_features + map_encoding_size,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            )
        )

        # Interaction history encoder.
        # Computes a node-wise history which incorporates influences from neighbouring nodes using messages.
        self.edge_history_encoder = edge_rnn_1(
            edge_features=latent_edge_features,
            rnn_size=rnn_edge_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        # Output GN
        self.GN_out = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=rnn_size + rnn_edge_size,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_mlp_out(
                hidden_size=hidden_size,
                node_features=rnn_size + rnn_edge_size,
                dropout=dropout,
                edge_features=latent_edge_features,
                out_features=out_features,
            ),
        )

    def forward(self, x, edge_index, edge_attr, u, hidden: tuple, batch=None):
        # x: [n_nodes, node_features]
        # edge_index: [2, n_edges]
        # edge_attr : [n_edges, edge_features]
        # u: [n_nodes, local_map_resolution]

        # Unpack hidden states
        h_node, h_edge = hidden

        #
        x = torch.cat([x, u], dim=-1)

        # Encode node histories. Shape [n_nodes, rnn_size]
        node_history, h_node = self.node_history_encoder(
            x=x, edge_index=None, edge_attr=None, u=None, batch=None, hidden=h_node
        )

        # Compute messages. Shape [n_edges, latent_edge_features]
        _, messages, _ = self.message_gn(
            x=x, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=None
        )

        # Aggregate and encode edge histories. Shape [n_nodes, rnn_edge_size]
        node_interaction_history, h_edge = self.edge_history_encoder(
            edge_attr=messages, hidden=h_edge, edge_index=edge_index, x_size=x.size(0)
        )

        # Concatenate. Shape [n_nodes, rnn_size+rnn_edge_size+map_encoding_size]
        full_node_representation = torch.cat(
            [node_history, node_interaction_history], dim=-1
        )

        # Final node update. [n_nodes, out_features]
        out, _, _ = self.GN_out(
            x=full_node_representation,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=None,
            batch=None,
        )

        return out, (h_node, h_edge)


class rnn_message_passing_global_v4(nn.Module):
    # Recurrent message-passing GNN
    def __init__(
        self,
        hidden_size: int = 64,
        dropout: float = 0.0,
        node_features: int = 5,
        edge_features: int = 0,
        num_layers: int = 1,
        rnn_size: int = 20,
        rnn_edge_size: int = 8,
        out_features: int = 4,
        rnn_type: str = "LSTM",
        latent_edge_features: int = 32,
        map_encoding_size: int = 32,
    ):
        super(rnn_message_passing_global_v4, self).__init__()
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.rnn_edge_size = rnn_edge_size
        self.dropout = dropout

        # Node history encoder.
        # Computes a node-wise representation which incorporates the nodes' respective histories.
        self.node_history_encoder = node_rnn_simple(
            node_features=node_features + map_encoding_size,
            edge_features=0,
            rnn_size=rnn_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        # GN-block to compute messages/interactions between nodes
        self.message_gn = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=node_features + map_encoding_size,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            )
        )

        # Interaction history encoder.
        # Computes a node-wise history which incorporates influences from neighbouring nodes using messages.
        self.edge_history_encoder = edge_rnn_1(
            edge_features=latent_edge_features,
            rnn_size=rnn_edge_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        # Output GN
        self.node_output = node_mlp_out_global(
            hidden_size=hidden_size,
            node_features=rnn_size + rnn_edge_size,
            dropout=dropout,
            out_features=out_features,
        )

    def forward(self, x, edge_index, edge_attr, u, hidden: tuple, batch=None):
        # x: [n_nodes, node_features]
        # edge_index: [2, n_edges]
        # edge_attr : [n_edges, edge_features]
        # u: [n_nodes, local_map_resolution]

        # Unpack hidden states
        h_node, h_edge = hidden

        x = torch.cat([x, u], dim=-1)

        # Encode node histories. Shape [n_nodes, rnn_size]
        node_history, h_node = self.node_history_encoder(
            x=x, edge_index=None, edge_attr=None, u=None, batch=None, hidden=h_node
        )

        # Compute messages. Shape [n_edges, latent_edge_features]
        _, messages, _ = self.message_gn(
            x=x, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=None
        )

        # Aggregate and encode edge histories. Shape [n_nodes, rnn_edge_size]
        node_interaction_history, h_edge = self.edge_history_encoder(
            edge_attr=messages, hidden=h_edge, edge_index=edge_index, x_size=x.size(0)
        )

        # Concatenate. Shape [n_nodes, rnn_size+rnn_edge_size+map_encoding_size]
        full_node_representation = torch.cat(
            [node_history, node_interaction_history], dim=-1
        )

        # Final node update. [n_nodes, out_features]
        out = self.node_output(x=full_node_representation)

        return out, (h_node, h_edge)


class rnn_message_passing_global(nn.Module):
    #   Encoding of node history
    #   Encoding of interaction history with message passing
    #   Concatenation of histories and mapping before node update
    def __init__(
        self,
        hidden_size: int = 64,
        dropout: float = 0.0,
        node_features: int = 5,
        edge_features: int = 0,
        num_layers: int = 1,
        rnn_size: int = 20,
        rnn_edge_size: int = 8,
        out_features: int = 4,
        rnn_type: str = "LSTM",
        latent_edge_features: int = 32,
        map_encoding_size: int = 32,
    ):
        super(rnn_message_passing_global, self).__init__()
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.rnn_edge_size = rnn_edge_size
        self.dropout = dropout

        # Node history encoder.
        # Computes a node-wise representation which incorporates the nodes' respective histories.
        self.node_history_encoder = node_rnn_simple(
            node_features=node_features,
            edge_features=0,
            rnn_size=rnn_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        # GN-block to compute messages/interactions between nodes
        self.message_gn = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=node_features,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            )
        )

        # Interaction history encoder.
        # Computes a node-wise history which incorporates influences from neighbouring nodes using messages.
        self.edge_history_encoder = edge_rnn_1(
            edge_features=latent_edge_features,
            rnn_size=rnn_edge_size,
            dropout=dropout,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        # Output GN
        self.node_output = node_mlp_out_global(
            hidden_size=hidden_size,
            node_features=rnn_size + rnn_edge_size + map_encoding_size,
            dropout=dropout,
            out_features=out_features,
        )

    def forward(self, x, edge_index, edge_attr, u, hidden: tuple, batch=None):
        # x: [n_nodes, node_features]
        # edge_index: [2, n_edges]
        # edge_attr : [n_edges, edge_features]
        # u: [n_nodes, local_map_resolution]

        # Unpack hidden states
        h_node, h_edge = hidden

        # Encode node histories. Shape [n_nodes, rnn_size]
        node_history, h_node = self.node_history_encoder(
            x=x, edge_index=None, edge_attr=None, u=None, batch=None, hidden=h_node
        )

        # Compute messages. Shape [n_edges, latent_edge_features]
        _, messages, _ = self.message_gn(
            x=x, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=None
        )

        # Aggregate and encode edge histories. Shape [n_nodes, rnn_edge_size]
        node_interaction_history, h_edge = self.edge_history_encoder(
            edge_attr=messages, hidden=h_edge, edge_index=edge_index, x_size=x.size(0)
        )

        # Concatenate. Shape [n_nodes, rnn_size+rnn_edge_size+map_encoding_size]
        full_node_representation = torch.cat(
            [node_history, node_interaction_history, u], dim=-1
        )

        # Final node update. [n_nodes, out_features]
        out = self.node_output(x=full_node_representation)

        return out, (h_node, h_edge)
