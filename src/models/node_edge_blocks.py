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
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.meta import MetaLayer
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add

# from src.data.dataset import SequentialNBodyDataModule, OneStepNBodyDataModule
from torch_geometric.utils import dropout_adj

# from torch_geometric_temporal.nn import GConvLSTM, GCLSTM, TGCN


class node_mlp_1(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
    ):
        # edge_features: dimension of input edge features
        super(node_mlp_1, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(
                in_features=node_features + edge_features, out_features=hidden_size
            ),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            # BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            # BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # batch: [N] with max entry B - 1.
        if edge_attr is not None:
            row, col = edge_index
            # Aggregate edge attributes
            edge_attr = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
            # Concatenate
            out = torch.cat([x, edge_attr], dim=1)
        else:
            out = x

        # Update dynamic node attributes
        out = self.node_mlp_1(out)
        # Concatenate with static attributes and return
        return out


class node_mlp_out(nn.Module):
    # Output level node update function. Produces target attributes
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
        out_features: int = 4,
    ):
        super(node_mlp_out, self).__init__()
        self.out_features = out_features
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(
                in_features=node_features + edge_features, out_features=hidden_size
            ),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            # BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            # BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=self.out_features),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        if edge_attr is not None:
            # Aggregate edge attributes
            edge_attr = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
            # Concatenate
            out = torch.cat([x, edge_attr], dim=1)
        else:
            out = x

        # Update dynamic node attributes
        out = self.node_mlp_1(out)
        # Return delta dynamics
        return out


class edge_mlp_1(nn.Module):
    # Input edge update function
    def __init__(
        self,
        node_features: int = 5,
        edge_features: int = 1,
        hidden_size: int = 128,
        dropout: float = 0.0,
        latent_edge_features: int = 32,
        **kwargs
    ):
        super(edge_mlp_1, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.latent_edge_features = latent_edge_features

        self.message_mlp = nn.Sequential(
            nn.Linear(
                in_features=node_features * 2 + edge_features, out_features=hidden_size
            ),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            # BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            # BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=latent_edge_features),
        )

    def forward(self, src, dest, edge_attr, edge_index=None, u=None, batch=None):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # batch: [E] with max entry B - 1.

        # Concatenate input values
        input = torch.cat([src, dest, edge_attr], dim=1)
        messages = self.message_mlp(
            input
        )  # Output size: [n_edges, latent_edge_features]
        return messages


class edge_rnn_1(nn.Module):
    # Input edge update function
    def __init__(
        self,
        node_features: int = 5,
        edge_features: int = 1,
        dropout: float = 0.0,
        rnn_size: int = 20,
        num_layers: int = 1,
    ):
        super(edge_rnn_1, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.dropout = dropout if num_layers > 1 else 0.0
        self.rnn_size = rnn_size

        self.message_rnn = nn.GRU(
            input_size=node_features * 2 + edge_features,
            hidden_size=rnn_size,
            num_layers=num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward(
        self,
        src,
        dest,
        edge_attr,
        hidden,
        edge_index=None,
        u=None,
        batch=None,
    ):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # batch: [E] with max entry B - 1.

        # Concatenate input values
        input = torch.cat([src, dest, edge_attr], dim=1)
        # Add empty seq_len dimension
        input = input.unsqueeze(1)
        # Compute messages
        messages, hidden = self.message_rnn(input, hidden)
        return messages.squeeze(), hidden


class edge_mlp_latent(nn.Module):
    # Input edge update function
    def __init__(
        self,
        node_features: int = 5,
        hidden_size: int = 128,
        dropout: float = 0.0,
        latent_edge_features: int = 32,
    ):
        super(edge_mlp_latent, self).__init__()
        self.latent_edge_features = latent_edge_features
        self.node_features = node_features
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.message_mlp = nn.Sequential(
            nn.Linear(
                in_features=node_features * 2 + latent_edge_features,
                out_features=hidden_size,
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=latent_edge_features),
        )

    def forward(self, src, dest, edge_attr, edge_index=None, u=None, batch=None):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # batch: [E] with max entry B - 1.
        # Concatenate input values
        input = torch.cat([src, dest, edge_attr], dim=1)
        messages = self.message_mlp(
            input
        )  # Output size: [n_edges, latent_edge_features]
        return messages


class node_mlp_latent(nn.Module):
    # Input node update function.
    # Assumes edge attributes have been updated
    def __init__(
        self,
        hidden_size: int,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
    ):
        super(node_mlp_latent, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(
                in_features=node_features + edge_features, out_features=hidden_size
            ),
            nn.Dropout(p=dropout, inplace=True),
            nn.ReLU(),
            BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=node_features),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        if edge_attr is not None:
            # Aggregate edge attributes
            edge_attr = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
            # Concatenate
            out = torch.cat([x, edge_attr], dim=1)
        else:
            out = x

        # Update dynamic node attributes
        out = self.node_mlp_1(out)

        return out


class node_mlp_encoder(nn.Module):
    # Input node update function.
    # Assumes edge attributes have been updated
    def __init__(
        self,
        hidden_size: int,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
    ):
        super(node_mlp_encoder, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(
                in_features=node_features + edge_features, out_features=hidden_size
            ),
            nn.Dropout(p=dropout, inplace=True),
            nn.ReLU(),
            BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # batch: [N] with max entry B - 1.
        if edge_attr is not None:
            row, col = edge_index
            # Aggregate edge attributes
            edge_attr = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
            # Concatenate
            out = torch.cat([x, edge_attr], dim=1)
        else:
            out = x

        # Update dynamic node attributes
        out = self.node_mlp_1(out)
        return out


class node_rnn_1(nn.Module):
    # Input node update function.
    # Assumes edge attributes have been updated
    def __init__(
        self,
        node_features: int = 5,
        dropout: float = 0.0,
        edge_features: int = 0,
        rnn_size: int = 20,
        num_layers: int = 1,
        out_features: int = 7,
        hidden_size: int = 64,
    ):
        super(node_rnn_1, self).__init__()
        self.out_features = out_features
        self.rnn_size = rnn_size
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0

        self.node_mlp = nn.Sequential(
            nn.Linear(
                in_features=node_features + edge_features, out_features=hidden_size
            ),
            nn.ReLU(),
            BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            BatchNorm(in_channels=hidden_size),
        )

        self.node_rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=rnn_size,
            num_layers=num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward(self, x, edge_index, edge_attr, u, batch, hidden):
        # x: [N, F_x], where N is the number of nodes.
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        # Aggregate edge attributes
        edge_attr = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
        # Concatenate
        out = torch.cat([x, edge_attr], dim=1)
        out = self.node_mlp(out)
        # Add extra dimension
        out = out.unsqueeze(1)
        out, hidden = self.node_rnn(out, hidden)
        return out.squeeze(), hidden


class node_rnn_2(nn.Module):
    # Input node update function.
    # Assumes edge attributes have been updated
    def __init__(
            self,
            node_features: int = 5,
            dropout: float = 0.0,
            edge_features: int = 0,
            rnn_size: int = 20,
            num_layers: int = 1,
            out_features: int = 7,
            hidden_size: int = 64,
    ):
        super(node_rnn_2, self).__init__()
        self.out_features = out_features
        self.rnn_size = rnn_size
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0

        self.node_rnn = nn.GRU(
            input_size=node_features + edge_features,
            hidden_size=rnn_size,
            num_layers=num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(
                in_features=rnn_size, out_features=hidden_size
            ),
            nn.ReLU(),
            # BatchNorm(in_channels=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )

    def forward(self, x, edge_index, edge_attr, u, batch, hidden):
        # x: [N, F_x], where N is the number of nodes.
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        # Aggregate edge attributes
        edge_attr = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
        # Concatenate
        out = torch.cat([x, edge_attr], dim=1)
        # Add extra dimension
        out = out.unsqueeze(1)
        out, hidden = self.node_rnn(out, hidden)
        out = self.node_mlp(out.squeeze())
        return out, hidden