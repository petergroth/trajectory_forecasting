from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch_geometric.utils
from torch import Tensor, nn
from torch_geometric.nn import (GATConv, GatedGraphConv, GCNConv,
                                MessagePassing, Sequential)
from torch_geometric.nn.meta import MetaLayer
from torch_geometric.nn.norm import BatchNorm, PairNorm
# from src.data.dataset import SequentialNBodyDataModule, OneStepNBodyDataModule
from torch_geometric.utils import dropout_adj
from torch_scatter import scatter_add, scatter_mean


class node_mlp_1(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
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


class node_mlp_out_global(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        out_features: int = 4,
    ):
        super(node_mlp_out_global, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(in_features=node_features, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=out_features),
        )

    def forward(self, x):
        # Update dynamic node attributes
        out = self.node_mlp_1(x)
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
        **kwargs,
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

        # Concatenate input values
        if edge_attr is not None:
            input = torch.cat([src, dest, edge_attr], dim=1)
        else:
            input = torch.cat([src, dest], dim=1)
        messages = self.message_mlp(input)
        return messages


class edge_rnn_1(nn.Module):
    def __init__(
        self,
        edge_features: int = 1,
        dropout: float = 0.0,
        rnn_size: int = 20,
        num_layers: int = 1,
        rnn_type: str = "LSTM",
    ):
        super(edge_rnn_1, self).__init__()
        self.edge_features = edge_features
        self.dropout = dropout if num_layers > 1 else 0.0
        self.rnn_size = rnn_size

        self.edge_history = eval(f"nn.{rnn_type}")(
            input_size=edge_features,
            hidden_size=rnn_size,
            num_layers=num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward(
        self,
        edge_attr,
        hidden,
        edge_index,
        x_size,
    ):
        # Split rows and columns
        rows, cols = edge_index
        # Sum over all edge attributes per node.
        input = scatter_add(edge_attr, rows, dim=0, dim_size=x_size)
        input = input.unsqueeze(1)
        # Compute messages
        messages, hidden = self.edge_history(input, hidden)
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
            nn.LeakyReLU(),
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
            nn.Linear(in_features=rnn_size, out_features=hidden_size),
            nn.LeakyReLU(),
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


class node_rnn_simple(nn.Module):
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
        rnn_type: str = "GRU",
    ):
        super(node_rnn_simple, self).__init__()
        self.out_features = out_features
        self.rnn_size = rnn_size
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0

        self.node_rnn = eval(f"nn.{rnn_type}")(
            input_size=node_features + edge_features,
            hidden_size=rnn_size,
            num_layers=num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward(self, x, hidden, edge_index=None, edge_attr=None, u=None, batch=None):
        # If using also encoding edge attributes
        if self.edge_features > 0:
            row, col = edge_index
            # Aggregate edge attributes if present
            edge_attr = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
            # Concatenate
            out = torch.cat([x, edge_attr], dim=1)
        else:
            # Only encoding node history
            out = x

        # Add extra dimension
        out = out.unsqueeze(1)
        out, hidden = self.node_rnn(out, hidden)
        return out.squeeze(), hidden


class node_gat_in(nn.Module):
    # Input node update function.
    def __init__(
        self,
        node_features: int = 5,
        dropout: float = 0.0,
        out_features: int = 64,
        heads: int = 4,
        edge_features: int = 1,
        norm: bool = False,
    ):
        super(node_gat_in, self).__init__()
        self.dropout = dropout
        self.norm = norm
        self.gat = GATConv(
            in_channels=node_features,
            out_channels=out_features,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=edge_features,
        )
        if norm:
            self.pairnorm = PairNorm(scale_individually=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # Dropout + GAT
        out = F.dropout(x, p=self.dropout)
        out = F.elu(self.gat(x=out, edge_index=edge_index, edge_attr=edge_attr))
        if self.norm:
            out = self.pairnorm(out, batch)
        return out


class node_gat_out(nn.Module):
    def __init__(
        self,
        node_features: int = 5,
        dropout: float = 0.0,
        out_features: int = 64,
        heads: int = 4,
        edge_features: int = 1,
    ):
        super(node_gat_out, self).__init__()
        self.dropout = dropout
        self.gat = GATConv(
            in_channels=node_features,
            out_channels=out_features,
            heads=heads,
            dropout=dropout,
            concat=False,
            edge_dim=edge_features,
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # Dropout + GAT
        out = F.dropout(x, p=self.dropout)
        out = self.gat(x=out, edge_index=edge_index, edge_attr=edge_attr)
        return out


class node_gcn(nn.Module):
    def __init__(
        self,
        node_features: int = 5,
        dropout: float = 0.0,
        hidden_size: int = 64,
        out_features: int = 4,
        edge_features: int = 1,
        skip: bool = False,
    ):
        super(node_gcn, self).__init__()
        assert edge_features == 1
        self.dropout = dropout
        self.skip = skip
        self.gcn_in = GCNConv(
            in_channels=node_features,
            out_channels=hidden_size,
        )
        out_in = hidden_size + node_features if skip else hidden_size
        self.gcn_out = GCNConv(in_channels=out_in, out_channels=out_features)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = F.relu(
            self.gcn_in(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_attr.squeeze() if edge_attr is not None else None,
            )
        )
        out = F.dropout(out, p=self.dropout)
        if self.skip:
            out = torch.cat([x, out], dim=-1)
        out = self.gcn_out(
            x=out,
            edge_index=edge_index,
            edge_weight=edge_attr.squeeze() if edge_attr is not None else None,
        )

        return out


class road_encoder(nn.Module):
    # Architecture is based on the Trajectron++'s map encoder:
    # https://github.com/StanfordASL/Trajectron-plus-plus
    def __init__(
        self, width: int = 300, hidden_size: int = 41, in_map_channels: int = 8
    ):
        super(road_encoder, self).__init__()
        hidden_channels = [20, 10, 5, 1]
        strides = [1, 1, 1, 1]
        filters = [5, 3, 3, 3]
        self.width = width

        self.conv_modules = nn.ModuleList()
        self.bn_modules = nn.ModuleList()
        x_dummy = torch.ones((1, in_map_channels, width, width))

        for i in range(4):
            self.conv_modules.append(
                nn.Conv2d(
                    in_channels=in_map_channels if i == 0 else hidden_channels[i - 1],
                    out_channels=hidden_channels[i],
                    kernel_size=filters[i],
                    stride=strides[i],
                )
            )
            self.bn_modules.append(nn.BatchNorm2d(hidden_channels[i]))
            x_dummy = self.conv_modules[i](x_dummy)

        self.fc_1 = nn.Linear(in_features=x_dummy.numel(), out_features=hidden_size)
        # self.fc_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)

    def forward(self, u):
        # Add single color channel dimension
        assert u.shape[-1] == self.width
        assert u.shape[-2] == self.width
        for i, conv_layer in enumerate(self.conv_modules):
            u = conv_layer(u)
            u = F.leaky_relu(self.bn_modules[i](u))
        # u = F.leaky_relu(self.fc_1(torch.flatten(u, start_dim=1)))
        # out = self.fc_2(u)
        out = self.fc_1(torch.flatten(u, start_dim=1))

        return out
