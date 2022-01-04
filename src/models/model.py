from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_geometric.utils
from torch import Tensor, nn
from torch_geometric.nn.meta import MetaLayer
from torch_scatter import scatter_add, scatter_mean

from src.models.node_edge_blocks import *


class GraphNetworkBlock(MetaLayer):
    # Extension of torch_geometric.nn.meta.MetaLayer to support propagation of hidden states for
    # RNNs
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


class NodeNN(nn.Module):
    # Node-wise MLP with skip connection.
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        n_nodes: int = 10,
        dropout: float = 0.0,
        out_features: int = 7,
    ):
        super(NodeNN, self).__init__()
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

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, u=None):
        # Forward pass
        x_latent = self.mlp_1(x)
        # Skip connection
        x_latent = torch.cat([x, x_latent], dim=1)
        # Second pass
        out = self.mlp_2(x_latent)
        return out


class NodeRNN(nn.Module):
    # Node-wise recurrent MLP with skip connection.
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        rnn_size: int = 20,
        num_layers: int = 1,
        rnn_type: str = "LSTM",
        out_features: int = 4,
    ):
        super(NodeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.out_features = out_features
        self.num_layers = num_layers

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
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features=rnn_size + node_features, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=out_features),
        )

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, u=None, hidden=None):     
        # Forward pass
        node_history, hidden = self.node_history_encoder(x=x, hidden=hidden)
        # Skip connection
        out = torch.cat([x, node_history], dim=1)
        # Second pass
        out = self.mlp_2(out)
        return out, hidden


class AttentionalGNN(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        heads: int = 4,
        middle_gat: bool = False,
        out_features: int = 4,
        edge_features: int = 1,
    ):
        super(AttentionalGNN, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.middle_gat = middle_gat
        self.GN1 = GraphNetworkBlock(
            node_model=node_gat_in(
                node_features=node_features,
                dropout=dropout,
                heads=heads,
                out_features=hidden_size,
                edge_features=edge_features,
            ),
        )

        if middle_gat:
            self.GN2 = GraphNetworkBlock(
                node_model=node_gat_in(
                    node_features=hidden_size * heads,
                    dropout=dropout,
                    heads=heads,
                    out_features=hidden_size,
                    edge_features=edge_features,
                ),
            )
            # Apply skip connection across this layer. Compute new input size
            hidden_size = hidden_size * 2

        self.GN3 = GraphNetworkBlock(
            node_model=node_gat_out(
                node_features=hidden_size * heads,
                dropout=dropout,
                heads=heads,
                out_features=out_features,
                edge_features=edge_features,
            ),
        )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        # Input GAT
        out, _, _ = self.GN1(
            x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch
        )

        if self.middle_gat:
            out_middle, _, _ = self.GN2(
                x=out, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch
            )
            # Skip connection
            out = torch.cat([out, out_middle], dim=-1)

        out, _, _ = self.GN3(
            x=out, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch
        )

        return out


class ConvolutionalGNN(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        node_features: int = 5,
        dropout: float = 0.0,
        skip: bool = False,
        out_features: int = 4,
        edge_features: int = 1,
    ):
        super(ConvolutionalGNN, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.GN1 = GraphNetworkBlock(
            node_model=node_gcn(
                node_features=node_features,
                dropout=dropout,
                skip=skip,
                out_features=out_features,
                edge_features=edge_features,
                hidden_size=hidden_size,
            ),
        )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        # Input GAT
        out, _, _ = self.GN1(
            x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch
        )

        return out


class MPGNN(nn.Module):
    # Implementation of the 'forward GN model' from <https://arxiv.org/abs/1806.01242> by Battaglia et al.
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
        super(MPGNN, self).__init__()
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.dropout = dropout
        self.edge_features = edge_features
        self.latent_edge_features = latent_edge_features

        # Whether to perform a graph-wise node feature aggregation after the fist GN block
        self.aggregate = aggregate

        # GN block with node and edge update functions
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

        # Determine dimensions of second GN block
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


class RMPGNN(nn.Module):
    # Recurrent Message-Passing Graph Neural Network
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
        super(RMPGNN, self).__init__()
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

        # Concatenate. Shape [n_nodes, rnn_size+rnn_edge_size]
        full_node_representation = torch.cat(
            [node_history, node_interaction_history], dim=-1
        )

        # Final node update. [n_nodes, out_features]
        out, _, _ = self.GN_out(x=full_node_representation, edge_index=edge_index, edge_attr=edge_attr,
                                u=None, batch=None)

        return out, (h_node, h_edge)


class LocalRMPGNN(nn.Module):
    # Recurrent message-passing GNN with local map information
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
        super(LocalRMPGNN, self).__init__()
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
        self.GN_out = GraphNetworkBlock(
            edge_model=edge_mlp_1(
                node_features=rnn_size + rnn_edge_size + map_encoding_size,
                edge_features=edge_features,
                hidden_size=hidden_size,
                dropout=dropout,
                latent_edge_features=latent_edge_features,
            ),
            node_model=node_mlp_out(
                hidden_size=hidden_size,
                node_features=rnn_size + rnn_edge_size + map_encoding_size,
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
        out, _, _ = self.GN_out(x=full_node_representation, edge_index=edge_index, edge_attr=edge_attr,
                                u=None, batch=None)

        return out, (h_node, h_edge)

