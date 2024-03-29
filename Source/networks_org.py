#----------------------------------------------------------------------
# Definition of the neural network architectures
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

import torch
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, GCNConv, PPFConv, MetaLayer, EdgeConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min
import numpy as np

# Edge convolution layer
class EdgeLayer(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(EdgeLayer, self).__init__(aggr='mean') #  "Mean" aggregation.
        self.mlp = Sequential(Linear(2 * in_channels, mid_channels),
                       ReLU(),
                       Linear(mid_channels, mid_channels),
                       ReLU(),
                       Linear(mid_channels, out_channels))
        self.messages = 0.
        self.input = 0.

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        input = torch.cat([x_i, x_j - x_i], dim=-1)  # tmp has shape [E, 2 * in_channels]

        self.input = input
        self.messages = self.mlp(input)

        return self.messages

#--------------------------------------------
# General Graph Neural Network architecture
#--------------------------------------------
class ModelGNN(torch.nn.Module):
    def __init__(self, node_features, global_features, n_layers, k_nn, hidden_channels=300, latent_channels=100, loop=False):
        super(ModelGNN, self).__init__()
        # Graph layers
        layers = []
        in_channels = node_features
        for i in range(n_layers):
            lay = EdgeLayer(in_channels, hidden_channels, latent_channels)
            layers.append(lay)
            in_channels = latent_channels

        self.layers = ModuleList(layers)

        lin_in = latent_channels + global_features
        self.lin = Sequential(Linear(lin_in, latent_channels),
                              ReLU(),
                              Linear(latent_channels, latent_channels),
                              ReLU(),
                              Linear(latent_channels, 1))

        self.k_nn = k_nn
        self.pooled = 0.
        self.h = 0.
        self.loop = loop

    def forward(self, data):

        x, pos, batch, u = data.x, data.pos, data.batch, data.u

        # Get edges using positions by computing the kNNs or the neighbors within a radius
        #edge_index = knn_graph(pos, k=self.k_nn, batch=batch, loop=self.loop)
        edge_index = radius_graph(pos, r=self.k_nn, batch=batch, loop=self.loop)

        # Start message passing
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index)
            self.h = x
            x = x.relu()
            
        # Mix different global pooling layers
        # addpool = global_add_pool(x, batch) # [num_examples, hidden_channels]
        # maxpool = global_max_pool(x, batch)
        meanpool = global_mean_pool(x, batch)
        self.pooled = torch.cat([meanpool, u], dim=1)
        # Final linear layer
        return self.lin(self.pooled)
