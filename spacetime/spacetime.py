import copy
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

class SpaceTime:
    def __init__(self, adj):
        self.adj = adj
        self.n_nodes = adj.shape[-1]
        
    def _mutilate(self, *nodes):
        mask = torch.ones(self.adj.shape)
        for node in nodes:
            mask[:, node:node+1]*=torch.zeros((self.n_nodes, 1))
        return mask
    
    def time_scores(self):
        n_nodes = self.adj.shape[0]
        I = torch.eye(n_nodes).double()
        expA = torch.matrix_power(I+(1/n_nodes)*torch.tanh(self.adj)**2, n_nodes)
        scores = torch.div(1.0, torch.sum(expA, dim=1))-torch.div(1.0, torch.sum(expA, dim=0))
        return scores
    
    def mutilate(self, *nodes):
        adj = copy.deepcopy(self.adj)
        return adj*self._mutilate(*nodes)
    
    def topk(self, k):
        n_bottom = int(self.n_nodes**2-k)
        adj = copy.deepcopy(self.adj).view(-1)
        adj[torch.topk(torch.abs(adj), n_bottom, largest=False).indices]=0.
        return adj.view(self.adj.shape)
    
    def sort(self):
        return torch.argsort(self.time_scores(), dim=-1, descending=False)
    
    def norm(self):
        return self.adj/torch.abs(self.adj).max()
    
    def threshold(self, threshold):
        return nn.Threshold(threshold, 0.0)(self.adj)-nn.Threshold(threshold, 0.0)(-self.adj)
    
    def layout(self):
        positions = dict()
        scores = self.time_scores().tolist()
        x_min, x_max = min(scores), max(scores)
        
        for node, t in enumerate(scores):
            positions[node] = np.array([t, np.random.uniform(x_min, x_max)])
        return positions
                
    def draw_graph(self, labels=None):
        G = nx.DiGraph(self.adj.numpy())
        labels = {i:i for i in range(self.adj.shape[0])} if labels is None else labels
        pos = self.layout()
        nx.draw_networkx(G, with_labels=False, pos=pos, node_size=800, node_color='gray')
        nx.draw_networkx_labels(G, pos=pos, labels=labels)
        plt.axis('off')
        plt.show()
        
class Data:
    def __init__(self, data):
        self.data = data
    
    def loader(self, batch_size):
        train_data = TensorDataset(torch.Tensor(self.data), torch.Tensor(self.data))
        train_data_loader = DataLoader(train_data, batch_size=batch_size)
        test_data_loader = DataLoader(train_data, batch_size=len(train_data))

        return train_data_loader, test_data_loader