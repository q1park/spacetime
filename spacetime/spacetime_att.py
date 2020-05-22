import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy as c

import torch
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

class SpaceTime:
    def __init__(self, adj=None, data=None, structure={}, **kwargs):
        self.adj = adj
        self.data = data
        self.structure = structure
        self.__dict__.update(kwargs)
            
    @classmethod
    def simulation(cls, sim):
        kwargs = {k:torch.Tensor(v) if type(v)==np.ndarray else v for k,v in sim.__dict__.items()}
        return cls(**kwargs)
    
    @classmethod
    def data(cls, data, structure={}):
        adj = torch.zeros((data.shape[1], data.shape[1]))
        data=torch.Tensor(data)
        return cls(adj=adj, data=data, structure=structure)
    
    @classmethod
    def adjacency(cls, adj, structure={}):
        adj = torch.Tensor(adj)
        return cls(adj=adj, data=None, structure=structure)
    
    def mutilate(self, *args):
        adj = self.adj.clone()
        for arg in args:
            adj[:,arg] = 0.0
        return adj
    
    def time_scores(self):
        n_nodes = self.adj.shape[0]
        I = torch.eye(n_nodes).double()
        expA = torch.matrix_power(I+(1/n_nodes)*torch.tanh(self.adj)**2, n_nodes)
        scores = torch.div(1.0, torch.sum(expA, dim=1))-torch.div(1.0, torch.sum(expA, dim=0))
        return scores
    
    def sort(self):
        return torch.argsort(self.time_scores(), dim=-1, descending=False)
    
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
    
    def data_loader(self, batch_size=1000):
        train_data = TensorDataset(self.data, self.data)
        train_data_loader = DataLoader(train_data, batch_size=batch_size)
        test_data_loader = DataLoader(train_data, batch_size=len(train_data))

        return train_data_loader, test_data_loader

    def graph_param(self):
        adj = self.show_adj().copy()
        return nn.Parameter(self.adj.double())