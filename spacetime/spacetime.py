import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

from spacetime.simulate import labels_indices, Adjacency, Simulator

class SpaceTime:
    def __init__(self, **kwargs):
        defaults = {'adj':None, 'order':dict(), 'label_dict':dict(), 'data':None}
        
        if set(kwargs.keys())-set(defaults.keys()):
            raise KeyError("Allowed keys: %s"%list(defaults.keys()))
            
        defaults.update(kwargs)
        self.graph = nx.DiGraph(defaults['adj'])
        self.order = defaults['order']
        self.label_dict = defaults['label_dict']
        nx.set_node_attributes(self.graph, self.label_dict, name = 'label')
            
    @classmethod
    def from_adjacency(cls, adj, node_list = None):
        adj_sorted, order, index_dict, label_dict = Adjacency.dag(adj, node_list)
        return cls(adj=adj_sorted, order=order, label_dict=label_dict)

    @classmethod
    def from_spacelike(cls, node_list, simulate = False, **kwargs):
        if simulate:
            adj, order, index_dict, label_dict = Simulator.random_dag(node_list, **kwargs)
        else:
            adj, order, index_dict, label_dict = Adjacency.dag(nodes=node_list)
        return cls(adj=adj, order=order, label_dict=label_dict)
        
    @classmethod
    def from_spacetime(cls, node_dict, simulate = False, **kwargs):
        if simulate:
            adj, order, index_dict, label_dict = Simulator.ordered_dag(node_dict, **kwargs)
        else:
            adj, order, index_dict, label_dict = Adjacency.dag(nodes=node_dict)
        return cls(adj=adj, order=order, label_dict=label_dict)
    
    def load_adjacency(self, adj, node_list = None):
        adj_sorted, order, index_dict, label_dict = Adjacency.dag(adj, node_list)
        self.graph = nx.DiGraph(adj_sorted)
        self.order = order
        self.label_dict = label_dict
        nx.set_node_attributes(self.graph, self.label_dict, name = 'label')
        
    def topological_sort(self, node_list=None):
        if node_list is not None:
            if len(node_list)!=self.show_adj().shape[0]:
                raise ValueError("node list must have length %s"%self.show_adj().shape[0])
        elif len(nx.get_node_attributes(self.graph, 'label')) == 0:
            node_list = list(self.graph.nodes)
        else:
            node_list = list(nx.get_node_attributes(self.graph, 'label').values())
        adj, order, index_dict, label_dict = Adjacency.dag(self.show_adj(), node_list)
        self.graph = nx.DiGraph(adj)
        self.order = order
        self.label_dict = label_dict
        nx.set_node_attributes(self.graph, self.label_dict, name = 'label')
    
    def layout(self):
        positions = dict()
        if len(self.order) == 0:
            self.topological_sort()
            
        for t, step in self.order.items():
            for x, node in enumerate(step):
                shift = np.random.uniform(0,0.2) if x%2==0 else 0
                shiftt = np.random.uniform(0,0.2) if t%2==0 else 0
                positions[node] = np.array([t+shift, x+shiftt])
        return positions
                
    def draw_graph(self):
        pos = self.layout()
        node_labels = nx.get_node_attributes(self.graph,'label')
        edge_labels = nx.get_edge_attributes(self.graph,'causal')
        nx.draw_networkx(self.graph, with_labels=False, pos=pos, node_size=800, node_color='gray')
        nx.draw_networkx_labels(self.graph, pos=pos, labels=node_labels)
        nx.draw_networkx_edge_labels(self.graph, pos=pos,edge_labels=edge_labels)
        plt.axis('off')
        plt.show()
        
    def show_adj(self, around=3):
        return np.around(nx.to_numpy_array(self.graph), around)
    
    def torch_data(self, data):
        return nn.Parameter(torch.FloatTensor(data))
    
    def torch_loader(self, data, batch_size=1000):
        feat_train, feat_test = torch.FloatTensor(data), torch.FloatTensor(data)

        train_data = TensorDataset(feat_train, feat_train)
        test_data = TensorDataset(feat_test, feat_test)

        train_data_loader = DataLoader(train_data, batch_size=batch_size)
        test_data_loader = DataLoader(test_data, batch_size=len(test_data))

        return train_data_loader, test_data_loader

    def torch_graph(self):
        adj = self.show_adj(around=10).copy()
        return nn.Parameter(torch.from_numpy(adj).double())