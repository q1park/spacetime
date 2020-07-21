import copy
import itertools
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from spacetime.losses import InteractionLoss

#===================================
# Adjacency matrix manipulator
#===================================

class SpaceTime:
    def __init__(self, A):
        self.A = A
        self.n_nodes = A.shape[-1]
        self.eps = 1e-16
        
    def _expm(self, A):
        return torch.matrix_power(torch.eye(self.n_nodes)+(1/A.shape[-1])*torch.tanh(A)**2, self.n_nodes)

    def _directed_scores(self, dim):
        node_mask = torch.ones(self.n_nodes)
        scores = torch.zeros(self.n_nodes)

        for n_node in range(self.n_nodes):
            mask = node_mask.repeat(self.n_nodes).view(self.n_nodes, self.n_nodes)
            if dim==0:
                mask = mask.transpose(-2,-1)
            score = torch.sum(self._expm(self.A*mask), dim=dim)
            node_mask = (score!=1.0).float()
            
            if torch.sum(node_mask)==0:
                break
            else:
                scores += 1/(score)
        return scores

    def time_scores(self, backscore=True):
        time_scores = self._directed_scores(dim=1)
        if backscore:
            time_scores -= self._directed_scores(dim=0)
        return time_scores
    
    def time_order(self):
        sorted_nodes = torch.argsort(self.time_scores(), dim=-1, descending=False).tolist()
        nx_graph = nx.DiGraph(self.A.numpy())
        A_sorted = nx.to_numpy_array(nx_graph, nodelist=sorted_nodes)
        return torch.Tensor(A_sorted)
    
    def topk(self, k):
        A = copy.deepcopy(self.A).reshape(-1)
        A[torch.topk(torch.abs(A), int(self.n_nodes**2-k), largest=False).indices]=0.
        return A.view(self.A.shape)

    def threshold(self, threshold):
        return nn.Threshold(threshold, 0.0)(self.A)-nn.Threshold(threshold, 0.0)(-self.A)
    
    def min_thresh(self):
        for thresh in (np.around(np.arange(0.001, 0.01, 0.001), 4).tolist()
                       +np.around(np.arange(0.01, 0.3, 0.01), 4).tolist()):
            if InteractionLoss.h_poly(self.threshold(thresh), self.n_nodes)==0:
                print("h=0 at threshold {:.4f}".format(thresh))
                break
        return self.threshold(thresh)
    
    def norm(self):
        return self.A/torch.abs(self.A).max()
    
    def normalize(self):
        self.A = torch.abs(self.norm())
        self.A = self.min_thresh()
    
    def binary(self):
        return self.A.abs()/(self.A.abs()+self.eps)
    
    def mutilate(self, *nodes):
        mask = torch.ones(self.A.shape)
        for node in nodes:
            mask[:, node:node+1]*=torch.zeros((self.n_nodes, 1))
            
        return self.A*mask
    
    def layout(self, backscore):
        positions = dict()
        scores = self.time_scores(backscore=backscore).tolist()
        t_min, t_max = min(scores), max(scores)
        t_mean, t_rad = np.mean([t_min, t_max]), 0.5*(t_max-t_min)
        
        positions = {node:np.array([t, np.sqrt(np.abs(t_rad**2-(t-t_mean)**2))]) 
                     for node, t in enumerate(scores)}
        return positions
                
    def draw_graph(self, labels=None, backscore=True):
        G = nx.DiGraph(self.A.numpy())
        labels = {i:i for i in range(self.A.shape[0])} if labels is None else labels
        pos = self.layout(backscore=backscore)
        scale = np.log10(self.n_nodes)
        fig, ax = plt.subplots(1, 1, figsize = (scale*8, scale*4))
        nx.draw_networkx(G, with_labels=False, pos=pos, node_size=600, node_color='gray')
        nx.draw_networkx_labels(G, pos=pos, labels=labels)
        plt.axis('off')
        
    def plot_temp(self, ax, vmin=-1.5, vmax=1.5, center=0.0, norm=False, pos=False):
        A = self.norm() if norm else self.A.numpy()
        if pos:
            A = np.abs(A)

        sns.heatmap(A, xticklabels=range(self.n_nodes), yticklabels=range(self.n_nodes), 
                    square=True, vmin=vmin, vmax=vmax, cbar=False, ax=ax, center=center)

#===================================
# data store
#===================================

class NodeData:
    def __init__(self, data, bins=None, sigma=5, bin_width=0.5):
        self.nodes = tuple(range(data.shape[1]))
        self.info = {node:self.bin_data(node, data, bins, sigma, bin_width) for node in self.nodes}
        self.n_bins = bins
    
    def bin_data(self, node, data, bins, sigma, bin_width):
        edges = self.make_edges(data[:,node,:], sigma, bin_width) if bins is None else bins[node]
        axis = edges[:-1]+np.diff(edges)/2
        return {'data':data[:,node,:], 'edges':edges, 'axis':axis}
    
    def make_edges(self, data, sigma, bin_width):
        n_bins = int(1+2*sigma/bin_width)
        flattened = data.squeeze()
        mu, std = data.mean(), data.std()

        edges = (np.arange(n_bins+1)-0.5*n_bins)*bin_width
        return edges*std+mu
    
    def data(self):
        data_list = [np.expand_dims(info['data'], axis=1) for node, info in self.info.items()]
        return np.hstack(data_list)
    
    def axes(self, *nodes):
        nodes = self.nodes if len(nodes) == 0 else nodes
        return [self.info[node]['axis'] for node in nodes]
    
    def idxs(self, *nodes):
        nodes = self.nodes if len(nodes) == 0 else nodes
        return [range(len(self.info[node]['axis'])) for node in nodes]
    
    def edges(self, *nodes):
        nodes = self.nodes if len(nodes) == 0 else nodes
        return [self.info[node]['edges'] for node in nodes] 
    
    def product_axes(self, *nodes):
        return np.array(np.meshgrid(*self.axes(*nodes))).T.reshape(-1,len(nodes))
    
    def product_idxs(self, *nodes):
        return np.array(np.meshgrid(*self.idxs(*nodes))).T.reshape(-1,len(nodes))
    
    def torch(self):
        return torch.Tensor(self.data()).double()
    
    def loader(self, batch_size, shuffle=False):
        return DataLoader(self.torch(), batch_size=batch_size, shuffle=shuffle)