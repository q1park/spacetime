import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

class arguments:
    def __init__(self, node_dict):
        # data parameters 
        # configurations
        self.node_dict = node_dict
        self.data_variable_size=sum(map(len, self.node_dict.values()))
        self.data_sample_size=20000
        self.noise_scale = 0.1
        self.graph_type='erdos-renyi'
        self.graph_degree=3
        self.graph_sem_type='linear-gauss'
        self.graph_linear_type='nonlinear_2'
        self.edge_types=2
        self.x_dims=1 #changed here
        self.z_dims=1

        # training hyperparameters
        self.optimizer='Adam'
        self.graph_threshold=0.3  # 0.3 is good 0.2 is error prune
        self.tau_A=1e-10
        self.lambda_A=0.
        self.c_A=1
        self.ordered_graph=True
        self.use_A_connect_loss=False
        self.use_A_positiver_loss=False

        self.seed=42
        self.epochs= 10
        self.batch_size=100 # note: should be divisible by sample size otherwise throw an error
        self.lr=3e-3  # basline rate = 1e-3
        self.encoder_hidden=32
        self.decoder_hidden=32
        self.temp=0.5
        self.k_max_iter=1e2

        self.save_folder='logs'

        self.h_tol=1e-8
        self.prediction_steps=10 
        self.lr_decay=200
        self.gamma= 1.0
        self.skip_first=False
        self.var=5e-5
        self.hard=False
        self.prior=False
        self.dynamic_graph=False

def nbins(data, bin_width = 0.1):
    return np.arange(min(data), max(data) + bin_width, bin_width)

def torch_loader(X, batch_size=1000):
    feat_train = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)

    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    test_data = TensorDataset(feat_test, feat_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=len(test_data))

    return train_data_loader, test_data_loader

def torch_graph(numpy_adj, mutilate = list(), flip = list()):
    graph = numpy_adj.copy()
    
    for i, j in mutilate:
        graph[i, j] = 0
        
    for i, j in flip:
        graph[i, j] *= -1.0
        
    return nn.Parameter(torch.from_numpy(graph).double())
