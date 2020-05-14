import numpy as np
from copy import deepcopy

class Parameters:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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

def spacetime_mutilator(spacetime, mutilate=list()):
    mutilated = deepcopy(spacetime)
    
    for edge in list(mutilated.graph.edges()):
        if edge[1] in mutilate:
            mutilated.graph.remove_edge(*edge)
    return mutilated

def graph_clipper(torch_graph, threshold):
    graph_clone = torch_graph.data.clone()
    graph_numpy = graph_clone.numpy()
    graph = graph_numpy.copy()
    graph[np.abs(graph) < threshold] = 0
    return graph_clone, graph_numpy, graph


