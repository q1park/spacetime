####################################################################
### Ripped heavily from https://github.com/fishmoon1234/DAG-GNN
####################################################################

import numpy as np
import networkx as nx
from copy import deepcopy as c

class Simulator:
    def __init__(self, adj, **kwargs):
        self.adj = adj
        self.data = None
        self.__dict__.update(kwargs)
        
    @classmethod
    def random(cls, n_nodes, degree, graph_type='erdos-renyi', w_range=(0.5, 2.0), force_positive=False, seed=0):
        print("simulating a random %s-degree %s dag with range %s (seed %s)"%(degree, graph_type, w_range, seed))
        np.random.seed(seed)

        if graph_type == 'erdos-renyi':
            prob = float(degree) / (n_nodes - 1)
            B = np.tril((np.random.rand(n_nodes, n_nodes) < prob).astype(float), k=-1)
        elif graph_type == 'barabasi-albert':
            m = int(round(degree / 2))
            B = np.zeros([n_nodes, n_nodes])
            bag = [0]
            for ii in range(1, n_nodes):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        else:
            raise ValueError('unknown graph type')

        # random permutation
        P = np.random.permutation(np.eye(n_nodes, n_nodes))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)

        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[n_nodes, n_nodes])
        if not force_positive:
            U[np.random.rand(n_nodes, n_nodes) < 0.5] *= -1
        adj = (B_perm != 0).astype(float)*U
        return cls(adj)
    
    @classmethod
    def ordered(cls, structure, degree, graph_type='erdos-renyi', w_range=(0.5, 2.0), force_positive=False, seed=0):
        print("simulating an ordered %s-degree %s dag with range %s (seed %s)"%(degree, graph_type, w_range, seed))
        node_list = [j for i in structure.values() for j in i]
        n_slice = list(map(len, structure.values()))

        np.random.seed(seed)
        d = len(node_list)
        B = np.zeros([d, d])

        if graph_type == 'erdos-renyi':
            prob = float(degree) / (d - 1)        
            ndone = 0
            for n in n_slice[:-1]:
                for ii in range(ndone, ndone+n):
                    B[ii, ndone+n:] = (np.random.rand(d-n-ndone) < prob).astype(float)
                ndone += n
        elif graph_type == 'barabasi-albert':
            m = int(round(degree / 2))
            bag = c(node_dict[0])
            ndone = 0
            for n in n_slice[:-1]:
                for ii in range(len(node_dict[0]), d):
                    dest = np.random.choice(bag, size=m)
                    for jj in dest:
                        B[jj, ii] = 1
                    bag.append(ii)
                    bag.extend(dest)
                ndone += n
        else:
            raise ValueError('unknown graph type')

        # random permutation
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        if not force_positive:
            U[np.random.rand(d, d) < 0.5] *= -1
        adj = (B != 0).astype(float)*U
        return cls(adj, structure=structure)
    
    def sem(self, n, linear_type, noise_scale=1.0, sem_type='linear-gauss', x_dims=1, seed=0):
        print("simulating %s samples from a %s sem with %s causal effects"%(n, sem_type, linear_type))
        np.random.seed(seed)
        G = nx.DiGraph(self.adj)
        X = np.zeros([n, self.adj.shape[0], x_dims])

        for j in list(nx.topological_sort(G)):
            parents = list(G.predecessors(j))

            if linear_type == 'linear':
                eta = X[:, parents, 0].dot(self.adj[parents, j])
            elif linear_type == 'nonlinear_1':
                eta = np.cos(X[:, parents, 0] + 1).dot(self.adj[parents, j])
            elif linear_type == 'nonlinear_2':
                eta = (X[:, parents, 0]+0.5).dot(self.adj[parents, j])
            else:
                raise ValueError('unknown linear data type')

            if sem_type == 'linear-gauss':
                if linear_type == 'linear':
                    X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
                elif linear_type == 'nonlinear_1':
                    X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
                elif linear_type == 'nonlinear_2':
                    X[:, j, 0] = 2.*np.sin(eta) + eta + np.random.normal(scale=noise_scale, size=n)
            elif sem_type == 'linear-exp':
                X[:, j, 0] = eta + np.random.exponential(scale=noise_scale, size=n)
            elif sem_type == 'linear-gumbel':
                X[:, j, 0] = eta + np.random.gumbel(scale=noise_scale, size=n)
            else:
                raise ValueError('unknown sem type')
        self.data = X