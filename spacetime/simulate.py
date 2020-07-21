####################################################################
### Ripped heavily from https://github.com/fishmoon1234/DAG-GNN
####################################################################

import torch
import numpy as np
import networkx as nx

#===================================
# toy data simulator
#===================================

class Simulator:
    @staticmethod
    def random(n_nodes, degree, graph_type, w_range, seed):
        print("simulating a random %s-degree %s dag with range %s (seed %s)"%(degree, graph_type, w_range, seed))
        np.random.seed(seed)

        if graph_type == 'erdos-renyi':
            prob = float(degree) / (n_nodes - 1)
            B = np.triu((np.random.rand(n_nodes, n_nodes) < prob).astype(float), k=1)
        else:
            raise ValueError('unknown graph type')

        # random permutation
        P = np.random.permutation(np.eye(n_nodes, n_nodes))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)

        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[n_nodes, n_nodes])
        U[np.random.rand(d, d) < 0.5] *= -1
        
        adj = (B_perm != 0).astype(float)*U
        return torch.Tensor(adj)
    
    @staticmethod
    def ordered(node_dict, degree, graph_type, w_range, force_positive, seed):
        print("simulating an ordered %s-degree %s dag with range %s (seed %s)"%(degree, graph_type, w_range, seed))
        node_list = [j for i in node_dict.values() for j in i]
        n_slice = list(map(len, node_dict.values()))

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
                
        elif graph_type == 'causal-chain1':
            assert d >= 4
            B[1:3,:2] = np.eye(2,2)
            B[3:,[0,2]] = 1.0
            
        elif graph_type == 'causal-chain2':
            assert d >= 4
            B[1:3,:2] = np.eye(2,2)
            B[3,[0,2]] = 1.0
            even_skip=False
            
            for node in range(4, d):
                if node%3==0:
                    even_skip = True if node%6==0 else False
                    B[node,[node-1, node-2]] = 1.0
                else:
                    if node%2==0:
                        if even_skip:
                            B[node,[node-2, node-4]] = 1.0
                        else:
                            B[node,[node-1, node-2]] = 1.0
                    else:
                        if even_skip:
                            B[node,[node-1, node-2]] = 1.0
                        else:
                            if node==5:
                                B[node,[node-2, 0]] = 1.0
                            else:
                                B[node,[node-2, node-4]] = 1.0
        else:
            raise ValueError('unknown graph type')

        # random permutation
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        
        adj = (B != 0).astype(float)*U
        return torch.Tensor(adj)

                
    @staticmethod
    def sem(adj, n_samples, linear_type, sem_type, noise_scale, x_dims, seed, f_dict = dict(), mutil=list()):
        np.random.seed(seed)
        graph = nx.DiGraph(adj)
        X = np.zeros([n_samples, adj.shape[0], x_dims])
        print("simulating %s samples from a %s sem with %s causal effects"%(n_samples, sem_type, linear_type))
        
        f_linear = lambda pa, ch: X[:, pa, 0].dot(adj[pa, ch])
        f_nonlinear_1 = lambda pa, ch: np.cos(X[:, pa, 0] + 1).dot(adj[pa, ch])
        f_nonlinear_2 = lambda pa, ch: (X[:, pa, 0]+0.5).dot(adj[pa, ch])
        f_nonlinear_3 = lambda pa, ch: np.cos(X[:, pa, 0] + 1).dot(adj[pa, ch])+0.5
        
        for j in list(nx.topological_sort(graph)):
            parents = list(graph.predecessors(j)) if j not in mutil else list()

            f_type = linear_type if j not in f_dict else f_dict[j][0]
            n_type = sem_type if j not in f_dict else f_dict[j][1]
            n_scale = noise_scale if j not in f_dict else f_dict[j][2]
                
            if f_type == 'linear':
                eta = f_linear(parents, j)
            elif f_type == 'nonlinear_1':
                eta = f_nonlinear_1(parents, j)
            elif f_type == 'nonlinear_2':
                eta = f_nonlinear_2(parents, j)
            elif f_type == 'nonlinear_3':
                eta = f_nonlinear_3(parents, j)
            else:
                raise ValueError('unknown linear data type')

            if n_type == 'linear-gauss':
                noise = np.random.normal(scale=n_scale, size=n_samples)
            elif n_type == 'linear-exp':
                noise = np.random.exponential(scale=n_scale, size=n_samples)
            elif n_type == 'linear-gumbel':
                noise = np.random.gumbel(scale=n_scale, size=n_samples)
            else:
                raise ValueError('unknown sem type') 
            
            if linear_type in ['linear', 'nonlinear_1']:
                X[:, j, 0] = eta + noise
            elif linear_type in ['nonlinear_2', 'nonlinear_3']:
                X[:, j, 0] = 2.*np.sin(eta) + eta + noise
                
        if x_dims > 1 :
            n_normal = lambda x: np.random.normal(scale=noise_scale, size=x)
            
            for i in range(x_dims-1):
                X[:, :, i+1] = n_normal(1)*X[:, :, 0] + n_normal(1) + n_normal((n_samples, adj.shape[0]))
            X[:, :, 0] = n_normal(1)*X[:, :, 0] + n_normal(1) + n_normal((n_samples, adj.shape[0]))
            
        return X