####################################################################
### Ripped heavily from https://github.com/fishmoon1234/DAG-GNN
####################################################################

import numpy as np
import networkx as nx
from copy import deepcopy

def labels_indices(node_list):
    index_dict = dict(zip(node_list, range(len(node_list))))
    label_dict = dict(map(reversed, index_dict.items()))
    return index_dict, label_dict

class Adjacency:
    @classmethod
    def dag(cls, adj=None, nodes=None):
        if type(adj) == np.ndarray:
            if type(nodes) == list:
                node_list = nodes
            elif type(nodes) == dict:
                node_list = [j for i in nodes.values() for j in i]
            else:
                node_list = list(range(adj.shape[0]))
            index_dict, label_dict = labels_indices(node_list)
            
            if not np.allclose(adj, np.triu(adj, k=1)):
                adj_sorted, sorted_nodes = cls.sort(adj, node_list)
            else:
                adj_sorted, sorted_nodes = adj, node_list
            order = cls.struct(adj_sorted)
        else:
            if type(nodes) == list:
                node_list, node_dict = nodes, {0:nodes}
            elif type(nodes) == dict:
                node_list, node_dict = [j for i in nodes.values() for j in i], nodes
            else:
                raise KeyError("you must provide an adajacency matrix or a node structure")
            index_dict, label_dict = labels_indices(node_list)
            order = {t:list(map(index_dict.get, xx)) for t, xx in nodes.items()}
            adj_sorted, sorted_nodes = np.zeros(2*[len(node_list)]), node_list
        return adj_sorted, order, index_dict, label_dict
    
    @staticmethod
    def sort(adj, node_list = None):
        if node_list == None:
            node_list = list(range(adj.shape[0]))

        graph = nx.DiGraph(adj)
        origin_nodes = [i for i, x in enumerate(adj.T) if len(np.nonzero(x)[0]) == 0]
        sorted_list = origin_nodes + [x for x in list(nx.topological_sort(graph)) if x not in origin_nodes]
        node_list, sorted_list = np.array(node_list), np.array(sorted_list)

        adj_sorted = nx.to_numpy_array(graph, nodelist=list(sorted_list))
        sorted_nodes = list(node_list[sorted_list])
        return adj_sorted, sorted_nodes

    @staticmethod
    def struct(adj):
        if not np.allclose(adj, np.triu(adj, k=1)):
            raise TypeError("adjacency matrix is not upper triangular")

        itime, n_obs = 0, 0
        order = {itime:list()}

        for i, col in enumerate(adj.T):
            if len(np.nonzero(col[n_obs:])[0]) == 0:
                order[itime] += [i]
            else:
                n_obs += len(order[itime])
                itime += 1
                order[itime] = [i]
        return order

class Simulator:
    @classmethod
    def random_dag(cls, node_list, **kwargs):
        defaults = {'degree':3, 'graph_type':'erdos-renyi', 'w_range':(0.5, 2.0), 'force_positive':False, 'seed':0}
        if set(kwargs.keys())-set(defaults.keys()):
            raise KeyError("Allowed keys: %s"%list(defaults.keys()))
        defaults.update(kwargs)
        return cls.simulate_random_dag(node_list, **defaults)
    
    @classmethod
    def ordered_dag(cls, node_dict, **kwargs):
        defaults = {'degree':3, 'graph_type':'erdos-renyi', 'w_range':(0.5, 2.0), 'force_positive':False, 'seed':0}
        if set(kwargs.keys())-set(defaults.keys()):
            raise KeyError("Allowed keys: %s"%list(defaults.keys()))
        defaults.update(kwargs)
        return cls.simulate_ordered_dag(node_dict, **defaults)
        
    @classmethod
    def sem(cls, graph, **kwargs):
        defaults = {'n':10000, 'x_dims':1, 'sem_type':'linear-gauss', 
                    'linear_type':'nonlinear_2', 'noise_scale':1.0, 'seed':0}
        if set(kwargs.keys())-set(defaults.keys()):
            raise KeyError("Allowed keys: %s"%list(defaults.keys()))
        defaults.update(kwargs)
        return cls.simulate_sem(graph, **defaults)
        
    @staticmethod
    def simulate_random_dag(node_list, degree, graph_type, w_range=(0.5, 2.0), force_positive=False, sort=True, seed=0):
        """Simulate random DAG with some expected degree.
        Args:
            node_list: list containing node labels
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)
        Returns:
            G: weighted DAG
        """
        print("simulating a random %s-degree %s dag with range %s (seed %s)"%(degree, graph_type, w_range, seed))
        np.random.seed(seed)
        d = len(node_list)

        if graph_type == 'erdos-renyi':
            prob = float(degree) / (d - 1)
            B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
        elif graph_type == 'barabasi-albert':
            m = int(round(degree / 2))
            B = np.zeros([d, d])
            bag = [0]
            for ii in range(1, d):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        elif graph_type == 'full':  # ignore degree, only for experimental use
            B = np.tril(np.ones([d, d]), k=-1)
        else:
            raise ValueError('unknown graph type')

        # random permutation
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)

        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        if not force_positive:
            U[np.random.rand(d, d) < 0.5] *= -1
        adj = (B_perm != 0).astype(float)*U

        adj_sorted, sorted_nodes = Adjacency.sort(adj, node_list)
        index_dict, label_dict = labels_indices(sorted_nodes)
        order = Adjacency.struct(adj_sorted)
        return adj_sorted, order, index_dict, label_dict

    @staticmethod
    def simulate_ordered_dag(node_dict, degree, graph_type, w_range=(0.5, 2.0), force_positive=False, seed=0):
        """Simulate ordered DAG with some expected degree.
        Args:
            node_dict: dictionary containing node orders by label
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)
        Returns:
            G: weighted DAG
        """
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
        elif graph_type == 'barabasi-albert':
            m = int(round(degree / 2))
            bag = deepcopy(node_dict[0])
            ndone = 0
            for n in n_slice[:-1]:
                for ii in range(len(node_dict[0]), d):
                    dest = np.random.choice(bag, size=m)
                    for jj in dest:
                        B[jj, ii] = 1
                    bag.append(ii)
                    bag.extend(dest)
                ndone += n
        elif graph_type == 'full':  # ignore degree, only for experimental use
            ndone = 0
            for n in n_slice[:-1]:
                for ii in range(ndone, ndone+n):
                    B[ii, ndone+n:] = np.ones(d-n-ndone)
                ndone += n
        else:
            raise ValueError('unknown graph type')

        # random permutation
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        if not force_positive:
            U[np.random.rand(d, d) < 0.5] *= -1
        adj = (B != 0).astype(float)*U
        index_dict, label_dict = labels_indices(node_list)
        order = {t:list(map(index_dict.get, xx)) for t, xx in node_dict.items()}
        return adj, order, index_dict, label_dict

    @staticmethod
    def simulate_sem(G, n, x_dims, sem_type, linear_type, noise_scale=1.0, seed=0):
        """Simulate samples from SEM with specified type of noise.
        Args:
            G: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM
        Returns:
            X: [n,d] sample matrix
        """
        print("simulating %s samples from a %s sem with %s causal effects"%(n, sem_type, linear_type))
        np.random.seed(seed)
        W = nx.to_numpy_array(G)
        d = W.shape[0]
        X = np.zeros([n, d, x_dims])

        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d

        for j in ordered_vertices:
            parents = list(G.predecessors(j))

            if linear_type == 'linear':
                eta = X[:, parents, 0].dot(W[parents, j])
            elif linear_type == 'nonlinear_1':
                eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
            elif linear_type == 'nonlinear_2':
                eta = (X[:, parents, 0]+0.5).dot(W[parents, j])
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

        if x_dims > 1 :
            for i in range(x_dims-1):
                X[:, :, i+1] = np.random.normal(scale=noise_scale, size=1)*X[:, :, 0] 
                X[:, :, i+1] += np.random.normal(scale=noise_scale, size=1) 
                X[:, :, i+1] += np.random.normal(scale=noise_scale, size=(n, d))
            X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] 
            X[:, :, 0] += np.random.normal(scale=noise_scale, size=1) 
            X[:, :, 0] += np.random.normal(scale=noise_scale, size=(n, d))
        return X