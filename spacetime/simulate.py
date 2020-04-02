####################################################################
### Ripped heavily from https://github.com/fishmoon1234/DAG-GNN
####################################################################

import numpy as np
import networkx as nx

import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

def simulate_random_dag(d: int,
                        degree: float,
                        graph_type: str,
                        w_range: tuple = (0.5, 2.0),
                        seed: int = 0) -> nx.DiGraph:
    """Simulate random DAG with some expected degree.

    Args:
        d: number of nodes
        degree: expected node degree, in + out
        graph_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)

    Returns:
        G: weighted DAG
    """
    np.random.seed(seed)
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
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G

def simulate_ordered_dag(node_dict: dict,
                         degree: float,
                         graph_type: str,
                         w_range: tuple = (0.5, 2.0),
                         seed: int = 0) -> nx.DiGraph:
    """Simulate ordered DAG with some expected degree.

    Args:
        node_dict: dictionary containing node orders
        degree: expected node degree, in + out
        graph_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)

    Returns:
        G: weighted DAG
    """
    np.random.seed(seed)
    nperslice = list(map(len, node_dict.values()))
    d = sum(nperslice)
    B = np.zeros([d, d])
    
    if graph_type == 'erdos-renyi':
        prob = float(degree) / (d - 1)        
        ndone = 0
        for n in nperslice[:-1]:
            for ii in range(ndone, ndone+n):
                B[ii, ndone+n:] = (np.random.rand(d-n-ndone) < prob).astype(float)
            ndone += n
                    
    elif graph_type == 'barabasi-albert':
        m = int(round(degree / 2))
        bag = node_dict[0]
        ndone = 0
        for n in nperslice[:-1]:
            for ii in range(len(node_dict[0]), d):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[jj, ii] = 1
                bag.append(ii)
                bag.extend(dest)
            ndone += n

    elif graph_type == 'full':  # ignore degree, only for experimental use
        ndone = 0
        for n in nperslice[:-1]:
            for ii in range(ndone, ndone+n):
                B[ii, ndone+n:] = np.ones(d-n-ndone)
            ndone += n
    else:
        raise ValueError('unknown graph type')
    # random permutation

    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G

def simulate_sem(G: nx.DiGraph,
                 n: int, x_dims: int,
                 sem_type: str,
                 linear_type: str,
                 noise_scale: float = 1.0,
                 seed: int = 0) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM

    Returns:
        X: [n,d] sample matrix
    """
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
            X[:, :, i+1] = np.random.normal(scale=noise_scale, size=1)*X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
        X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
    return X

def simulate_data(graph_type, degree, sem_type, linear_type, sample_size, variable_size, x_dims):
    # generate data
    G = simulate_random_dag(variable_size, degree, graph_type)
    X = simulate_sem(G, sample_size, x_dims, sem_type, linear_type)
    
    return G, X

def torch_loader(G, X, batch_size=1000):
    feat_train = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)

    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    test_data = TensorDataset(feat_test, feat_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=len(test_data))

    return train_data_loader, test_data_loader