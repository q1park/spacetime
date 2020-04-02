####################################################################
### Ripped heavily from https://github.com/fishmoon1234/DAG-GNN
####################################################################

import math
import numpy as np
import networkx as nx

def count_accuracy(G_true: nx.DiGraph,
                   G: nx.DiGraph,
                   G_und: nx.DiGraph = None) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        G_true: ground truth graph
        G: predicted graph
        G_und: predicted undirected edges in CPDAG, asymmetric

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    B_und = None if G_und is None else nx.to_numpy_array(G_und)
    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size

def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# BiC scores

def compute_BiCScore(G, D):
    '''compute the bic score'''
    # score = gm.estimators.BicScore(self.data).score(self.model)
    origin_score = []
    num_var = G.shape[0]
    for i in range(num_var):
        parents = np.where(G[:,i] !=0)
        score_one = compute_local_BiCScore(D, i, parents)
        origin_score.append(score_one)

    return sum(origin_score)

def compute_local_BiCScore(np_data, target, parents):
    # use dictionary
    sample_size = np_data.shape[0]
    var_size = np_data.shape[1]

    # build dictionary and populate
    count_d = dict()
    if len(parents) < 1:
        a = 1

    # slower implementation
    for data_ind in range(sample_size):
        parent_combination = tuple(np_data[data_ind, parents].reshape(1, -1)[0])
        self_value = tuple(np_data[data_ind, target].reshape(1, -1)[0])
        if parent_combination in count_d:
            if self_value in count_d[parent_combination]:
                count_d[parent_combination][self_value] += 1.0
            else:
                count_d[parent_combination][self_value] = 1.0
        else:
            count_d[parent_combination] = dict()
            count_d[parent_combination][self_value] = 1.0

    # compute likelihood
    loglik = 0.0
    num_parent_state = np.prod(np.amax(np_data[:, parents], axis=0) + 1)
    num_self_state = np.amax(np_data[:, target], axis=0) + 1

    for parents_state in count_d:
        local_count = sum(count_d[parents_state].values())
        for self_state in count_d[parents_state]:
            loglik += count_d[parents_state][self_state] * (
                        math.log(count_d[parents_state][self_state] + 0.1) - math.log(local_count))

    # penality
    num_param = num_parent_state * (num_self_state - 1) 
    bic = loglik - 0.5 * math.log(sample_size) * num_param

    return bic