import torch
import time
import math
import numpy as np
import networkx as nx

from torch.autograd import Variable

#===================================
# loss functions:
#===================================

# compute constraint h(A) value
def _h_A(A, m):
    x = torch.eye(m).double()+ torch.div(A*A, m)
    expm_A = torch.matrix_power(x, m)
    return torch.trace(expm_A) - m

def _h_A_ordered(A, node_dict):
    h_A = 0
    block_row, block_col = 0, 0

    for t, time_slice in node_dict.items():
        block_width = len(time_slice)
        block = A[block_row:, block_col:block_col+block_width]
        h_A += torch.sum(block*block)
        
        block_row += block_width
        block_col += block_width
    return h_A

def stau(w, tau):
    w1 = torch.nn.Threshold(0.,0.)(torch.abs(w)-tau)
    return torch.sign(w)*w1

# matrix loss: makes sure at least A connected to another parents for child
def A_connect_loss(A, tol, z):
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss +=  2 * tol - torch.sum(torch.abs(A[:,i])) - torch.sum(torch.abs(A[i,:])) + z * z
    return loss

# element loss: make sure each A_ij > 0
def A_positive_loss(A, z_positive):
    result = - A + z_positive * z_positive
    loss =  torch.sum(result)

    return loss

def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5

def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))

#===================================
# training:
#===================================

def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR, MIN_LR = 1e-2, 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr

def train(lambda_A, c_A, optimizer, scheduler, encoder, decoder, train_loader, args):
    t = time.time()
    elbo_train = []
    nll_train = []

    encoder.train()
    decoder.train()
    optimizer.step()
    scheduler.step()

    # update optimizer
    optimizer, lr = update_optimizer(optimizer, args.lr, c_A)

    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data).double()
        optimizer.zero_grad()

        enc_x, z_train, origin_A, z_gap, z_positive, myA, Wa = encoder(data)  # logits is of size: [num_sims, z_dims]
        dec_x, preds = decoder(z_train, origin_A, Wa)
        variance = 0.
        
        if torch.sum(preds != preds):
            print('nan error\n')

        # compute h(A)
        if args.ordered_graph:
            h_A = _h_A_ordered(origin_A, args.node_dict)
        else:
            h_A = _h_A(origin_A, args.data_variable_size)
    
        # reconstruction accuracy loss
        loss_elbo = nll_gaussian(preds, data, variance) + kl_gaussian_sem(z_train)
        loss_nll = nll_gaussian(preds, data, variance)
        
        loss = loss_elbo
        loss += lambda_A*h_A + 0.5*c_A*h_A*h_A + 100.*torch.trace(origin_A*origin_A)
        loss += args.tau_A*torch.sum(torch.abs(origin_A))

        # other loss terms
        if args.use_A_connect_loss:
            connect_gap = A_connect_loss(origin_A, args.graph_threshold, z_gap)
            loss += lambda_A*connect_gap + 0.5*c_A*connect_gap*connect_gap

        if args.use_A_positiver_loss:
            positive_gap = A_positive_loss(origin_A, z_positive)
            loss += 0.1*(lambda_A*positive_gap + 0.5*c_A*positive_gap*positive_gap)
            
        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, args.tau_A*lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        elbo_train.append(loss_elbo.item())
        nll_train.append(loss_nll.item())

    return np.mean(elbo_train), np.mean(nll_train), graph, origin_A