import time
import numpy as np
import networkx as nx
import torch
from torch import nn
from torch.autograd import Variable
from spacetime.metrics import count_accuracy, adjacency_error

#===================================
# constraint functions:
#===================================

def h_A(A, n_nodes):
    x = torch.eye(n_nodes).double()+torch.div(A*A, n_nodes)
    return torch.trace(torch.matrix_power(x, n_nodes))-n_nodes

def h_A_timed(A, node_dict):
    h_A, step = 0.0, 0
    for t, xx in node_dict.items():
        h_A += torch.sum(A[step:,step:step+len(xx)]**2)
        step += len(xx)
    return h_A

#===================================
# loss module:
#===================================

class LagrangeLoss(nn.Module):
    def __init__(self, opt=None):
        super(LagrangeLoss, self).__init__()
        self.opt = opt
        self.kld = []
        self.nll = []
        self.adj = None
      
    @staticmethod
    def kl_gauss(preds):
        kl_div = preds**2
        return 0.5*(kl_div.sum()/preds.size(0))
    
    @staticmethod
    def nll_gauss(preds, target, variance):
        mean1, mean2 = preds, target
        neg_log_p = variance + torch.div(torch.pow(mean1-mean2, 2), 2.*np.exp(2.*variance))
        return neg_log_p.sum()/(target.size(0))
        
    def __call__(self, adj, preds, data, z_train, variance=0.0):
        loss_kld = LagrangeLoss.kl_gauss(z_train)
        loss_nll = LagrangeLoss.nll_gauss(preds, data, variance)
        
        self.kld += [loss_kld.item()]
        self.nll += [loss_nll.item()]
        self.adj = adj
        
        if self.opt is not None:
            h_A = self.opt.constraint(adj)
            loss_action = self.opt.l*h_A + 0.5*self.opt.c*h_A**2
            loss_reg = 100.*torch.trace(adj**2) + self.opt.tau*torch.sum(adj**2)
            
            loss = loss_kld + loss_nll + loss_action + loss_reg
            loss.backward()
            
            self.opt.step()
            self.opt.optimizer.zero_grad()
            
    def end_epoch(self):
        assert self.opt is not None
        self.opt.epoch(self.kld, self.nll, self.adj)
        self.kld.clear()
        self.nll.clear()
        self.h_A = None

#===================================
# optimizer wrapper:
#===================================

class ActionOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model, args, h_factor=0.25, c_factor=10.0):
        self.optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        self.constraint = args.constraint
        self._iter = 0
        self._epoch = 0
        self._step = 0
        
        self.original_lr = args.lr
        self.max_iter = args.max_iters
        self.h_tol = args.h_tol
        self.tau = args.tau
        self.h_factor = h_factor
        self.c_factor = c_factor
        self.lr_range = (1e-4,1e-2)
        
        self.lr = args.lr
        self.l = args.l
        self.c = args.c
        self.h = np.inf
        self.adj = None
        
        self.min_elbo = np.inf
        self.min_kld = np.inf
        self.min_nll = np.inf
        self.best_epoch = 0
        self.best_adj = None

        self.log = {k:[] for k in ('elbo', 'kld', 'nll', 'lr', 'l', 'c', 'h')}
        
    def show_adj(self, threshold=0.3):
        graph = self.best_adj.data.clone().numpy()
        graph[np.abs(graph) < threshold] = 0
        return graph
        
        
    def update_lr(self):
        rate = self.original_lr/(np.log10(self.c)+1e-10)
        rate = np.clip(rate, *self.lr_range)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.lr = rate
        
    def update_lagrange(self, adj):
        h = self.constraint(adj).item()
        if h<=self.h_factor*self.h or self.log['elbo'][-1]>2*self.min_elbo:
            self.l += self.c*h
            self.h = h
        else:
            self.c*=self.c_factor
        
    def iterate(self):
        self.update_lagrange(self.adj)
        self.update_lr()
        self._iter += 1
        self._epoch = 0
        
    def epoch(self, kld, nll, adj):
        self.log['lr'] += [self.lr]
        self.log['l'] += [self.l]
        self.log['c'] += [self.c]
        
        kld = np.mean(kld)
        nll = np.mean(nll)
        elbo = kld+nll
        h_A = self.constraint(adj).item()
        
        self.log['kld'] += [kld]
        self.log['nll'] += [nll]
        self.log['elbo'] += [elbo]
        self.log['h'] += [h_A]
        
        self.adj = adj
        if elbo < self.min_elbo:
            self.min_elbo = elbo
            self.best_adj = adj
            self.best_epoch = self._epoch
        if kld < self.min_kld:
            self.min_kld = kld
        if nll < self.min_nll:
            self.min_nll = nll
        
        self._epoch += 1
        
    def step(self):
        self.optimizer.step()
        self._step += 1

#===================================
# training:
#===================================

def train(model, train_loader, loss):
    t = time.time()

    model.train()    
    loss.opt.update_lr()

    for batch_idx, (data, _) in enumerate(train_loader):
        preds, z_train, origin_A, Wa = model(data.double())  
        if torch.sum(preds != preds):
            raise ValueError('nan error\n')
        if torch.sum(origin_A != origin_A):
            raise ValueError('nan error\n')

        loss(origin_A, preds, data, z_train)
    loss.end_epoch()
    
def truth_evaluation(true_adj, optimizer, threshold):
    _, _, _, shd, _ = count_accuracy(true_adj, optimizer.show_adj(threshold=threshold))
    err = adjacency_error(true_adj, optimizer.show_adj(threshold=0.0))
    return shd, err