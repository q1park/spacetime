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
    x = torch.eye(n_nodes).double()+(1/n_nodes)*A**2
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
        self.criterion = nn.KLDivLoss(reduction='sum')
        
    def bmm(self, adj, x):
        return torch.einsum('ij,ajc->aic', adj, x)
        
    def __call__(self, adj, preds, data, z_train):
#         loss_nll = torch.abs(self.criterion(preds, data.double())/data.size(0))
        loss_nll = torch.pow(preds-data, 2).sum()/(2*data.size(0))
        loss_kld = (z_train**2).sum()/(2*z_train.size(0))
        
        self.nll += [loss_nll.item()]
        self.kld += [loss_kld.item()]
        self.adj = adj
        
        if self.opt is not None:
            h_A = self.opt.constraint(adj)
            
            loss_action = self.opt.l*h_A + 0.5*self.opt.c*h_A**2
            loss_reg = 100.*torch.trace(adj**2)# + self.opt.tau*torch.sum(adj**2)
            
            loss = loss_action + loss_reg + loss_nll# + loss_kld
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
    def __init__(self, model, args, h_factor=0.25, c_factor=10.0, warmups=0):
        self.optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        self.constraint = args.constraint
        self._iter = 0
        self._epoch = 0
        self._step = 0
        
        self.original_lr = args.lr
        self._c_warmup = None
        
        self.max_iter = args.max_iters
        self.h_tol = args.h_tol
        self.tau = args.tau
        self.h_factor = h_factor
        self.c_factor = c_factor
        self.lr_range = (1e-4,1e-2)
        
        self.l = args.l
        self.c = args.c
        self.h = np.inf
        self.adj = None
        
        self.warmups = warmups
        self._incr = self.original_lr/(warmups+1)
        self.lr = 0.0 if warmups>0 else self.original_lr
        
        self.min_elbo = np.inf
        self.min_kld = np.inf
        self.min_nll = np.inf
        self.best_epoch = 0
        self.best_adj = None
        
        self.log = {k:[] for k in ('elbo', 'kld', 'nll', 'lr', 'l', 'c', 'h')}
        
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr
        
    def show_adj(self, threshold=0.0):
        clip = nn.Threshold(threshold, 0.0)
        adj = self.best_adj.detach()
        return clip(adj)-clip(-adj)
    
    def update_warmup(self):
        rate = self._incr*self._step
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.lr = rate
        
    def update_lr(self):
        if self._c_warmup is None:
            self._c_warmup = self.c
            
        rate = self.original_lr/(1.0+np.log10(self.c)-np.log10(self._c_warmup))
        rate = np.clip(rate, *self.lr_range)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.lr = rate
        
    def update_lagrange(self, adj):
        h = self.constraint(adj).item()
        if h<=self.h_factor*self.h or self.log['elbo'][-1]>1.5*self.min_elbo:
            self.h = h
            self.l += self.c*h
        else:
            self.c*=self.c_factor
        
    def iterate(self):
        if self._step < self.warmups:
            self.update_lagrange(self.adj)
            self.update_warmup()
        else:
            
            self.update_lagrange(self.adj)
            self.update_lr()
        self._iter += 1
        self._epoch = 0
        
    def epoch(self, kld, nll, adj):
        self.log['lr'] += [self.lr]
        self.log['l'] += [self.l]
        self.log['c'] += [self.c]
        
        nll = np.mean(nll)
        kld = np.mean(kld)
        
        elbo = nll#+kld
        h_A = self.constraint(adj).item()
        
        self.log['nll'] += [nll]
        self.log['kld'] += [kld]
        
        self.log['elbo'] += [elbo]
        self.log['h'] += [h_A]
        
        self.adj = adj
        if elbo < self.min_elbo:
            self.min_elbo = elbo
            self.best_adj = adj
            self.best_epoch = self._epoch
        if nll < self.min_nll:
            self.min_nll = nll
        if kld < self.min_kld:
            self.min_kld = kld
        
        
        self._epoch += 1
        
    def step(self):
        self.optimizer.step()
        if self._step < self.warmups:
            self.update_warmup()
        self._step += 1
        
def train(model, train_loader, loss):
    t = time.time()

    model.train()    
    loss.opt.update_lr()

    for batch_idx, (data, _) in enumerate(train_loader):
        preds, z_train, origin_A = model(data.double())  
        if torch.sum(preds != preds):
            raise ValueError('nan error\n')
        if torch.sum(origin_A != origin_A):
            raise ValueError('nan error\n')

        loss(origin_A, preds, data, z_train)
    loss.end_epoch()
   
    
def truth_evaluation(true_adj, optimizer, threshold, norm=False):
    test_adj = optimizer.show_adj(threshold).numpy()
    if norm:
        test_adj /= np.max(test_adj)
    _,_,_,shd,_ = count_accuracy(true_adj.numpy(), test_adj)
    err = adjacency_error(true_adj.numpy(), optimizer.show_adj(0.0).numpy())
    return shd, err