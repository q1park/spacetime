import copy
import torch
import numpy as np

from torch import nn
from spacetime.utils import Logger

#===================================
# optimizer wrapper:
#===================================

class OptimAction:    
    def __init__(self, model, warmups, lr_init, h_tol, max_iters):
        self.clip = lambda x: np.clip(x, *(1e-4,1e-2))
        self.warmups = warmups
        self._incr = lr_init/(self.warmups+1)
        self._c_warm = None
        
        self._iter = 0
        self._epoch = 0
        self._step = 0
        
        self.lr_max = self.clip(lr_init)
        self.lr = 0.0 if self.warmups>0 else self.lr_max
        
        self.max_iters = max_iters
        self.h_tol = h_tol
        self.quit = False
        
        self.optimizer = torch.optim.Adam(list(model.parameters()), lr=self.lr)
            
    def _update_lr(self):
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr
            
    def _step_rate(self):
        if self._step < self.warmups:
            self.lr = self._incr*self._step
            self._update_lr()
        
    def _iter_rate(self, ctrl):
        if not self._step < self.warmups:
            if self._c_warm is None:
                self._c_warm = ctrl.c
            lr_decay = 1.0/(1.0+np.log10(ctrl.c)-np.log10(self._c_warm))
            self.lr = self.clip(lr_decay*self.lr_max)
            self._update_lr()
            
    def check_quit(self, ctrl):
        if self._iter >= self.max_iters or ctrl.h <= self.h_tol:
            self.quit = True
        
    def step(self):
        self._step_rate()
        self._step += 1
        self.optimizer.step()
        
    def epoch(self):
        self._epoch += 1
        
    def iterate(self, ctrl):
        self._iter_rate(ctrl)
        self._iter += 1
        self._epoch = 0
        self.check_quit(ctrl)
        
    def logger(self):
        return {'lr':self.lr}
    
#===================================
# lagrange multiplier loss:
#===================================
        
class LagrangianLoss(nn.Module):
    def __init__(self, kinetic, interaction, ctrl, opt=None):
        super(LagrangianLoss, self).__init__()
        self.kinetic = kinetic
        self.interaction = interaction
        
        self.ctrl = ctrl
        self.opt = opt
        
    def __call__(self, x, z, xx, adj):
        loss_kld, loss_nll = self.kinetic(x, z, xx)
        
        if self.opt is not None:
            loss_action, loss_reg = self.interaction(adj)
            loss = loss_action + loss_reg + loss_kld + loss_nll
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
    
    def next_epoch(self):
        kinetic_log, interaction_log = self.kinetic.end_epoch(), self.interaction.end_epoch()
        self.ctrl.update(kinetic_log, interaction_log, self.interaction.coeffs.c)
        self.opt.epoch()
        return kinetic_log, interaction_log
        
    def next_iter(self):
        self.interaction.coeffs.update(self.ctrl)
        self.opt.iterate(self.ctrl)
    
#===================================
# loss components:
#===================================

class KineticLoss(nn.Module):
    params = ('elbo', 'kld', 'nll')
    
    def __init__(self):
        super(KineticLoss, self).__init__()
        self.log = Logger(*KineticLoss.params)
        self.kld = None
        self.nll = None
        
    def __call__(self, x, z, xx):
        self.kld = KineticLoss.kld(z)
        self.nll = KineticLoss.nll(x, xx)
        
        self.log.append(self.logger(self.kld.item(), self.nll.item()))
        return self.kld, self.nll
    
    def end_epoch(self):
        return_log = copy.deepcopy(self.log)
        self.log.clear()
        return return_log
    
    def logger(self, kld, nll):
        return dict(zip(KineticLoss.params, (kld+nll, kld, nll)))
    
    @staticmethod
    def kld(z):
        return torch.pow(z, 2).sum()/(2*z.size(0))
    
    @staticmethod
    def nll(x, xx):
        return torch.pow(xx-x, 2).sum()/(2*x.size(0))
            
class InteractionLoss(nn.Module):
    params = ('h', 'l1')
    
    def __init__(self, acyc, coeffs):
        super(InteractionLoss, self).__init__()
        self.log = Logger(*InteractionLoss.params)
        self.norm = lambda x: torch.sum(torch.abs(x))
        self.acyc = acyc
        self.coeffs = coeffs
        
        self.adj = None
        self.quad = None
        self.reg = None
        
    @classmethod
    def poly(cls, n_nodes, coeffs):
        acyc = lambda x: InteractionLoss.h_poly(x, n_nodes)
        return cls(acyc, coeffs)
    
    @classmethod
    def ordered(cls, node_dict, coeffs):
        acyc = lambda x: InteractionLoss.h_ordered(x, node_dict)
        return cls(acyc, coeffs)
    
    def __call__(self, adj):
        self.quad = self.coeffs.l*self.acyc(adj) + 0.5*self.coeffs.c*self.acyc(adj)**2
        self.reg = self.coeffs.tau*self.norm(adj) + self.coeffs.tr*torch.trace(adj**2)
        
        self.adj = adj.detach()
        self.log.append(self.logger())
        return self.quad, self.reg
    
    def end_epoch(self):
        return_log = copy.deepcopy(self.log)
        self.log.clear()
        return return_log
    
    def logger(self):
        return {'h':self.acyc(self.adj).item(), 'l1':self.norm(self.adj).item()}
    
    @staticmethod
    def h_poly(adj, n_nodes):
        alpha = (1/n_nodes)
        x = torch.eye(n_nodes).double()+alpha*(adj**2)
        return torch.trace(torch.matrix_power(x, n_nodes))-n_nodes
    
    @staticmethod
    def h_ordered(adj, node_dict):
        h, step = 0.0, 0
        for t, X in node_dict.items():
            h += torch.sum(adj[step:,step:step+len(X)]**2)
            step += len(X)
        return h
    
#===================================
# param update helpers:
#===================================

class Coefficients:
    params = ('l', 'c')
    
    def __init__(self, l_init, c_init, tau_init, tr_init):        
        self.l = l_init
        self.c = c_init
        self.tau = tau_init
        self.tr = tr_init
        
    def update(self, ctrl):
        if ctrl.h <= ctrl.gamma*ctrl.last_h:
            ctrl.last_h = ctrl.h
            self.l += self.c*ctrl.h
        elif ctrl.loss > 1.5*ctrl.min_loss:
            ctrl.last_h = ctrl.h
            self.l += self.c*ctrl.h
        else:
            self.c *= ctrl.eta
            
    def logger(self):
        return dict(zip(Coefficients.params, (self.l, self.c)))
    
class Control:    
    def __init__(self, gamma, eta):
        self.gamma = gamma
        self.eta = eta
        
        self.min_loss = np.inf
        self.last_h = np.inf
        
        self.loss = None
        self.h = None
        self.l1 = None
        self.c = None
        
    def update(self, kinetic_log, interaction_log, coeff_c):
        self.loss = kinetic_log.mean()['elbo']
        self.h = interaction_log.last()['h']
        self.l1 = interaction_log.last()['l1']
        self.c = coeff_c
        
        if self.loss < self.min_loss:
            self.min_loss = self.loss