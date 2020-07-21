import numpy as np
import torch

import torch.nn.functional as F
from torch import nn
from spacetime.optimizers import OptimModule, Coefficients, Control
from spacetime.utils import Logger, print_format, rand_norm
    
#===================================
# loss functions
#===================================

def _kernel(N, V, d, x, y, l):
    x, y = x.unsqueeze(1), y.unsqueeze(0) # (N, 1, V, d), (1, N, V, d)
    tiled_x = x.expand(N, N, V, d)
    tiled_y = y.expand(N, N, V, d)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(-1).mean(-1)/float(V)
    return torch.exp(-kernel_input) + l*torch.eye(N).double()

def _listprod(tensorlist):
    for i, tensor in enumerate(reversed(tensorlist)):
        if i==0:
            prod = tensor
        else:
            prod = torch.matmul(tensor, prod)
    return prod

def _cond_diag(N, V, d, xq, l):
    if xq is None:
        return torch.ones(N, N).double()
    else:
        k_q_cond = _kernel(N, V, d, xq, xq, l=0)
        kt_q_cond_inv = _kernel(N, V, d, xq, xq, l=l).inverse()
        return _listprod([kt_q_cond_inv, k_q_cond, kt_q_cond_inv])

def _cond_cross(N, V, d, xq, xp, l):
    if xq is None and xp is None:
        return torch.ones(N, N).double()
    else:
        k_pq_cond = _kernel(N, V, d, xp, xq, l=0)
        kt_p_cond_inv = _kernel(N, V, d, xp, xp, l=l).inverse()
        kt_q_cond_inv = _kernel(N, V, d, xq, xq, l=l).inverse()
        return _listprod([kt_p_cond_inv, k_pq_cond, kt_q_cond_inv])

class Losses:
    @staticmethod
    def nll(x, xx):
        return torch.pow(xx-x, 2).sum()/(2*x.size(0))
    
    @staticmethod
    def kld(n, x=None):
        mu_n, std_n = torch.mean(n, dim=0), torch.std(n, dim=0)
        if x is None:
            mu_x, std_x = torch.zeros(mu_n.shape), torch.ones(std_n.shape)
        else:
            mu_x, std_x = torch.mean(x, dim=0), torch.std(x, dim=0)
        
        return 0.5*((std_n**2/std_x**2).sum()+((mu_n-mu_x)**2/std_x**2).sum()
                    +torch.log((std_x**2).prod()/(std_n**2).prod()) - n.shape[1])
    
    @staticmethod
    def cmmd(Q, P, endo=(), exo=(), l=0.1):
        if len(endo)==0:
            endo = tuple(range(Q.shape[1]))
        N = Q.shape[0]
        V_endo = len(endo)
        V_exo = len(exo)
        d = Q.shape[2]

        qendo = Q[:,[*endo],:]
        pendo = P[:,[*endo],:]

        qexo = Q[:,[*exo],:] if len(exo)>0 else None
        pexo = P[:,[*exo],:] if len(exo)>0 else None

        Lqq = _kernel(N, V_endo, d, qendo, qendo, l=0)
        Lpp = _kernel(N, V_endo, d, pendo, pendo, l=0)
        Lqp = _kernel(N, V_endo, d, qendo, pendo, l=0)

        Cqq = _cond_diag(N, V_exo, d, qexo, l=l)
        Cpp = _cond_diag(N, V_exo, d, pexo, l=l)
        Cqp = _cond_cross(N, V_exo, d, qexo, pexo, l=l)

        mmd = (1./N**2)*(torch.matmul(Lqq, Cqq).trace() 
                         +torch.matmul(Lpp, Cpp).trace()
                         -2*torch.matmul(Lqp, Cqp).trace())
        return mmd
    
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
        
    @classmethod
    def poly(cls, n_nodes, coeffs, ctrls, opt):
        acyc = lambda x: InteractionLoss.h_poly(x, n_nodes)
        kinetic = ELBOLoss()
        interaction = InteractionLoss(acyc, Coefficients(**coeffs))
        ctrl = Control(**ctrls)
        return cls(kinetic, interaction, ctrl, opt)
    
    @classmethod
    def ordered(cls, node_dict, coeffs, ctrls, opt):
        acyc = lambda x: InteractionLoss.h_ordered(x, node_dict)
        kinetic = ELBOLoss()
        interaction = InteractionLoss(acyc, Coefficients(**coeffs))
        ctrl = Control(**ctrls)
        return cls(kinetic, interaction, ctrl, opt)
        
    def __call__(self, x, z, xx, adj):
        loss_kld, loss_nll = self.kinetic(x, z, xx)
        loss_action, loss_reg = self.interaction(adj)
        
        if self.opt is not None:
            loss = loss_action + loss_reg + loss_nll
#             loss = loss_action + loss_reg + loss_kld + loss_nll
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        else:
            return loss_kld, loss_nll, loss_action, loss_reg
    
    def evolve(self):
        self.kinetic.evolve()
        self.interaction.evolve()
        self.ctrl.update(self.kinetic, self.interaction)
        
        if self.opt is not None:
            self.opt.evolve()
        
    def iterate(self):
        self.interaction.coeffs.update(self.ctrl)
        
        if self.opt is not None:
            self.opt.iterate(self.ctrl)
            
    def iter_summary(self):
        epoch_losses = self.kinetic.log.log['elbo'][-self.opt._epoch:]
        summary = [self.opt._iter, np.argmin(epoch_losses), self.opt._epoch]
        return "Iteration: %s, Best Epoch: %s/%s"%tuple(summary)
    
    def print_summary(self, loss):
        print(3*"*"+self.iter_summary())
        print(3*" "+print_format([self.kinetic.log, self.opt.log], log10=True))
        print(3*" "+print_format([self.interaction.log, self.interaction.coeffs.log], log10=True))
        
    def get_logger(self, var):
        if var in ELBOLoss.params:
            return self.kinetic.log
        elif var in InteractionLoss.params:
            return self.interaction.log
        elif var in Coefficients.params:
            return self.interaction.coeffs.log
        elif var in OptimModule.params:
            return self.opt.log
        else:
            raise ValueError("invalid parameter %s"%var)

#===================================
# loss components:
#===================================

class InfoLoss(nn.Module):
    params = ['elbo', 'mmd', 'nll']
    
    def __init__(self, mask, chain, opt, beta=1.0, gamma=500.0, reg=0.2):
        super(InfoLoss, self).__init__()
        self.log = Logger(*InfoLoss.params)
        self._log = Logger(*InfoLoss.params)
        
        self.mask = mask
        self.chain = tuple(chain)
        self.opt = opt
        
        self.beta = beta
        self.gamma = gamma
        self.reg = reg
        
        self.mmd = None
        self.nll = None
        
    def __call__(self, x, z, zz, xx):
        std_norm = rand_norm(0.0, 1.0, z.shape[0], z.shape[1]).unsqueeze(-1).double()

        self.mmd = self.beta*Losses.cmmd(z, std_norm) 
        self.mmd += self.gamma*(Losses.cmmd(zz, z, endo=self.chain[0:1], exo=self.chain[1:2], l=self.reg)
                                +Losses.cmmd(zz, z, endo=self.chain[1:2], exo=self.chain[2:3], l=self.reg)
                                +Losses.cmmd(zz, z, endo=self.chain[0:1], exo=self.chain[2:3], l=self.reg))
        self.nll = Losses.nll(x, xx)
        
        self._log.append(self.logger())
        
        if self.opt is not None:
            loss = self.mmd + self.nll
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        else:
            return self.mmd, self.nll
    
    def evolve(self):
        self.log.append(self._log.mean())
        self._log.clear()
        
        if self.opt is not None:
            self.opt.evolve()
    
    def logger(self):
        mmd, nll = self.mmd.item(), self.nll.item()
        return dict(zip(InfoLoss.params, (mmd+nll, mmd, nll)))
    
    def print_summary(self):
        print(3*" "+print_format([self.log, self.opt.log], log10=True))
        
    def get_logger(self, var):
        if var in InfoLoss.params:
            return self.log
        elif var in OptimModule.params:
            return self.opt.log
        else:
            raise ValueError("invalid parameter %s"%var)

class ELBOLoss(nn.Module):
    params = ['elbo', 'kld', 'nll']
    
    def __init__(self, opt=None):
        super(ELBOLoss, self).__init__()
        self.log = Logger(*ELBOLoss.params)
        self._log = Logger(*ELBOLoss.params)
        self.kld = None
        self.nll = None
        self.opt = opt
        
    def __call__(self, x, z, xx):
        self.kld = Losses.kld(z)
        self.nll = Losses.nll(x, xx)
        self._log.append(self.logger())
        
        if self.opt is not None:
            loss = self.kld + self.nll
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        else:
            return self.kld, self.nll
    
    def evolve(self):
        self.log.append(self._log.mean())
        self._log.clear()
        
        if self.opt is not None:
            self.opt.evolve()
    
    def logger(self):
        kld, nll = self.kld.item(), self.nll.item()
        return dict(zip(ELBOLoss.params, (kld+nll, kld, nll)))
    
    def print_summary(self):
        print(3*" "+print_format([self.log, self.opt.log], log10=True))
        
    def get_logger(self, var):
        if var in ELBOLoss.params:
            return self.log
        elif var in OptimModule.params:
            return self.opt.log
        else:
            raise ValueError("invalid parameter %s"%var)
            
class InteractionLoss(nn.Module):
    params = ['h', 'l1']
    
    def __init__(self, acyc, coeffs, opt=None):
        super(InteractionLoss, self).__init__()
        self.log = Logger(*InteractionLoss.params)
        self._log = Logger(*InteractionLoss.params)
        self.norm = lambda x: torch.sum(torch.abs(x))
        self.acyc = acyc
        self.coeffs = coeffs
        
        self.adj = None
        self.quad = None
        self.reg = None
        self.opt = opt
        
    @classmethod
    def poly(cls, n_nodes, **coeffs):
        acyc = lambda x: InteractionLoss.h_poly(x, n_nodes)
        return cls(acyc, Coefficients(**coeffs))
    
    @classmethod
    def ordered(cls, node_dict, **coeffs):
        acyc = lambda x: InteractionLoss.h_ordered(x, node_dict)
        return cls(acyc, Coefficients(**coeffs))
    
    def __call__(self, adj):
        self.quad = self.coeffs.l*self.acyc(adj) + 0.5*self.coeffs.c*self.acyc(adj)**2
        self.reg = self.coeffs.tau*self.norm(adj) + self.coeffs.tr*torch.trace(adj**2)
        self.adj = adj.detach()
        self._log.append(self.logger())
        
        if self.opt is not None:
            loss = self.quad + self.reg
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        else:
            return self.quad, self.reg
    
    def evolve(self):
        self.log.append(self._log.last())
        self._log.clear()
        
        self.coeffs.log.append(self.coeffs.logger())
        
        if self.opt is not None:
            self.opt.evolve()
    
    def logger(self):
        return {'h':self.acyc(self.adj).item(), 'l1':self.norm(self.adj).item()}
    
    def get_logger(self, var):
        if var in InteractionLoss.params:
            return self.log
        elif var in OptimModule.params:
            return self.opt.log
        else:
            raise ValueError("invalid parameter %s"%var)
    
    @staticmethod
    def h_poly(adj, n_nodes):
        alpha = (1/n_nodes)
        x = torch.eye(n_nodes)+alpha*(adj**2)
        return torch.trace(torch.matrix_power(x, n_nodes))-n_nodes
    
    @staticmethod
    def h_ordered(adj, node_dict):
        h, step = 0.0, 0
        for t, X in node_dict.items():
            h += torch.sum(adj[step:,step:step+len(X)]**2)
            step += len(X)
        return h