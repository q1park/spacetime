import torch
import numpy as np
from spacetime.utils import Logger

class OptimModule:
    params = ['lr']
    
    def __init__(self, model, lr_init, warmups=0):
        self.log = Logger(*OptimModule.params)
        self.clip = lambda x: np.clip(x, *(1e-4,1e-2))
        
        self.warmups = warmups
        self._incr = lr_init/(self.warmups+1)
        self._end_warmup = None
        
        self.lr_max = self.clip(lr_init)
        self.lr = 0.0 if self.warmups>0 else self.lr_max
        
        self.optimizer = torch.optim.Adam(list(model.parameters()), lr=self.lr)
        
    def _update_lr(self):
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr
            
    def _warmup_rate(self):
        if self._step < self.warmups:
            self.lr = self._incr*self._step
            self._update_lr()
        
    def _decay_rate(self, current):
        if not self._step < self.warmups:
            lr_decay = self._lr_decay(current)
            self.lr = self.clip(lr_decay*self.lr_max)
            self._update_lr()
            
    def _lr_decay(self, current):
        if self._end_warmup is None:
            self._end_warmup = current
        return 1.0/(1.0+np.log10(current)-np.log10(self._end_warmup))
    
    def logger(self):
        return {'lr':self.lr}
    
class OptimGen(OptimModule):
    def __init__(self, model, lr_init, warmups, decay=False):
        super(OptimGen, self).__init__(model, lr_init, warmups)
        self.quit = False
        
        self._step = 0
        self._epoch = 0
        self.decay = decay
            
    def check_quit(self, ctrl):
        pass
        
    def step(self):
        self._step += 1
        self._warmup_rate()
        self.optimizer.step()
        
    def evolve(self):
        self._epoch += 1
        if self.decay:
            self._decay_rate(self._epoch)
        self.log.append(self.logger())

class OptimGraph(OptimModule):
    def __init__(self, model, lr_init, warmups, h_tol, max_iters):
        super(OptimGraph, self).__init__(model, lr_init, warmups)
        self.h_tol = h_tol
        self.max_iters = max_iters
        self.quit = False
        
        self._step = 0
        self._epoch = 0
        self._iter = 0
            
    def check_quit(self, ctrl):
        if self._iter >= self.max_iters or ctrl.h <= self.h_tol:
            self.quit = True
        
    def step(self):
        self._warmup_rate()
        self.optimizer.step()
        self._step += 1
        
    def evolve(self):
        self.log.append(self.logger())
        self._epoch += 1
        
    def iterate(self, ctrl):
        self._decay_rate(ctrl.c)
        self.check_quit(ctrl)
        self._epoch = 0
        self._iter += 1
        
#===================================
# param update helpers:
#===================================

class Coefficients:
    params = ['l', 'c']
    
    def __init__(self, l_init, c_init, tau_init, tr_init):      
        self.log = Logger(*Coefficients.params)
        self.l = l_init
        self.c = c_init
        self.tau = tau_init
        self.tr = tr_init
        
    def update(self, ctrl):
        if ctrl.h <= ctrl.gamma*ctrl.last_h:
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
        
    def update(self, kinetic, interaction):
        self.loss = kinetic.log.recall('elbo')
        self.h = interaction.log.recall('h')
        self.l1 = interaction.log.recall('l1')
        self.c = interaction.coeffs.log.recall('c')
        
        if self.loss < self.min_loss:
            self.min_loss = self.loss