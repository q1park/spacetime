import time
import torch
import numpy as np

from spacetime.spacetime import SpaceTime
from spacetime.losses import KineticLoss, InteractionLoss
from spacetime.metrics import count_accuracy, adjacency_error
from spacetime.utils import check_nan, Logger

class Trainer:
    params = ('l', 'c', 'lr')
    
    def __init__(self, model, loss, data, batch_size, epochs, val=None):
        self.kinetic_log = Logger(*KineticLoss.params)
        self.interaction_log = Logger(*InteractionLoss.params)
        self.param_log = Logger(*Trainer.params)
        
        self.model = model
        self.loss = loss
        self.data, _ = data.loader(batch_size=batch_size)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.val = val
        
    def train(self):
        t_total = time.time()
        while not self.loss.opt.quit:
            self.train_iter()
            
        print("\nTrial finished in %s seconds"%(time.time() - t_total))
        if self.val is not None:
            self.val.print_summary()
        print("-"*75)
            
    def train_iter(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            
        self.print_summary()
        self.loss.next_iter()
        
        if self.val is not None:
            self.val.print_summary()
    
    def train_epoch(self):
        self.model.train()    

        for x, _ in self.data:
            xx, z, adj = self.model(x.double())
            check_nan({'adj':adj, 'preds':xx})
            self.loss(x, z, xx, adj)
            
        if self.val is not None:
            self.val.validate(SpaceTime(adj.detach()))
            
        train_logs = self.loss.next_epoch()
        self.update_logs(*train_logs)
        
    def update_logs(self, kinetic_log, interaction_log):
        self.kinetic_log.append(kinetic_log.mean())
        self.interaction_log.append(interaction_log.last())
        self.param_log.append(self.logger())
        
    def logger(self):
        param_dict = self.loss.interaction.coeffs.logger()
        param_dict.update(self.loss.opt.logger())
        return param_dict
        
    def print_summary(self):
        _line_1 = "***Iteration: {:d}, Best Epoch: {:d}/{:d} || elbo: 10^{:.3f} || kl: 10^{:.3f} || nll: 10^{:.3f}"
        _line_2 = "     h: 10^{:.3f} || l1: 10^{:.3f} || ""l: 10^{:.2f} || c: 10^{:.2f} || lr: 10^{:.3f}"
        
        summarize = lambda x, y: list(map(np.log10, map(x.recall, y)))
        epoch_losses = self.kinetic_log.log['elbo'][-self.loss.opt._epoch:]
        
        print(_line_1.format(*[self.loss.opt._iter, np.argmin(epoch_losses), self.loss.opt._epoch], 
                             *summarize(self.kinetic_log, KineticLoss.params)))
        print(_line_2.format(*summarize(self.interaction_log, InteractionLoss.params),
                             *summarize(self.param_log, Trainer.params)))

class Validator:
    params = ('shd', 'tpr', 'fpr', 'err')
    
    def __init__(self, st_truth):
        self.log = Logger(*Validator.params)
        
        self.st_truth = st_truth
        self.n_nodes = self.st_truth.adj.shape[-1]
        self.n_edges = len(self.st_truth.adj.nonzero())
        self.l1 = torch.sum(torch.abs(self.st_truth.adj)).item()

    def validate(self, st_learned):
        shd, tpr, fpr, _, _ = count_accuracy(self.st_truth.adj.numpy(), st_learned.topk(self.n_edges).numpy())
        err = adjacency_error(self.st_truth.adj.numpy(), st_learned.adj.numpy())
        self.log.append(self.logger(shd, tpr, fpr, err))
    
    def logger(self, shd, tpr, fpr, err):
        return dict(zip(Validator.params, (shd, tpr, fpr, err)))
    
    def print_summary(self):
        _line = "     shd: {:.3f} || tpr: {:.3f} || fpr: {:.3f} || err: {:.3f}"
        summarize = lambda x: list(map(self.log.recall, x))
        
        print(_line.format(*summarize(Validator.params)))