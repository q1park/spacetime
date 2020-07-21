import time
import torch
import numpy as np

from spacetime.spacetime import SpaceTime
from spacetime.losses import LagrangianLoss, ELBOLoss, InfoLoss
from spacetime.optimizers import OptimGraph, OptimGen
from spacetime.metrics import count_accuracy, adjacency_error
from spacetime.utils import ModelStore, check_nan, print_format, Logger

#===================================
# DAG validator
#===================================

class Validator:
    params = ['shd', 'tpr', 'fpr', 'err']
    
    def __init__(self, st_truth):
        self.log = Logger(*Validator.params)
        self.st_truth = st_truth
        self.n_nodes = self.st_truth.A.shape[-1]
        self.n_edges = len(self.st_truth.A.nonzero())
        self.l1 = torch.sum(torch.abs(self.st_truth.A)).item()

    def validate(self, st_learned):
        shd, tpr, fpr, _, _ = count_accuracy(self.st_truth.A.numpy(), 
                                             st_learned.topk(self.n_edges).numpy())
        err = adjacency_error(self.st_truth.A.numpy(), 
                              st_learned.A.numpy())
        self.log.append(self.logger(shd, tpr, fpr, err))
    
    def logger(self, shd, tpr, fpr, err):
        return dict(zip(Validator.params, (shd, tpr, fpr, err)))
    
    def print_summary(self):
        print(3*" "+print_format([self.log]))
        
    def get_logger(self, var):
        if var in Validator.params:
            return self.log
        else:
            raise ValueError("invalid parameter %s"%var)

#===================================
# DAG trainer
#===================================            

class DAGTrainer:
    def __init__(self, n_nodes, data, graph_params, true_spacetime=None):
        self.models = list()
        self.n_nodes = n_nodes
        self.data = data
        self.p_graph = graph_params
        self.true_spacetime = true_spacetime
        
    def train(self, model, print_every=2):
        t_total = time.time()
        loss = LagrangianLoss.poly(self.n_nodes, self.p_graph.coeffs, self.p_graph.ctrls, 
                                   opt=OptimGraph(model, **self.p_graph.opts))
        val = None if self.true_spacetime is None else Validator(self.true_spacetime) 
        
        model.train()
        while not loss.opt.quit:
            for epoch in range(self.p_graph.train['epochs']):
                self.train_epoch(model, loss)
                
                if val is not None:
                    val.validate(SpaceTime(loss.interaction.adj))
                    
            if loss.opt._iter%print_every==0:
                loss.print_summary(loss)
                
                if val is not None:
                    val.print_summary()
            loss.iterate()
        
        print("\nTrial %s finished in %s seconds"%(len(self.models), time.time()-t_total))
        print("-"*75)
        model.update_mask()
        model.eval()
        self.models.append(ModelStore(model=model, graph_loss=loss, graph_eval=val))
        
    def train_epoch(self, model, loss):
        for x in self.data.loader(batch_size=self.p_graph.train['batch_size']):
            xx, z, A = model(x.double())
            check_nan({'preds':xx})
            loss(x, z, xx, A)
        loss.evolve()
        
    def plot_loggers(self, logger_type, var, ax, skip_first=0, skip_last=0):
        for model in self.models:
            if logger_type!='eval':
                model.graph_loss.get_logger(var).plot(var, ax, skip_first, skip_last, log10=True)
            else:
                model.graph_eval.get_logger(var).plot(var, ax, skip_first, skip_last, log10=False)

#===================================
# SEM trainer
#===================================  

class SEMTrainer:
    def __init__(self, n_nodes, data, gen_params, chain):
        self.models = list()
        self.n_nodes = n_nodes
        self.data = data
        self.p_gen = gen_params
        self.chain = chain
        
    def train(self, model, print_every=10):
        loss = InfoLoss(model.mask, self.chain, opt=OptimGen(model, **self.p_gen.opts))
        model.train()
        t_total = time.time()

        for epoch in range(self.p_gen.train['epochs']):
            self.train_epoch(self.data, model, loss)
            if loss.opt._epoch%print_every==0:
                print(3*"*"+"Epoch: %s/%s"%(loss.opt._epoch, self.p_gen.train['epochs']))
                loss.print_summary()

        print("\nModel finished in %s seconds"%(time.time()-t_total))
        print("-"*75)
        model.eval()
        self.models.append(ModelStore(model=model, gen_loss=loss))

    def train_epoch(self, data, model, loss):
        for x in data.loader(batch_size=self.p_gen.train['batch_size'], shuffle=True):
            z = model.encode(x)
            zz = model.causal(z, self.chain[-1])
            xx = model.decode(z)

            check_nan({'preds':xx})
            loss(x, z, zz, xx)
        loss.evolve()
        
    def plot_loggers(self, logger_type, var, ax, skip_first=0, skip_last=0):
        for model in self.models:
            model.gen_loss.get_logger(var).plot(var, ax, skip_first, skip_last, log10=True)