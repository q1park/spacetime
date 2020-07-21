import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#===================================
# misc helpers
#===================================

class ModelStore:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
            
class Parameters:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        
def check_nan(var_dict):
    for k, v in var_dict.items():
        if torch.sum(v!=v):
            raise ValueError('nan error on %s'%k)
            
#===================================
# initialization helpers
#===================================
            
def torch_pad(t, x, y):
    return F.pad(t, pad=(x, y), mode='constant', value=0)

def rand_norm(mu, std, x, y):
    return torch.normal(mu, torch.Tensor([std]*x*y).view(x, y))
                        
def row_norm(t):
    return t/torch.sum(t, dim=-1, keepdims=True)

def ortho_norm(mu, std, x, y):
#     assert y>=2*x
    norm_1 = rand_norm(mu, std, x, y)
    norm_2 = rand_norm(mu, std, x, y)
#     full_norm = torch.eye(2*x)
#     norm_1 = row_norm(torch_pad(full_norm[:x], 0, y-2*x)+rand_norm(0.0, std, x, y))
#     norm_2 = -row_norm(torch_pad(full_norm[x:], 0, y-2*x)+rand_norm(0.0, std, x, y))
    return norm_1, norm_2

#===================================
# training log plots
#===================================

def plot_trials(trainer, var_list, logger_type=None, skip_first=0, skip_last=0):
    fig, axs = plt.subplots(1, len(var_list), figsize=(3*len(var_list), 2))
    for var, ax in zip(var_list, axs):
        trainer.plot_loggers(logger_type, var, ax, skip_first, skip_last)
    plt.show()
    
def plot_graphs(trainer):
    n_models = len(trainer.models)
    fig, axs = plt.subplots(1, n_models, figsize=(3*n_models, 3))
    if n_models > 1:
        for model, ax in zip(trainer.models, axs):
            model.model.mask.plot_temp(ax, vmin=-1.0, vmax=1.0, norm=True)
    else:
        trainer.models[0].model.mask.plot_temp(axs, vmin=-1.0, vmax=1.0, norm=True)
    plt.show()
    
#===================================
# pdf plots
#===================================

def plot_conditionals(col_plot_dict, sampler_list, label_list, smooth=False, normref=True):
    assert len(sampler_list)==len(label_list)
    fig, axs = plt.subplots(len(sampler_list), len(col_plot_dict), sharex='col', sharey='col', 
                            figsize = (4*len(col_plot_dict), 3*len(sampler_list)))
    
    for col, (d,c) in col_plot_dict.items():
        for i, sampler in enumerate(sampler_list):
            if len(sampler_list)>1:
                axs[i][col].set_title(r'$P(%s|%s)$    (%s)'%(d,c,label_list[i]))
                axs[i][col].contour(*sampler.get_contour_conditional(d,c, smooth=smooth, normref=normref), levels=50)
            else:
                axs[col].set_title(r'$P(%s|%s)$    (%s)'%(d,c,label_list[i]))
                axs[col].contour(*sampler.get_contour_conditional(d,c, smooth=smooth, normref=normref), levels=50)
    pass;

def plot_joints(col_plot_dict, sampler_list, label_list, smooth=False):
    assert len(sampler_list)==len(label_list)
    fig, axs = plt.subplots(len(sampler_list), len(col_plot_dict), sharex='col', sharey='col', 
                            figsize = (4*len(col_plot_dict), 3*len(sampler_list)))
    
    for col, (d,c) in col_plot_dict.items():
        for i, sampler in enumerate(sampler_list):
            if len(sampler_list)>1:
                axs[i][col].set_title(r'$P(%s,%s)$    (%s)'%(d,c,label_list[i]))
                axs[i][col].contour(*sampler.get_contour_joint(d,c, smooth=smooth), levels=50)
            else:
                axs[col].set_title(r'$P(%s,%s)$    (%s)'%(d,c,label_list[i]))
                axs[col].contour(*sampler.get_contour_joint(d,c, smooth=smooth), levels=50)
    pass;

def plot_marginals(sampler_list):
    fig, axs = plt.subplots(1,len(sampler_list), figsize = (4*len(sampler_list),3))
    
    for i, (sampler, ax) in enumerate(zip(sampler_list, axs)):
        ax.set_title(r'P(node)')
        nodes = sampler.node_data.nodes
        for node in nodes:
            if i==0:
                ax.plot(*sampler.get_plot_marginal(node), label=node)
            else:
                ax.plot(*sampler.get_plot_marginal(node))
    fig.legend()
    pass;

#===================================
# text format
#===================================

def print_format(loggers, around=3, log10=False):
    _stats = logger_stats(loggers).items()
    
    _vals = [x[1] for x in _stats]
    _fmt = "{:.%sf}"%str(around)
    
    if log10:
        _vals = [1e-15+v for v in _vals]
        _vals = list(map(np.log10, _vals))
        _fmt = "10^%s"%_fmt
        
    _str = ["%s: %s"%(k, _fmt) for k in [x[0] for x in _stats]]
    return ' || '.join(_str).format(*_vals)

#===================================
# training logger
#===================================

def logger_stats(loggers):
    stats = dict()
    for logger in loggers:
        for k in logger.log.keys():
            stats[k] = logger.recall(k)
    return stats

class Logger:
    def __init__(self, *args):
        self.log_vars = args
        self.log = {k:list() for k in self.log_vars}
        
        self._mean = lambda x: np.mean(x[:])
        self._last = lambda x: x[:][-1]
        self._full = lambda x: x[:]
    
    def clear(self):
        for k in self.log:
            self.log[k].clear()
            
    def append(self, var_dict):
        for log_var, var in zip(self.log, var_dict):
            self.log[log_var] += [var_dict[var]]

    def mean(self):
        return {k:self._mean(v) for k,v in self.log.items()}
        
    def last(self):
        return {k:self._last(v) for k,v in self.log.items()}
        
    def full(self):
        return {k:self._full(v) for k,v in self.log.items()}
    
    def recall(self, var):
        return self.log[var][:][-1]
    
    def plot(self, var, ax, skip_first=0, skip_last=0, log10=False):
        data = self.log[var]
        ax.set_title(r'%s'%var)
        
        if log10:
            clip = lambda x: [np.log10(1e-15+x) for x in x[skip_first:-skip_last-1]]
        else:
            clip = lambda x: x[skip_first:-skip_last-1]
            
        if type(data[0])==list:
            for d in data:
                data_clip = clip(d)
                ax.plot(range(len(data_clip)), data_clip)
        else:
            data_clip = clip(data)
            ax.plot(range(len(data_clip)), data_clip)
            