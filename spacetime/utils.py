import torch
import torch.nn.functional as F
import numpy as np

            
def torch_pad(t, x, y):
    return F.pad(t, pad=(x, y), mode='constant', value=0)

def rand_norm(mu, std, x, y):
    return torch.normal(mu, torch.Tensor([std]*x*y).view(x, y))
                        
def row_norm(t):
    return t/torch.sum(t, dim=-1, keepdims=True)

def ortho_norm(std, x, y):
#     assert y>=2*x
    norm_1 = rand_norm(0.0, std, x, y)
    norm_2 = rand_norm(0.0, std, x, y)
#     full_norm = torch.eye(2*x)
#     norm_1 = row_norm(torch_pad(full_norm[:x], 0, y-2*x)+rand_norm(0.0, std, x, y))
#     norm_2 = -row_norm(torch_pad(full_norm[x:], 0, y-2*x)+rand_norm(0.0, std, x, y))
    return norm_1, norm_2

def check_nan(var_dict):
    for k, v in var_dict.items():
        if torch.sum(v!=v):
            raise ValueError('nan error on %s'%k)

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
    
    def plot(self, var, ax):
        data = self.log[var]
        ax.set_title(r'%s'%var)
        
        if type(data[0])==list:
            for d in data:
                ax.plot(range(len(d)), np.log10(d))
        else:
            ax.plot(range(len(data)), np.log10(data))