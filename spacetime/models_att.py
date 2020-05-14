####################################################################
### Ripped heavily from https://github.com/fishmoon1234/DAG-GNN
####################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

_EPS = 1e-10

class AutoEncoder(nn.Module):
    def __init__(self, mod_params):
        super(AutoEncoder, self).__init__()
        self.embedder = LinearBlock(mod_params.x_dims, mod_params.h_dims, mod_params.z_dims).double()
        self.debedder = LinearBlock(mod_params.z_dims, mod_params.h_dims, mod_params.x_dims).double()
        self.semblock = SEMBlock(mod_params.n_nodes, mod_params.z_dims).double()
        self.ennoiser = LinearBlock(mod_params.z_dims, int(mod_params.h_dims), mod_params.z_dims).double()
        self.denoiser = LinearBlock(mod_params.z_dims, int(mod_params.h_dims), mod_params.z_dims).double()
    
    def encode(self, x):
#         x = self.embedder(x)
        x, adj, Wa = self.semblock.forward(x)
        z = self.ennoiser(x)
        return z, adj, Wa
    
    def decode(self, z, adj, Wa):
        x = self.denoiser(z)
        x = self.semblock.inverse(x, adj, Wa)
#         x = self.debedder(x)
        return x
    
    def forward(self, x):
        z, adj, Wa = self.encode(x)
        out = self.decode(z, adj, Wa)
        return out, z, adj, Wa

class LinearBlock(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.0):
        super(LinearBlock, self).__init__()
        self.w_1 = nn.Linear(n_in, n_hid)
        self.w_2 = nn.Linear(n_hid, n_out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class SEMBlock(nn.Module):
    """SEM operator module."""
    def __init__(self, num_nodes, n_out):
        super(SEMBlock, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.zeros((num_nodes, num_nodes)).double(), requires_grad=True))
        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)

    def _check_nan(self):
        if torch.sum(self.adj_A != self.adj_A):
            raise ValueError('nan error \n')
            
    def op(self, adj):
        return torch.eye(adj.shape[0]).double() - (adj.transpose(0,1))
    
    def bmm(self, adj, x):
        return torch.einsum('ij,ajc->aic', adj, x)

    def forward(self, x):
        self._check_nan
        adj_A1 = torch.sinh(3*self.adj_A) # amplifying A accelerates convergence
        z = self.bmm(self.op(adj_A1), x+self.Wa)-self.Wa
        return z, adj_A1, self.Wa
    
    def inverse(self, z, adj, Wa):
        self._check_nan
        x = self.bmm(torch.inverse(self.op(adj)), z+Wa)-Wa
        return x