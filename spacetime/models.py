####################################################################
### Ripped heavily from https://github.com/fishmoon1234/DAG-GNN
####################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

_EPS = 1e-10

class MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, tol = 0.1):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def preprocess_adj(self, adj):
        adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
        return adj_normalized

    def forward(self, inputs):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = self.preprocess_adj(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa

        return x, logits, adj_A1, self.z, self.z_positive, self.adj_A, self.Wa

class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, n_hid):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def preprocess_adj_inv(self, adj):
        adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
        return adj_normalized

    def forward(self, inputs, input_z, n_in_node, origin_A, Wa):
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = self.preprocess_adj_inv(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out