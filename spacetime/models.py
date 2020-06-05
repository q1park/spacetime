import torch
import torch.nn as nn
import torch.nn.functional as F

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