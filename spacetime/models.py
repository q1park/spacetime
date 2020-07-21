import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from spacetime.spacetime import SpaceTime
from spacetime.utils import rand_norm, ortho_norm

def clones(module, n_layers):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])

#===================================
# neural net modules
#===================================

class LinearBlock(nn.Module):
    """Linear module."""
    def __init__(self, n_in, n_hid, n_out, dropout, gain):
        super(LinearBlock, self).__init__()
        self.gain = gain
        self.w_1 = nn.Linear(n_in, n_hid)
        self.w_2 = nn.Linear(n_hid, n_out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=self.gain)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
#===================================
# causal modules
#===================================

class GraphMatrix(nn.Module):
    def __init__(self, n_nodes):
        super(GraphMatrix, self).__init__()
        self.A = nn.Parameter(rand_norm(0.0, 1e-7, n_nodes, n_nodes), requires_grad=True)
        
    def mat(self):
        return self.A
    
    def mat_boost(self):
        return torch.sinh(3*self.A)
    
class GraphVector(nn.Module):
    def __init__(self, n_nodes):
        super(GraphVector, self).__init__()
        norm_1, norm_2 = ortho_norm(0.0, 1e-7, n_nodes, 2*n_nodes)
        self.Q = nn.Parameter(norm_1, requires_grad=True)
        self.K = nn.Parameter(norm_2, requires_grad=True)
        
    def mat(self):
        return torch.matmul(self.Q, self.K.transpose(-2,-1))
    
    def mat_boost(self):
        return torch.sinh(3*torch.matmul(self.Q, self.K.transpose(-2,-1)))
    
    def vectors(self):
        return self.Q.detach(), self.K.detach()

#===================================
# attention modules
#===================================

class AttentionBlock(nn.Module):
    def __init__(self, spacetime, n_heads, d_embed, dropout=0.1, mask_self=False):
        super(AttentionBlock, self).__init__()
        assert d_embed%n_heads==0
        self.mask = spacetime
        self.d_k = d_embed//n_heads
        self.h = n_heads
        
        self.linears = clones(nn.Linear(d_embed, d_embed), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None
        self._init_weights()
        
    def _init_weights(self):
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def attention(self, q, k, v, dropout=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2,-1))/np.sqrt(d_k)
        p_attn = scores*self.mask.A.transpose(-2,-1)
    
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn
        
    def forward(self, q, k, v):
        nbatches = q.size(0)
        q, k, v = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
                   for l, x in zip(self.linears, (q, k, v))]
        x, self.attn = self.attention(q, k, v, dropout=self.dropout) 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears[-1](x)
    
class EmbedBlock(nn.Module):
    def __init__(self, d_in, d_out):
        super(EmbedBlock, self).__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.proj(x)
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_embed, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_embed)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) *
                             -(np.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
#===================================
# DAG models
#===================================

class DAGGNN(nn.Module):
    def __init__(self, n_nodes, graph_params):
        super(DAGGNN, self).__init__()
        self.encoder = LinearBlock(**graph_params.model).double()
        self.decoder = LinearBlock(**graph_params.model).double()

        self.causal = GraphMatrix(n_nodes).double()
        self.mask = SpaceTime(torch.zeros((n_nodes, n_nodes)))
    
    def update_mask(self):
        self.mask.A = self.causal.mat().detach()

    def sem(self):
        return torch.eye(self.mask.A.shape[-1]).double()-self.causal.mat_boost().transpose(-2,-1)

    def forward(self, x):
        z = self.encoder(x)
        z = torch.einsum('jl,ilk->ijk', self.sem(), z)
        xx = torch.einsum('jl,ilk->ijk', self.sem().inverse(), z)
        xx = self.decoder(xx)
        return xx, z, self.causal.mat()
    
class DAGGAE(nn.Module):
    def __init__(self, n_nodes, graph_params):
        super(DAGGAE, self).__init__()
        self.encoder = LinearBlock(**graph_params.model).double()
        self.decoder = LinearBlock(**graph_params.model).double()

        self.causal = GraphMatrix(n_nodes).double()
        self.mask = SpaceTime(torch.zeros((n_nodes, n_nodes)))
    
    def update_mask(self):
        self.mask.A = self.causal.mat().detach()
    
    def op(self):
        op = self.causal.mat().transpose(-2,-1)
        with torch.no_grad():
            op.fill_diagonal_(0.0)
        return op
    
    def forward(self, x):
        z = self.encoder(x)
        xx = torch.einsum('jl,ilk->ijk', self.op(), z)
        xx = self.decoder(xx)
        return xx, z, self.causal.mat()
    
#===================================
# SEM generator
#===================================

class SEMCAE(nn.Module):
    def __init__(self, mask, gen_params, latent):
        super(SEMCAE, self).__init__()
        self.n_nodes = gen_params.n_nodes
        self.I = torch.eye(gen_params.n_nodes).double()
        
        self.encoder = LinearBlock(**gen_params.coders).double()
        self.decoder = LinearBlock(**gen_params.coders).double()
        self.samplers = clones(LinearBlock(**gen_params.samplers).double(), gen_params.n_nodes)
        
        self.mask = mask
        self.latent = latent
        
    def W(self, *mutil):
        if len(mutil)>0:
            return self.mask.mutilate(*mutil).transpose(-2,-1).double()
        else:
            return self.mask.A.transpose(-2,-1).double()
    
    def delta(self, *mutil):
        return self.I*self.n_nodes**(-2*self.n_nodes*torch.tanh(2*self.n_nodes*self.W(*mutil).sum(dim=-1)))
    
    def aux_dim(self, x):
        return (self.I.unsqueeze(0)*x)
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def causal(self, z, *mutil):
        noise = torch.normal(0.0, 0.1*z.std(dim=0).expand(z.shape[0], z.shape[1], 1))
        
        z = self.aux_dim(z)
        z = torch.einsum('jm,imk->ijk', self.W(*mutil)+self.delta(*mutil), z)
        
        z_params = torch.cat([self.samplers[i](z[:,i:i+1,:]) 
                              if i not in self.latent+list(mutil)
                              else torch.cat([z[:,i:i+1,i:i+1], self.samplers[i](z[:,i:i+1,:])[:,:,1:]], dim=2)
                              for i in range(self.n_nodes)], dim=1)
        
        z, logvar_z = z_params[:,:,0:1], z_params[:,:,1:2]
        
        std = torch.exp(0.5*logvar_z)
        eps = torch.randn_like(std)
        return z + eps*std # + noise