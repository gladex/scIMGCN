import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np

class KANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True):
        super(KANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = 32
        self.outdim = 32

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        
        y = y.view(outshape)
        return y
    
class KanGNN(torch.nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat=32, grid_feat=200, num_layers=2, use_bias=False):
        super().__init__()
        self.num_layers = num_layers
        self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            self.lins.append(KANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        self.lins.append(nn.Linear(hidden_feat, out_feat, bias=False))
   
    def forward(self, x, adj):
        x = self.lin_in(x)
        for layer in self.lins[:self.num_layers-1]:
            #sparse_adj = adj.to_sparse()
            adj=adj.float()
            x = layer(torch.matmul(adj, x))
        x = self.lins[-1](x)
            
        return x
    
class GlobalAttentionMask(nn.Module):
    def __init__(self, feature_dim):
        super(GlobalAttentionMask, self).__init__()
        self.mask_weights = nn.Parameter(torch.randn(1, feature_dim))  

    def forward(self, x):
        mask = torch.sigmoid(self.mask_weights)
        mask = mask.expand(x.size(1), -1)
        return mask


def full_attention_conv(qs, ks, vs, kernel, output_attn=False):

    if kernel == 'simple':
        # normalize input
        qs = qs / torch.norm(qs, p=2)
        ks = ks / torch.norm(ks, p=2)
        N = qs.shape[0]
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)   
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)
    
        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)
        attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1) 


       
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum) 

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape)) 
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer 


        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer 

    elif kernel == 'sigmoid':
        # numerator
        attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", qs, ks)) 

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        attention_normalizer = torch.einsum("nlh,l->nh", attention_num, all_ones)
        attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, ks.shape[0], 1) 

        # compute attention and attentive aggregated results
        attention = attention_num / attention_normalizer
        attn_output = torch.einsum("nlh,lhd->nhd", attention, vs)  

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


def gcn_conv(value, adj):
    N, H = value.shape[0], value.shape[1]
    gcn_conv_output = []
    adj=adj.float()
    for i in range(value.shape[1]):
        gcn_conv_output.append(torch.matmul(adj, value[:, i]))
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1)
    return gcn_conv_output


class IMGraphConv(nn.Module):
    '''
    One DIFFormer layer with KanGNN replacing gcn_conv
    '''
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 num_heads,
                 kernel='simple',
                 grid_size=200,  
                 num_layers=2, 
                 use_graph=True,
                 use_weight=True):
        super(IMGraphConv, self).__init__()
        
        self.kernel = kernel
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.use_graph = use_graph


        self.kangnn = KanGNN(
            in_feat=in_channels,
            hidden_feat=hidden_channels,
            out_feat=out_channels,
            grid_feat=grid_size,
            num_layers=num_layers,
            use_bias=use_weight
        )

        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.kangnn.apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)

    def forward(self, query_input, source_input, adj, output_attn=False):

        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)


        attention_output = full_attention_conv(query, key, value, self.kernel)


        if self.use_graph:
            #final_output = attention_output + self.kangnn(value, adj) 
            final_output = attention_output + self.kangnn(query_input, adj) 
            
        else:
            final_output = attention_output

 
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, None 
        else:
            return final_output



class IMGCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, kernel='simple',
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, grid_size=200):
        super(IMGCN, self).__init__()
        self.global_attention_mask = GlobalAttentionMask(in_channels)
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))

        for i in range(num_layers):
            self.convs.append(
                IMGraphConv(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,  
                    num_heads=num_heads,
                    kernel=kernel,
                    use_graph=use_graph,
                    use_weight=use_weight
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, adj):
        layer_ = []

        mask = self.global_attention_mask(x)
        #x = mask.T * x 

        # input MLP layer
        x = self.fcs[0](x.T)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs): 

            x = conv(x, x, adj) 
            if self.residual: 
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn: 
                x = self.bns[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training) 
            layer_.append(x) 

        # output MLP layer
        x_out = self.fcs[-1](x)
        return x_out, mask


    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0) 