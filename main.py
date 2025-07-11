from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
import h5py
import scanpy as sc
import pandas as pd
import time
import scipy as sp

from preprocess2 import read_dataset, normalize,normalize2
from train import train_model
from readdata import prepro




seed = 0
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 

if __name__ == "__main__":
    # Quake_10x_Bladder, Quake_Smart-seq2_Lung, Quake_10x_Limb_Muscle, Romanov, 
    # Muraro, Young, Quake_10x_Spleen, Adam, Quake_Smart-seq2_Trachea, Quake_Smart-seq2_Diaphragm
    Fileformat = ['data.h5','csv','dataset.h5']
    Method = ['pearson','spearman','NE']
    
    parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', type=str, default='Quake_10x_Bladder')
    parser.add_argument('--method', default=Method[2], type=str)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--format', type=str, default=Fileformat[2])
    parser.add_argument('--cpu', action='store_true')   
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1,
                            help='number of distinct runs')
    parser.add_argument('--metric', type=str, default='f1', choices=['acc', 'rocauc', 'f1'],
                            help='evaluation metric')


    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2,
                            help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=1,
                            help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_bn', default=True, help='use layernorm')
    parser.add_argument('--use_residual', default=True, help='use residual link for each GNN layer')
    parser.add_argument('--use_graph', default=True, help='use pos emb')
    parser.add_argument('--use_weight', default=True, help='use weight for GNN convolution')
    parser.add_argument('--kernel', type=str, default='simple', choices=['simple', 'sigmoid'])


    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.0) 
    parser.add_argument('--batch_size', type=int, default=10000, help='mini batch training for large graphs')

    parser.add_argument('--display_step', type=int,
                            default=1, help='how often to print')
    

    args = parser.parse_args([])
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    device = torch.device("cuda" if args.cuda else "cpu")
    start = time.time()
    if args.name=='Romanov': args.format = 'data.h5'
    else :args.format = 'dataset.h5'

    if args.format == 'data.h5':
            X,Y,var = prepro(args.name)
            X = np.ceil(X).astype(int)
            count_X = X

            adata = sc.AnnData(X,dtype='float32')
            adata.var_names = var.index
            adata.obs['Group'] = Y
    elif args.format == 'csv':
            adata=sc.read_csv('data/PBMC/pbmc.csv')
            y= pd.read_csv('data/PBMC/clusters.csv', index_col=0)
            y= np.array(y.values)
            adata.obs['Group'] = y
            adata.var_names_make_unique()
    elif args.format == 'dataset.h5':
            #data_mat = h5py.File('data/real_data/Quake_Smart-seq2_Lung.h5')
            data_mat = h5py.File('data/real_data/'+args.name+'.h5')
            #Quake_10x_Bladder，Quake_Smart-seq2_Lung，Quake_10x_Limb_Muscle，Muraro，Young，Quake_10x_Spleen，Adam，Quake_Smart-seq2_Trachea，Quake_Smart-seq2_Diaphragm
            x = np.array(data_mat['X'])
            y = np.array(data_mat['Y'], dtype=np.int64)
            data_mat.close()
            adata = sc.AnnData(x,dtype='float32')
            adata.obs['Group'] = y
    else:
            print(f'Unknown name: {args.name}')
    

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)
    adata = normalize2(adata,
                    size_factors=True,
                    normalize_input=True,
                    logtrans_input=True,
                    select_hvg=True)
    X = adata.X
    X=X.T
    X_raw = adata.raw.X.T
    sf = adata.obs.size_factors

    train_model(adata, X_raw, sf, args)
    time_used = time.time()-start
    print(time_used)

