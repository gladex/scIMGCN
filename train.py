import torch
import scanpy as sc
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
import torch.nn.functional as F
from get_graph import getGraph
from sklearn.metrics import accuracy_score, f1_score
from model import IMGCN
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
def target_distribution(q):
    # Pij
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()



def create_train_test_adj(adj, obs_train, obs_test):
    adj = adj.clone()
    adj[np.ix_(obs_train, obs_test)] = 0
    adj[np.ix_(obs_test, obs_train)] = 0   
    return adj


class load_data(Dataset):
    def __init__(self, dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
               torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx))


def adjacency_to_edge_index(adjacency_matrix):
    """
    Convert an adjacency matrix to edge_index and edge_weight format.
    """
    edge_indices = adjacency_matrix.nonzero(as_tuple=True)    
    edge_index = torch.stack(edge_indices, dim=0)
    edge_weight = adjacency_matrix[edge_indices]

    return edge_index, edge_weight



def train_model(dataset, X_raw, sf, args):
    global p
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    
    X = dataset.X.T
    _, n = X.shape
    data = torch.Tensor(X).to(device)
    y = dataset.obs['Group']
    y=torch.tensor(y.values).to(device)
    if y.min()== 0:
        c = y.max().item()+1
    else: 
        c = y.max().item() 

    L = 0
    N = dataset.shape[1]
    N = int(N)
    avg_N = N // c
    K = avg_N // 10
    K = min(K, 20)
    K = max(K, 6)
    method=args.method
    adj = getGraph(X, L, K, method).to(device)
    #adj = calculate_adjacency_matrix(X, method='pearson', threshold=0.5, soft_threshold=False)


    criterion = nn.CrossEntropyLoss()
    def F_MEL(mask):
        mask_clamped = torch.clamp(mask, min=1e-6)
        return -torch.sum(mask_clamped * torch.log(mask_clamped)) / mask.numel()
    def F_MSL(mask):
        return torch.sum(mask) / mask.numel()
    def L_IA(mask, alpha1, alpha2):
        mask = mask.mean(dim=0, keepdim=True)
        return alpha1 * F_MEL(mask) + alpha2 * F_MSL(mask)

    
        
    d = dataset.shape[1]
    model=IMGCN(d,args.hidden_channels, c, num_layers=args.num_layers, alpha=args.alpha, dropout=args.dropout, num_heads=args.num_heads, kernel=args.kernel,
                       use_bn=args.use_bn, use_residual=args.use_residual, use_graph=args.use_graph, use_weight=args.use_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float('-inf')
  
    model.train()
    num_columns = dataset.shape[0]
    observations = np.arange(num_columns)
    X_train, X_test, obs_train, obs_test = train_test_split(data.T, observations, test_size=0.3, random_state=42)
    adj = create_train_test_adj(adj, obs_train, obs_test)
    model.train()

    if args.metric == 'acc':
        eval_func = eval_acc_2
    else:
        eval_func = eval_f1_2
    
    

    for run in range(args.runs):

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out,mask = model(data, adj)
            out = F.softmax(out, dim=1)
            loss = 0.6*criterion(
                out[obs_train], y[obs_train]).to(device)+L_IA(mask,0.8,0.8)*0.4
            loss.backward()
            optimizer.step()
            train_acc, test_acc, train_f1, test_f1, _ = evaluate(
        model, data, adj, y, obs_train, obs_test,eval_func, criterion, args
    )
            if epoch % args.display_step == 0:
                print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")
                print(f"Train Accuracy: {100 * train_acc:.2f}%, Test Accuracy: {100 * test_acc:.2f}%")
                print(f"Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")


def eval_f1_2(y_true, y_pred):

    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1).detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1


def eval_acc_2(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1).detach().cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    return acc
def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[0]):
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc_list.append(f1)

    return sum(acc_list)/len(acc_list)

def evaluate(model, data, adj, y,obs_train,obs_test, eval_func, criterion, args, result=None):
    model.eval()
    out,_ = model(data, adj)

    train_acc = eval_acc_2(
        y[obs_train], out[obs_train])
    test_acc = eval_acc_2(
        y[obs_test], out[obs_test])

    train_f1 = eval_f1_2(y[obs_train], out[obs_train])
    test_f1 = eval_f1_2(y[obs_test], out[obs_test])

    return train_acc,  test_acc , train_f1, test_f1,out

