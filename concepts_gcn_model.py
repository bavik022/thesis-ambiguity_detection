import torch
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import math
import os
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

num_docs = 100
num_clusters = 20

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device", device)

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

embed_size = num_docs
batch_size = 10
epochs =  95
max_len = 256
epsilon = 0.0001

print("Initializing the model...")

class GCN(torch.nn.Module):
    def __init__(self, n_h, input_features):
        super().__init__()
        W = torch.FloatTensor(input_features, n_h)
        self.weights = torch.nn.Parameter(W)
        self.relu = torch.nn.ReLU()
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        bs = torch.FloatTensor(n_h)
        self.bias = torch.nn.Parameter(bs)
        self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, nfs, A):
        out = torch.matmul(nfs, self.weights)
        out = torch.matmul(A, out)
        out = out + self.bias
        out = self.relu(out)
        return out

class AmbiguityNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCN(20, embed_size)
        self.gcn2 = GCN(10, 20)
        self.gcn3 = GCN(10,10)
        self.dp = torch.nn.Dropout(p = 0.5)
        self.dense1 = torch.nn.Linear(in_features = num_clusters * 10, out_features = 64)
        self.dense2 = torch.nn.Linear(in_features = 64, out_features = 4)
    
    def forward(self, H, A):
        out = self.gcn1(H, A)
        out = self.dp(out)
        out = self.gcn2(out, A)
        out = self.dp(out)
        out = self.gcn3(out, A)
        out = self.dp(out)
        out = torch.flatten(out, 1)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

f_adj_mat = open('concept_adjacency_matrices.pkl', 'rb')
f_node_feat = open('concept_node_features.pkl', 'rb')

f_dev_adj_mat = open('dev_concept_adjacency_matrices.pkl', 'rb')
f_dev_node_feat = open('dev_concept_node_features.pkl', 'rb')

adj_matrices = pickle.load(f_adj_mat)
node_features = pickle.load(f_node_feat)

dev_adj_matrices = pickle.load(f_dev_adj_mat)
dev_node_features = pickle.load(f_dev_node_feat)

tids = list(adj_matrices.keys())
dev_tids = list(dev_adj_matrices.keys())
print(tids)
print(dev_tids)

f = open('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/doclists.pkl', 'rb')
label_list = {}
while True:
    try:
        data = pickle.load(f)
        tid = int(data['topic_id'])
        label_list[tid] = data['label']
    except EOFError:
        break
f.close()

f_dev = open('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/doclists_dev.pkl', 'rb')
label_list_dev = {}
while True:
    try:
        data = pickle.load(f_dev)
        tid = int(data['topic_id'])
        label_list_dev[tid] = data['label']
    except EOFError:
        break
f_dev.close()

labels = torch.LongTensor(torch.zeros(len(tids)).long())
adjms = torch.FloatTensor(len(tids), num_clusters, num_clusters)
nfts = torch.FloatTensor(len(tids), num_clusters, num_docs)

labels_dev = torch.LongTensor(torch.zeros(len(dev_tids)).long())
adjms_dev = torch.FloatTensor(len(dev_tids), num_clusters, num_clusters)
nfts_dev = torch.FloatTensor(len(dev_tids), num_clusters, num_docs)

for i in tqdm(range(len(tids))):
    tid = tids[i]
    A = torch.Tensor(adj_matrices[tid])
    S = torch.sum(A, axis = 1)
    S = S + epsilon
    S = torch.pow(S, -0.5)
    D_ = torch.diag(S)
    A_ = torch.chain_matmul(D_, A, D_)
    A_ = A_ + torch.eye(A_.shape[0])
    adjms[i] = A_
    nfts[i] = torch.FloatTensor(node_features[tid])
    labels[i] = float(label_list[int(tid)])
    
for i in tqdm(range(len(dev_tids))):
    tid = dev_tids[i]
    A = torch.Tensor(dev_adj_matrices[tid])
    S = torch.sum(A, axis = 1)
    S = S + epsilon
    S = torch.pow(S, -0.5)
    D_ = torch.diag(S)
    A_ = torch.chain_matmul(D_, A, D_)
    A_ = A_ + torch.eye(A_.shape[0])
    adjms_dev[i] = A_
    nfts_dev[i] = torch.FloatTensor(dev_node_features[tid])
    labels_dev[i] = float(label_list_dev[int(tid)])
    
    
def train(epochs, batch_size):
    model = AmbiguityNetwork()
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(nfts, adjms, labels)
    dev_dataset = torch.utils.data.TensorDataset(nfts_dev, adjms_dev, labels_dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    loader = torch.utils.data.DataLoader(dataset, sampler = torch.utils.data.RandomSampler(dataset), batch_size = batch_size, pin_memory = True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, sampler = torch.utils.data.RandomSampler(dev_dataset), batch_size = dev_dataset.__len__(), pin_memory = True)
    for epoch in tqdm(range(epochs)):
        model.train()
        for idx, data in enumerate(loader):
            nf, A, tgts = data
            tgts = tgts - 1
            tgts = torch.LongTensor(tgts)      
            preds = model(nf, A)
            loss = loss_fn(preds, tgts)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, loss: {loss}")
        
        with torch.no_grad():
            for idx, data in enumerate(dev_loader):
                nf, A, tgts = data
                tgts = tgts - 1
                val_preds = model(nf, A)
                m = torch.nn.Softmax()
                val_npreds = m(val_preds).numpy()
                class_preds = np.argmax(val_npreds, axis = 1)
                print(class_preds+1)
                val_acc = accuracy_score(tgts, class_preds)
                val_loss = log_loss(tgts, val_npreds, labels = [0,1,2,3])
                val_prec = precision_score(tgts, class_preds, average = 'weighted')
                val_rec = recall_score(tgts, class_preds, average = 'weighted')
                val_f1 = f1_score(tgts, class_preds, average = 'weighted')
            print(f"Validation_loss: {val_loss}, validation acc: {val_acc}, Precision: {val_prec}, Recall: {val_rec}, F1 score: {val_f1}")
    torch.save(model.state_dict(), 'concept_model.pt')

train(epochs, batch_size)
