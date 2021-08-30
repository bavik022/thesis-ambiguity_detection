import torch
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['CUDA_VISIBLE_DEVICES'] = '6, 5'

world_size = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print("Device", device)

bert_model = SentenceTransformer("/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/msmarco-MiniLM-L-6-v3/")
if torch.cuda.is_available():
    bert_model.cuda()

print("Initializing initial query encoder...")
bert_model.max_seq_length = 1000
torch.manual_seed(10)
torch.autograd.set_detect_anomaly(True)

embed_size = 256
init_query_embed_size = 384
num_docs = 20
batch_size = 10
epochs = 20
max_len = 256
epsilon = 0.0001

tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', truncation = True)

print("Initializing the model...")
class SBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained('google/electra-small-discriminator').train()
        self.pooler = torch.nn.AvgPool1d(embed_size)
    def forward(self, ids, attention_masks, token_type_ids):
        embeds = [self.roberta(id, attention_mask, token_type_id).last_hidden_state for (id, attention_mask, token_type_id) in zip(ids, attention_masks, token_type_ids)]
        embeds = [self.pooler(embed) for embed in embeds]
        embeds = [torch.squeeze(embed, dim = 2) for embed in embeds]
        ln = len(embeds)
        embeds = torch.cat(embeds, dim = 0).reshape((ln, num_docs, embed_size))      
        return embeds

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
        self.sbert = SBERT()
        self.gcn1 = GCN(20, embed_size)
        self.gcn2 = GCN(10, 20)
        self.gcn3 = GCN(10,10)
        self.dp = torch.nn.Dropout(p = 0.5)
        self.dense1 = torch.nn.Linear(in_features = num_docs*10 + init_query_embed_size, out_features = 64)
        self.dense2 = torch.nn.Linear(in_features = 64, out_features = 4)
        self.relu = torch.nn.ReLU()
    
    def forward(self, ids, attention_masks, token_type_ids, init_query):
        embeds = self.sbert(ids, attention_masks, token_type_ids)
        emb_norm = torch.nn.functional.normalize(embeds, dim = 2)
        emb_sum = torch.sum(torch.square(emb_norm), axis = 2)
        csml = torch.matmul(emb_norm, emb_norm.transpose(1,2))
        A = csml
        S = torch.sum(A, axis = 2)
        S = S + epsilon
        S = torch.pow(S, -0.5)
        #print(S)
        #D = torch.diag_embed(torch.sum(A, axis = 2))
        #D_inv = torch.linalg.inv(D) 
        D_ = torch.diag_embed(S)    
        #print("D_", D_)
        #D_ = torch.pow(D_inv, 0.5)
        A_ = torch.matmul(A, torch.transpose(D_, 1,2))
        A_ = torch.matmul(D_, A_)
        A = A_ + torch.eye(embeds.shape[1]).to(device)
        out = self.gcn1(embeds, A)
        out = self.dp(out)
        out = self.gcn2(out, A)
        out = self.dp(out)
        out = self.gcn3(out, A)
        out = self.dp(out)
        out = torch.flatten(out, 1)
        out = torch.cat((out, init_query.to(device)), dim = 1)
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dense2(out)
        return out
        
class InputDataset(torch.utils.data.Dataset):
    def __init__(self, ids, masks, token_type_ids, labels, init_qrs):
        self.ids = ids
        self.masks = masks
        self.token_type_ids = token_type_ids
        self.labels = torch.LongTensor(labels.long())
        self.init_qrs = init_qrs
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.masks[idx], self.token_type_ids[idx], self.labels[idx], self.init_qrs[idx]    
    
def prep_data():
    print("Preparing the data...")
    
    requests = pd.read_table('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/data/train.tsv', sep = '\t', header = 0).drop_duplicates('topic_id')
    rdev = pd.read_table('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/data/dev.tsv', sep = '\t', header = 0).drop_duplicates('topic_id')
    
    df = requests.append(rdev)
    
    dev_tids = rdev['topic_id'].to_numpy(dtype = int)
    
    f = open('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/doclists_dev_full_docs.pkl', 'rb')
    docs = {}
    dev_queries = {}
    dev_query_list = []
    dev_label_list = {}
    while True:
        try:
            data = pickle.load(f)
            tid = int(data['topic_id'])
            docs[tid] = data['docs']
            dev_queries[tid] = data['queries']
            dev_label_list[tid] = data['label']
        except EOFError:
            break
    f.close()
    print(dev_tids)
    dev_inputs = []
    for tid in dev_tids:
        dev_inputs.append(docs[tid][:num_docs])
    
    dev_labels = torch.LongTensor(torch.zeros(len(dev_tids)).long())
    dev_init_qrs = torch.zeros((len(dev_tids), init_query_embed_size))
    dev_ids = torch.LongTensor(len(dev_tids), num_docs, max_len)
    dev_masks = torch.LongTensor(len(dev_tids), num_docs, max_len)
    dev_token_type_ids = torch.LongTensor(len(dev_tids), num_docs, max_len)
    
    print("Preparing dev data...")
    
    t = 0
    
    for i in tqdm(range(len(dev_tids))):
        tid = dev_tids[i]
        documents = dev_inputs[i]
        tokens = [tokenizer.encode_plus(doc, None, add_special_tokens = True, max_length = max_len, padding = 'max_length', return_token_type_ids = True, truncation = True) for doc in documents]
        idlist = [torch.tensor(token['input_ids'], dtype = torch.long) for token in tokens]
        torch.cat(idlist, out = dev_ids[i])
        masklist = [torch.tensor(token['attention_mask'], dtype = torch.long) for token in tokens]
        torch.cat(masklist, out = dev_masks[i])
        tokentypelist = [torch.tensor(token['token_type_ids'], dtype = torch.long) for token in tokens]
        torch.cat(tokentypelist, out = dev_token_type_ids[i])
        dev_labels[i] = float(dev_label_list[tid])
        init_qr = dev_queries[tid]
        dev_init_qrs[i] = torch.Tensor(bert_model.encode(init_qr))
    
    
    f = open('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/doclists_full_docs.pkl', 'rb')
    docs = {}
    queries = {}
    query_list = []
    label_list = {}
    while True:
        try:
            data = pickle.load(f)
            tid = int(data['topic_id'])
            docs[tid] = data['docs']
            queries[tid] = data['queries']
            label_list[tid] = data['label']
        except EOFError:
            break
    f.close()
    
    ks = list(docs.keys())
    tids = requests['topic_id'].to_numpy(dtype = int)
    tids = np.intersect1d(tids, np.array(ks))
    
    inputs = []
    for tid in tids:
        for i in range(len(queries[tid])):
            inputs.append(docs[tid][:num_docs])
    
    for tid in tids:
        query_list.extend(queries[tid])
    
    print(len(inputs[19]))
    print(len(queries[1]))
    
    
    labels = torch.LongTensor(torch.zeros(len(query_list)).long())
    init_qrs = torch.zeros((len(query_list), init_query_embed_size))
    ids = torch.LongTensor(len(query_list), num_docs, max_len)
    masks = torch.LongTensor(len(query_list), num_docs, max_len)
    token_type_ids = torch.LongTensor(len(query_list), num_docs, max_len)
    
    print("Preparing training data...")
    
    t = 0
    
    for i in tqdm(range(len(tids))):
        tid = tids[i]
        lq = len(queries[tid])
        documents = inputs[t : t + lq]
        for j in range(lq):
            tokens = [tokenizer.encode_plus(doc, None, add_special_tokens = True, max_length = max_len, padding = 'max_length', return_token_type_ids = True, truncation = True) for doc in documents[j]]
            idlist = [torch.tensor(token['input_ids'], dtype = torch.long) for token in tokens]
            torch.cat(idlist, out = ids[t + j])
            masklist = [torch.tensor(token['attention_mask'], dtype = torch.long) for token in tokens]
            torch.cat(masklist, out = masks[t + j])
            tokentypelist = [torch.tensor(token['token_type_ids'], dtype = torch.long) for token in tokens]
            torch.cat(tokentypelist, out = token_type_ids[t + j])
            labels[t+j] = float(label_list[tid])
            init_qr = queries[tid][j]
            init_qrs[t+j] = torch.Tensor(bert_model.encode(init_qr))
        t = t+lq
    
    
    train_dataset = InputDataset(ids, masks, token_type_ids, labels, init_qrs)
    dev_dataset = InputDataset(dev_ids, dev_masks, dev_token_type_ids, dev_labels, dev_init_qrs)
    
    return train_dataset, dev_dataset

print("Training...")
def train(device, epochs, batch_size):
    print(batch_size)
    dist.init_process_group(backend = 'nccl', world_size = world_size, rank = device)
    torch.manual_seed(10)
    torch.cuda.set_device(device)
    train_dataset, dev_dataset = prep_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = world_size, rank = device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler, shuffle = False)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size = dev_dataset.__len__(), sampler = torch.utils.data.SequentialSampler(dev_dataset))
    model = AmbiguityNetwork()
    model.cuda(device)
    model = DDP(model, device_ids = [device])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay = 0.1)
    for epoch in range(epochs):
        model.train()
        print("Epoch ", epoch)
        for idx, data in enumerate(tqdm(train_loader)):
            ids, masks, token_type_ids, tgts, init_query = data
            ids = ids.to(device)
            masks = masks.to(device)
            token_type_ids = token_type_ids.to(device)
            tgts = tgts - 1
            tgts = torch.LongTensor(tgts).to(device)
            init_query.to(device)
            preds = model(ids, masks, token_type_ids, init_query)
            #print("Preds", preds)
            loss = loss_fn(preds, tgts)
            model.zero_grad()
            loss.backward()
            optimizer.step()        
        print(f"Epoch: {epoch}, Training loss: {loss}")
        
        print("Validating...")
        with torch.no_grad():
            for dev_idx, dev_data in enumerate(dev_loader):
                ids, masks, token_type_ids, dev_labels, init_query = dev_data
                ids = ids.to(device)
                masks = masks.to(device)
                token_type_ids = token_type_ids.to(device)
                val_preds = model(ids, masks, token_type_ids, init_query)
                print(val_preds)
                m = torch.nn.Softmax()
                val_npreds = m(val_preds).cpu().numpy()
                class_preds = np.argmax(val_npreds, axis = 1)
                class_preds = class_preds + 1
                val_acc = accuracy_score(dev_labels, class_preds)
                val_loss = log_loss(dev_labels, val_npreds, labels = [1,2,3,4])
                val_prec = precision_score(dev_labels, class_preds, average = 'weighted')
                val_rec = recall_score(dev_labels, class_preds, average = 'weighted')
                val_f1 = f1_score(dev_labels, class_preds, average = 'weighted')
        print(f"Validation_loss: {val_loss}, validation acc: {val_acc}, Precision: {val_prec}, Recall: {val_rec}, F1 score: {val_f1}")
        torch.save(model.state_dict(), "roberta_finetune_aug_model.pt")
        print("Model saved")

#def run_ddp(rank, world_size):
#    setup(rank, world_size)
#    train(rank, 100, batch_size)
#    cleanup()
#train(100, batch_size)

#def run_ddp_train(fn, world_size):
#    mp.spawn(fn, args = (world_size,), nprocs = world_size, join = True)

if __name__ ==  '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '3901825'
    mp.spawn(train, nprocs = world_size, args = (100,batch_size,))


