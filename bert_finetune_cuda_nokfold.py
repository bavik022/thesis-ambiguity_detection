import torch
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
print("Device", device)

bert_model = SentenceTransformer("/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/msmarco-MiniLM-L-6-v3/")
if device == "cuda:5":
    bert_model.cuda()

print("Initializing initial query encoder...")
bert_model.max_seq_length = 1000
torch.manual_seed(10)
torch.autograd.set_detect_anomaly(True)

embed_size = 256
init_query_embed_size = 384
num_docs = 10
batch_size = 10
epochs = 20
max_len = 256
epsilon = 0.0001

cos = torch.nn.CosineSimilarity()
tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', truncation = True)

print("Initializing the model...")
class SBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained('google/electra-small-discriminator')
        self.pooler = torch.nn.AvgPool1d(embed_size)
    def forward(self, ids, attention_masks, token_type_ids):
        #output = self.roberta(input_ids = ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        #print(self.roberta(ids[0], attention_masks[0], token_type_ids[0]).last_hidden_state)
        embeds = [self.roberta(id, attention_mask, token_type_id).last_hidden_state for (id, attention_mask, token_type_id) in zip(ids, attention_masks, token_type_ids)]
        embeds = [self.pooler(embed) for embed in embeds]
        embeds = [torch.squeeze(embed, dim = 2) for embed in embeds]
        ln = len(embeds)
        embeds = torch.cat(embeds, dim = 0).reshape((ln, num_docs, embed_size))
        #print("BERT output:", embeds)
        
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
    
    def forward(self, ids, attention_masks, token_type_ids, init_query):
        #ids = torch.cat(ids, dim = 0).reshape((len(ids), num_docs, max_len))
        #attention_masks = torch.cat(attention_masks, dim = 0).reshape((len(ids), num_docs, max_len))
        #token_type_ids = torch.cat(token_type_ids, dim = 0).reshape((len(ids), num_docs, max_len))
        #embeds = torch.FloatTensor((len(ids), num_docs, embed_size))
        #embeds.requires_grad = True
        #embeds = [self.sbert(id, attention_mask, token_type_id) for (id, attention_mask, token_type_id) in zip(ids, attention_masks, token_type_ids)]
        #embeds = [torch.squeeze(embed, dim = 2) for embed in embeds]
        #ln = len(embeds)
        #embeds = torch.cat(embeds, dim = 0).reshape((ln, num_docs, embed_size))
        #print(embeds)
        embeds = self.sbert(ids, attention_masks, token_type_ids)
        emb_norm = torch.nn.functional.normalize(embeds, dim = 2)
        csml = torch.matmul(emb_norm, emb_norm.transpose(1,2))
        A = csml
        D = torch.diag_embed(torch.sum(A, axis = 2))
        #D = torch.cat([torch.diag(torch.sum(mat, axis = 1)) for _,mat in enumerate(A)]).reshape((batch_size, num_docs, num_docs))
        D_inv = torch.linalg.inv(D) 
        D_inv = D_inv + epsilon
        #D_inv = (1.0/D)
        #D_inv.masked_fill_(D_inv == float('inf'), 0.)
        #D_inv = D_inv.masked_fill_(D_inv < 0., 1e-6)
        #print(D_inv)        
        D_ = torch.pow(D_inv, 0.5)
        A_ = torch.matmul(A, torch.transpose(D_, 1,2))
        A_ = torch.matmul(D_, A_)
        A = A_ + torch.eye(embeds.shape[1]).to(device)
        out = self.gcn1(embeds, A)
        out = self.dp(out)
        #print("GCN1:",out)
        out = self.gcn2(out, A)
        out = self.dp(out)
        #print("GCN2:",out)
        out = self.gcn3(out, A)
        out = self.dp(out)
        #print("GCN3:",out)
        out = torch.flatten(out, 1)
        out = torch.cat((out, init_query.to(device)), dim = 1)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

print("Preparing the data...")
requests = pd.read_table('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/data/train.tsv', sep = '\t', header = 0).drop_duplicates('topic_id')
rdev = pd.read_table('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/data/dev.tsv', sep = '\t', header = 0).drop_duplicates('topic_id')

df = requests.append(rdev)

f = open('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/retrieved_docs_encoded_d2v_keywords.pkl', 'rb')
docs = {}
while True:
    try:
        data = pickle.load(f)
        tid = list(data.keys())[0]
        docs[tid] = list(data.values())[0]
    except EOFError:
        break
f.close()

ks = list(docs.keys())
tids = requests['topic_id'].to_numpy(dtype = int)
tids = np.intersect1d(tids, np.array(ks))

inputs = []
for tid in tids:
    inputs.append(docs[tid]['docs'][:num_docs])

labels = torch.LongTensor(torch.zeros(len(tids)).long())
init_qrs = torch.zeros((len(tids), init_query_embed_size))
ids = torch.LongTensor(len(tids), num_docs, max_len)
masks = torch.LongTensor(len(tids), num_docs, max_len)
token_type_ids = torch.LongTensor(len(tids), num_docs, max_len)

print("Preparing training data...")

for i in tqdm(range(len(tids))):
    tid = tids[i]
    documents = inputs[i]
    tokens = [tokenizer.encode_plus(doc, None, add_special_tokens = True, max_length = max_len, padding = 'max_length', return_token_type_ids = True, truncation = True) for doc in documents]
    idlist = [torch.tensor(token['input_ids'], dtype = torch.long) for token in tokens]
    torch.cat(idlist, out = ids[i])
    masklist = [torch.tensor(token['attention_mask'], dtype = torch.long) for token in tokens]
    torch.cat(masklist, out = masks[i])
    tokentypelist = [torch.tensor(token['token_type_ids'], dtype = torch.long) for token in tokens]
    torch.cat(tokentypelist, out = token_type_ids[i])
    labels[i] = df.loc[df['topic_id'] == tid]['clarification_need'].to_numpy(dtype = np.int_)[0]
    init_qr = df.loc[df['topic_id'] == tid]['initial_request'].to_numpy(dtype = str)[0]
    init_qrs[i] = torch.Tensor(bert_model.encode(init_qr))

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

dev_tids = rdev['topic_id'].to_numpy(dtype = int)

dev_labels = rdev['clarification_need'].to_numpy(dtype = int)

dev_inputs = []
for tid in dev_tids:
    dev_inputs.append(docs[tid]['docs'][:num_docs])
dev_inputs = np.array(dev_inputs)

dev_init_qrs = torch.zeros((len(dev_tids), init_query_embed_size))
dev_labels = torch.LongTensor(torch.zeros(len(dev_tids)).long())
dev_ids = torch.LongTensor(len(dev_tids), num_docs, max_len)
dev_masks = torch.LongTensor(len(dev_tids), num_docs, max_len)
dev_token_type_ids = torch.LongTensor(len(dev_tids), num_docs, max_len)

print("Preparing dev data...")

for i in tqdm(range(len(dev_tids))):
    tid = dev_tids[i]
    docs = dev_inputs[i]
    tokens = [tokenizer.encode_plus(doc, None, add_special_tokens = True, max_length = max_len, padding = 'max_length', return_token_type_ids = True, truncation = True) for doc in docs]
    dev_id_list = [torch.tensor(token['input_ids'], dtype = torch.long) for token in tokens]
    torch.cat(dev_id_list, out = dev_ids[i])
    dev_mask_list = [torch.tensor(token['attention_mask'], dtype = torch.long) for token in tokens]
    torch.cat(dev_mask_list, out = dev_masks[i])
    dev_token_type_id_list = [torch.tensor(token['token_type_ids'], dtype = torch.long) for token in tokens]
    torch.cat(dev_token_type_id_list, out = dev_token_type_ids[i])
    dev_labels[i] = df.loc[df['topic_id'] == tid]['clarification_need'].to_numpy(dtype = np.int_)[0]
    init_qr = rdev.loc[rdev['topic_id'] == tid]['initial_request'].to_numpy(dtype = str)[0]
    dev_init_qrs[i] = torch.Tensor(bert_model.encode(init_qr))

train_dataset = InputDataset(ids, masks, token_type_ids, labels, init_qrs)
dev_dataset = InputDataset(dev_ids, dev_masks, dev_token_type_ids, dev_labels, dev_init_qrs)

print("Training...")
def train(epochs, batch_size):
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, sampler = torch.utils.data.RandomSampler(train_dataset))
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size = dev_dataset.__len__(), sampler = torch.utils.data.SequentialSampler(dev_dataset))
    model = AmbiguityNetwork()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay = 0.1)
    for epoch in range(epochs):
        model.train()
        print("Epoch ", epoch)
        for idx, data in enumerate(tqdm(train_loader)):
            ids, masks, token_type_ids, tgts, init_query = data
            ids = [id.to(device) for id in ids]
            masks = [mask.to(device) for mask in masks]
            token_type_ids = [tti.to(device) for tti in token_type_ids]
            tgts = tgts - 1
            tgts = torch.LongTensor(tgts).to(device)
            init_query.to(device)
            preds = model(ids, masks, token_type_ids, init_query)
            #print("Predictions:", preds)
            loss = loss_fn(preds, tgts)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            #for param in model.parameters():
            #    print("param.data",torch.isfinite(param.data).all())
            #    print("param.grad.data",torch.isfinite(param.grad.data).all(),"\n")
        print(f"Epoch: {epoch}, Training loss: {loss}")
        print("Validating...")
        with torch.set_grad_enabled(False):
            model.eval()
            for dev_idx, dev_data in enumerate(dev_loader):
                ids, masks, token_type_ids, dev_labels, init_query = dev_data
                ids = [id.to(device) for id in ids]
                masks = [mask.to(device) for mask in masks]
                token_type_ids = [tti.to(device) for tti in token_type_ids]
                val_preds = model(ids, masks, token_type_ids, init_query)
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
    torch.save(model.state_dict(), "roberta_finetune_model.pt")

train(50, batch_size)