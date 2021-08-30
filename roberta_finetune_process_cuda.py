import torch
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sentence_transformers import SentenceTransformer, models
from sklearn.model_selection import KFold
from transformers import RobertaModel, RobertaTokenizer

k_folds = 5

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print("Device", device)

bert_model = SentenceTransformer('./roberta-base-nli-mean-tokens/')
if device == "cuda:7":
    bert_model.cuda()

print("Initializing initial query encoder...")
bert_model.max_seq_length = 1000
torch.manual_seed(10)

embed_size = 512
init_query_embed_size = 768
num_docs = 20
batch_size = 10
epochs = 20
max_len = 512

cos = torch.nn.CosineSimilarity()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation = True)

print("Initializing the model...")
class SBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.pooler = torch.nn.AvgPool1d(embed_size)
    def forward(self, ids, attention_mask, token_type_ids):
        output = self.roberta(input_ids = ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = output[0]
        output = self.pooler(output)
        return output

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
        embeds = [self.sbert(id, attention_mask, token_type_id) for (id, attention_mask, token_type_id) in zip(ids, attention_masks, token_type_ids)] 
        embeds = [torch.squeeze(embed, dim = 2) for embed in embeds]
        ln = len(embeds)
        embeds = torch.cat(embeds, dim = 0).reshape((ln, num_docs, embed_size))
        emb_norm = torch.nn.functional.normalize(embeds, dim = 2)
        csml = torch.matmul(emb_norm, emb_norm.transpose(1,2))
        A = torch.where(csml>0.1, csml, torch.zeros(csml.shape))
        D = torch.diag_embed(torch.sum(A, axis = 2))
        D_ = torch.pow(D, -0.5)
        A_ = torch.matmul(A, D_)
        A_ = torch.matmul(D_, A)
        A = A_ + torch.eye(embeds.shape[1])
        out = self.gcn1(embeds, A)
        out = self.dp(out)
        out = self.gcn2(out, A)
        out = self.dp(out)
        out = self.gcn3(out, A)
        out = self.dp(out)
        out = torch.flatten(out, 1)
        out = torch.cat((out, init_query), dim = 1)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

print("Preparing the data...")
requests = pd.read_table('./data/train.tsv', sep = '\t', header = 0).drop_duplicates('topic_id')
rdev = pd.read_table('./data/dev.tsv', sep = '\t', header = 0).drop_duplicates('topic_id')

df = requests.append(rdev)

f = open('retrieved_docs_encoded_d2v_keywords.pkl', 'rb')
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
dataset = torch.utils.data.ConcatDataset([train_dataset, dev_dataset])
kfold = KFold(n_splits = k_folds, shuffle = True)

print("Training...")
def train(epochs, batch_size):
    loss_fn = torch.nn.CrossEntropyLoss()
    results = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(len(train_ids), len(test_ids))
        print(f"Fold: {fold}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = train_subsampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = test_subsampler)
        model = AmbiguityNetwork()
        #model = torch.nn.DataParallel(model).to(device)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.1)
        curr_count = 0
        acc = 0
        curr_loss = 0
        for epoch in tqdm(range(epochs)):
            model.train()
            for idx, data in enumerate(train_loader):
                ids, masks, token_type_ids, tgts, init_query = data
                ids = [id.to(device) for id in ids]
                masks = [mask.to(device) for mask in masks]
                token_type_ids = [tti.to(device) for tti in token_type_ids]
                tgts = tgts - 1
                tgts = torch.LongTensor(tgts).to(device)
                init_query.to(device)
                preds = model(ids, masks, token_type_ids, init_query)
                loss = loss_fn(preds, tgts)
                m = torch.nn.Softmax()
                npreds = m(preds).detach().numpy()
                class_preds = np.argmax(npreds, axis = 1)
                acc += accuracy_score(tgts.detach().numpy(), class_preds)
                curr_loss += loss.item()
                curr_count += 1
                model.zero_grad()
                loss.backward()
                optimizer.step()
        curr_loss = curr_loss/curr_count
        acc = acc/curr_count

        with torch.set_grad_enabled(False):
            model.eval()
            val_acc = 0
            val_loss = 0
            val_prec = 0
            val_rec = 0
            val_f1 = 0
            total = 0
            count = 0
            for dev_idx, dev_data in enumerate(test_loader):
                ids, masks, token_type_ids, dev_labels, init_query = dev_data
                val_preds = model(ids, masks, token_type_ids, init_query)
                m = torch.nn.Softmax()
                val_npreds = m(val_preds).detach().numpy()
                class_preds = np.argmax(val_npreds, axis = 1)
                class_preds = class_preds + 1
                val_acc += accuracy_score(dev_labels, class_preds)
                val_loss += log_loss(dev_labels, val_npreds, labels = [1,2,3,4])
                val_prec += precision_score(dev_labels, class_preds, average = 'weighted')
                val_rec += recall_score(dev_labels, class_preds, average = 'weighted')
                val_f1 += f1_score(dev_labels, class_preds, average = 'weighted')
                count += 1
        results[fold] = {
            'training_loss': curr_loss,
            'training_acc': acc,
            'validation_loss': val_loss/count,
            'validation_acc': val_acc/count,
            'precision': val_prec/count,
            'recall': val_rec/count,
            'f1': val_f1/count
        }
        print(f"Loss: {curr_loss}, training acc: {acc}, validation_loss: {val_loss/count}, validation acc: {val_acc/count}")

    train_acc = 0
    train_loss = 0
    val_acc = 0
    val_loss = 0
    prec = 0
    f1 = 0
    rec = 0
    for key, value in results.items():
        train_acc += value['training_acc']
        train_loss += value['training_loss']
        val_acc += value['validation_acc']
        val_loss += value['validation_loss']
        prec += value['precision']
        rec += value['recall']
        f1 += value['f1']
    print('Final Metrics:')
    print(f"Training loss: {train_loss/k_folds}, Training acc: {train_acc/k_folds}, Validation loss: {val_loss/k_folds}, Validation acc: {val_acc/k_folds}, Precision: {prec/k_folds}, Recall: {rec/k_folds}, F1: {f1/k_folds}")

train(50, batch_size)