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

os.environ['CUDA_VISIBLE_DEVICES'] = '6,5'

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
        print(A[0])
        S = torch.sum(A, axis = 2)
        S = S + epsilon
        S = torch.pow(S, -0.5)
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
    
    requests = pd.read_table('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/data/dev.tsv', sep = '\t', header = 0)
    
    df = requests
    
    f = open('/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/doclists_dev.pkl', 'rb')
    docs = {}
    queries = {}
    query_list = []
    while True:
        try:
            data = pickle.load(f)
            tid = int(data['topic_id'])
            docs[tid] = data['docs']
            queries[tid] = data['queries']
        except EOFError:
            break
    f.close()
    
    tids = list(docs.keys())
    
    inputs = []
    for tid in tids:
        inputs.append(docs[tid][:num_docs])
    
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
        init_qrs[i] = torch.Tensor(bert_model.encode(queries[tid]))
    
    return tids, ids, masks, token_type_ids, init_qrs

print("Testing")
def test():
    tids, ids, masks, token_type_ids, init_qrs = prep_data()
    model = torch.nn.DataParallel(AmbiguityNetwork())
    model.load_state_dict(torch.load('roberta_finetune_aug_model_for_test.pt'))
    model.cuda(device)
    model.eval()
    print("Testing...")
    ids = ids.to(device)
    masks = masks.to(device)
    token_type_ids = token_type_ids.to(device)
    with torch.no_grad():
        val_preds = model(ids, masks, token_type_ids, init_qrs)
    m = torch.nn.Softmax()
    val_npreds = m(val_preds).cpu().numpy()
    class_preds = np.argmax(val_npreds, axis = 1)
    class_preds = class_preds + 1
    outs = [(tids[i], class_preds[i]) for i in range(len(class_preds))]
    outs = np.array(outs)
    np.savetxt('preds_model_1_dev.txt', outs, fmt="%s %s")
    np.save('classpreds_model_1_dev.npy', class_preds)

test()



