import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import torch
from yaspin import yaspin
from sklearn.cluster import KMeans
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '8'
num_clusters = 20

f = open('test_docs_concepts.pkl', 'rb')

bert_model = SentenceTransformer("/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/roberta-base-nli-mean-tokens/")
if torch.cuda.is_available():
    bert_model.cuda()

keywords = {}

print("Reading document data...")

with yaspin():
    while True:
        try:
            data = pickle.load(f)
            tid = data['topic_id']
            print(tid)
            kwds = data['main_keywords']
            kwd_list = []
            for kwd in kwds:
                kwd_list.extend(kwd)
            kwd_enc = [bert_model.encode(kwds) for kwds in kwd_list]
            keywords[tid] = {
              'keywords': kwd_list,
              'enc_kwds': kwd_enc
            }
        except EOFError:
            break
wrf = open('test_keywords_encoded.pkl', 'wb')
pickle.dump(keywords, wrf)

#f = open('keywords_encoded.pkl', 'rb')
#keywords = pickle.load(f)

tids = list(keywords.keys())

writefile = open('test_clusters.pkl', 'wb')
commclusters = {}

for i in tqdm(range(len(tids))):
    tid = tids[i]
    kmeans = KMeans(n_clusters = num_clusters)
    enc = np.array(keywords[tid]['enc_kwds'])
    kwrds = keywords[tid]['keywords']
    labels = kmeans.fit_predict(enc)
    clusters = ['']*num_clusters
    for i in range(len(labels)):
        if clusters[labels[i]] == '':
            clusters[labels[i]] = kwrds[i]
        else:
            clusters[labels[i]] = clusters[labels[i]] + ' ' + kwrds[i]
    commclusters[tid] = clusters

pickle.dump(commclusters, writefile)
    
    

    
