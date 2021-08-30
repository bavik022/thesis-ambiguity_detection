import pickle
from scipy.linalg import norm
from yaspin import yaspin
import numpy as np
from tqdm import tqdm

num_clusters = 20
num_docs = 100

def gen_tfidf(text, idf_dict):
    """
    Given a segmented string and idf dict, return a dict of tfidf.
    """
    tokens = text.split()
    total = len(tokens)
    tfidf_dict = {}
    for w in tokens:
        tfidf_dict[w] = tfidf_dict.get(w, 0.0) + 1.0
    for k in tfidf_dict:
        tfidf_dict[k] = tfidf_dict[k] * idf_dict.get(k, 0.0) / total
    return tfidf_dict

def cosine_sim(a, b):
    if len(b) < len(a):
        a, b = b, a
    res = 0
    for key, a_value in a.items():
        res += a_value * b.get(key, 0)
    if res == 0:
        return 0
    try:
        res = res / (norm(list(a.values())) * norm(list(b.values())))
    except ZeroDivisionError:
        res = 0
    return res

def tfidf_cos_sim(text1, text2, idf_dict):
    tfidf1 = gen_tfidf(text1, idf_dict)
    tfidf2 = gen_tfidf(text2, idf_dict)
    return cosine_sim(tfidf1, tfidf2)

f_cls = open('test_clusters.pkl', 'rb')
f_idf = open('test_idf_dict.pkl', 'rb')
f_docs = open('test_docs_concepts.pkl', 'rb')

clusters = pickle.load(f_cls)
idf_dict = pickle.load(f_idf)

docs = {}

print("Reading files...")

with yaspin():
    while True:
        try:
            data = pickle.load(f_docs)
            tid = data['topic_id']
            docs[tid] = data['docs'][:num_docs]
        except EOFError:
            break

tids = list(clusters.keys())
print(tids)

concept_node_features = {}
concept_adjacency_matrix = {}

print("Preparing graph features...")

for i in tqdm(range(len(tids))):
    tid = tids[i]
    kwd_cls = clusters[tid]
    node_features = np.zeros((num_clusters,num_docs))
    doc_clusters = {}
    adj_matrix = np.zeros((num_clusters, num_clusters))
    for j in range(num_clusters):
        doc_clusters[j] = []
        node_features[j] = np.array([tfidf_cos_sim(kwd_cls[j], doc, idf_dict[tid]) for doc in docs[tid]], dtype = np.double)
    for doc in docs[tid]:
        sims = np.array([tfidf_cos_sim(kwd, doc, idf_dict[tid]) for kwd in kwd_cls])
        maxsim = np.argmax(sims)
        doc_clusters[maxsim].append(doc)
    concept_node_features[tid] = node_features
    for t in range(num_clusters):
        doc_clusters[t] = ' '.join(doc_clusters[t])
    for p in range(num_clusters):
        adj_matrix[p] = np.array([tfidf_cos_sim(doc_clusters[p], doc_clusters[k], idf_dict[tid]) for k in doc_clusters], dtype = np.double)
    concept_adjacency_matrix[tid] = adj_matrix
    
feature_file = open('test_concept_node_features.pkl', 'wb')
adj_file = open('test_concept_adjacency_matrices.pkl', 'wb')

pickle.dump(concept_node_features, feature_file)
pickle.dump(concept_adjacency_matrix, adj_file)
        
        