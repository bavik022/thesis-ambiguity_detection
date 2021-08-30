import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

f = open("test_docs_concepts.pkl",  'rb')

corpus = {}

while True:
    try:
        data = pickle.load(f)
        corpus[data['topic_id']] = data['docs']
    except EOFError:
        break

writefile = open('test_idf_dict.pkl', 'wb')
tids = list(corpus.keys())
idf_dict_all = {}
for i in tqdm(range(len(tids))):
    tid = tids[i]
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus[tid])
    idf = vectorizer.idf_
    idf_dict = dict(zip(vectorizer.get_feature_names(), idf))
    idf_dict_all[tid] = idf_dict
    
pickle.dump(idf_dict_all, writefile)
        