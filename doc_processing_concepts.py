import pandas as pd
import requests
from tqdm import tqdm
import uuid
import pickle
from bs4 import BeautifulSoup
import nltk
import spacy
import pytextrank
import numpy as np
import string
import stanza

url = "https://www.chatnoir.eu/api/v1/_search"

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
nlp.max_length = 5000000

stanza.download('en', model_dir = "/mount/arbeitsdaten/thesis-dp-1/banerjak/stanza_resources")
#nlp_ner = stanza.Pipeline('en', dir = "/mount/arbeitsdaten/thesis-dp-1/banerjak/stanza_resources", processors = 'tokenize, ner')

nltk.download('punkt')
nltk.download('stopwords')

api_key = "83eaf18c-f42b-49e1-8d97-1062f00ac2eb"

data_dir = './data/'

reqs = pd.read_table(data_dir + "dev.tsv", sep = '\t', header = 0).drop_duplicates('topic_id')
df = pd.DataFrame(reqs)
init_requests = df['initial_request'].to_numpy(dtype = str)
topic_ids = df['topic_id'].to_numpy(dtype = int)
clarification_need = df['clarification_need'].to_numpy(dtype = int)
req_data = [(topic_ids[i], init_requests[i], clarification_need[i]) for i in range(len(init_requests))]
req_data = np.array(req_data)
inputs = req_data[:,1]
labels = req_data[:,2]
topics = req_data[:,0]

inputs = np.array([req.lower() for req in inputs])
inputs = np.array([req.strip() for req in inputs])
inputs = np.array([req.translate(str.maketrans('', '', string.punctuation)) for req in inputs])

doc_data = pd.read_pickle("./data/top10k_docs_dict.pkl")

writefile = open('dev_docs_concepts.pkl', 'wb')

for i in tqdm(range(len(inputs))):
    res = doc_data[int(topics[i])][:100]
    doc_list = []
    kw_list = []
    main_kw_list = []
    ner_list = []
    for r in res:
        id = r
        index = id[0:9]
        doc_id = ':'.join((index, id))
        uid = uuid.uuid5(uuid.NAMESPACE_URL, doc_id)
        doc_index = 'cw09' if index == 'clueweb09' else 'cw12'
        doc_url = 'https://www.chatnoir.eu/cache?apikey=' + api_key + '&uuid=' + str(uid) + '&index=' + doc_index +'&raw&plain'
        doc = requests.get(doc_url).text
        doc = BeautifulSoup(doc, "lxml").text.replace('\n', ' ')
        doc_list.append(doc)
        kws = nlp(doc)._.phrases
        kws = [kw.text for kw in kws]
        main_kws = kws[:100]
        kw_list.append(kws)
        main_kw_list.append(main_kws)
    add_dict = {
        'topic_id': topics[i],
        'query': inputs[i],
        'docs': doc_list,
        'label': labels[i],
        'keywords': kw_list,
        'main_keywords': main_kw_list,
        }
    pickle.dump(add_dict, writefile)
writefile.close()