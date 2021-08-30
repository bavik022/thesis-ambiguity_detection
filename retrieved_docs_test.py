import pickle
from tqdm import tqdm
import spacy
import pytextrank
import requests
from bs4 import BeautifulSoup
import uuid
import pandas as pd


api_key = "83eaf18c-f42b-49e1-8d97-1062f00ac2eb"
url = "https://www.chatnoir.eu/api/v1/_search"
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

doc_data = pd.read_pickle("./data/top10k_docs_dict.pkl")

writefile = open('doclists_test.pkl', 'wb')
with open('test_init_qrs.pkl', 'rb') as file:
    while True:
        try:
            data = pickle.load(file)
            print("Topic:", data['topic_id'])
            query_doc_list = []
            res = doc_data[int(data['topic_id'])][:100]
            for i in tqdm(range(len(res))):
                id = res[i]
                index = id[0:9]
                doc_id = ':'.join((index, id))
                uid = uuid.uuid5(uuid.NAMESPACE_URL, doc_id)
                doc_index = 'cw09' if index == 'clueweb09' else 'cw12'
                doc_url = 'https://www.chatnoir.eu/cache?apikey=' + api_key + '&uuid=' + str(uid) + '&index=' + doc_index +'&raw&plain'
                doc = requests.get(doc_url).text
                doc = BeautifulSoup(doc, "lxml").text.replace('\n', ' ')
                kws = nlp(doc)._.phrases[:100]
                kws = [kw.text for kw in kws]
                doc_sum = ' '.join(kws)
                query_doc_list.append(doc_sum)              
            add_dict = {
                'topic_id': data['topic_id'],
                'queries': data['queries'],
                'docs': query_doc_list,
                }
            pickle.dump(add_dict, writefile)
        except EOFError:
            break
writefile.close()

