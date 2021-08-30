import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import string
import nltk
from sympy.utilities.iterables import multiset_permutations

nltk.download('punkt', download_dir = '/mount/arbeitsdaten43/projekte/thesis-dp-1/banerjak/virt_env/nltk_data')

data_dir = './data/'

requests = pd.read_table(data_dir + "test.tsv", sep = '\t', header = 0)
df = pd.DataFrame(requests)
init_requests = df['initial_request'].to_numpy(dtype = str)
topic_ids = df['topic_id'].to_numpy(dtype = int)
req_data = [(topic_ids[i], init_requests[i]) for i in range(len(init_requests))]
req_data = np.array(req_data)
inputs = req_data[:,1]
topics = req_data[:,0]

inputs = np.array([req.lower() for req in inputs])
inputs = np.array([req.strip() for req in inputs])
inputs = np.array([req.translate(str.maketrans('', '', string.punctuation)) for req in inputs])


writefile = open('test_init_qrs.pkl', 'wb')
for i in tqdm(range(len(topics))):
    add_dict = {
      'topic_id': topics[i],
      'queries': inputs[i],
      }
    pickle.dump(add_dict, writefile)

writefile.close()