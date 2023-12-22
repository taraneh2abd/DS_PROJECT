import math
from termcolor import colored
from tqdm import tqdm
import pickle
import base64

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    dot_product = sum(v1.get(term, 0) * v2.get(term, 0) for term in set(v1) & set(v2))
    magnitude1 = math.sqrt(sum(value**2 for value in v1.values()))
    magnitude2 = math.sqrt(sum(value**2 for value in v2.values()))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)

def search(qcase, candidates, df,unique_tokens, dataset_generator ):
    res={}

    print(colored("!!!Calculating tf_idf of query!!!", "red"))

    qcase['sentence'] = dataset_generator.similar_tokens(qcase, unique_tokens)
    qcase = dataset_generator.calculate_tfidf2(qcase)
    print(qcase['sentence'])
    print(qcase['tf_idf'])
    for i, candidate in enumerate(candidates):
        res[candidate] = cosine_similarity(qcase['tf_idf'],df[candidate]['tf_idf'])
    res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

    ans = next(iter(res)) 
    target = []
    max_val = res[ans]
  
    for v in tqdm(df[0]['vectors']):
        target.append(cosine_similarity(qcase['tf_idf'],v))
    

    max_val_sent = 0
    max_inx = 0
    for i in range(len(target)):
        if max_val_sent < target[i]:
            max_inx = i
            max_val_sent = target[i]


    return ans, max_inx, max_val, max_val_sent

    
def save_pickle_to_text(data, filename):

    serialized_data = pickle.dumps(data)

    encoded_data = base64.b64encode(serialized_data).decode('utf-8')

    with open(filename, 'w', encoding='utf-8') as text_file:
        text_file.write(encoded_data)

def load_pickle_from_text(filename):

    with open(filename, 'r', encoding='utf-8') as text_file:
        encoded_data = text_file.read()

    decoded_data = base64.b64decode(encoded_data)

    data = pickle.loads(decoded_data)
    return data
