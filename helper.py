import math
from termcolor import colored
from tqdm import tqdm
from dataset_generator import DatasetGenerator

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    dot_product = sum(v1.get(term, 0) * v2.get(term, 0) for term in set(v1) & set(v2))
    magnitude1 = math.sqrt(sum(value**2 for value in v1.values()))
    magnitude2 = math.sqrt(sum(value**2 for value in v2.values()))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)

def search(qcase, candidates, df,unique_tokens, tfidf_vectorizer ):
    res={}

    print(colored("!!!Calculating tf_idf of query!!!", "red"))
    dataset_gerator = DatasetGenerator()
    qcase['sentence'] = dataset_gerator.similar_tokens(qcase, unique_tokens)
    qcase = dataset_gerator.calculate_tfidf2(qcase,tfidf_vectorizer)

    for i, candidate in enumerate(candidates):
        res[candidate] = cosine_similarity(qcase['tf_idf'],df[candidate]['tf_idf'])
    res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

    ans = next(iter(res)) 
    target = []
    max_val = res[ans]

    for v in tqdm(df[ans]['vectors']):
        target.append(cosine_similarity(qcase['tf_idf'],v))
    

    max_val_sent = 0
    max_inx = 0
    for i in range(len(target)):
        if max_val_sent < target[i]:
            max_inx = i
            max_val_sent = target[i]


    return ans, max_inx, max_val, max_val_sent

    

