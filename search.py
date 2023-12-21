from word_tokenizer import tokenizer 
from dataset_generator import count_dict
from cosine_similarity import cosine_similarity2
from difflib import SequenceMatcher, get_close_matches
from tf_idf import calculate_tfidf, calculate_tfidf2
from termcolor import colored
import numpy as np

def search(sentence, candidates, df, unique_tokens, word_map):
    res={}

    sent = tokenizer(sentence)
    tokens = similar_tokens(sent, unique_tokens)
    data = {'sentences': tokens, 'count_dict': count_dict(tokens, tokens),'tf-idf': None}
    # data['tf-idf'] = tf_idf(tokens, word_map, data)
    # print(tokens)
    # print(sent)
    print(colored("!!!Calculating tf_idf of documents!!!", "red"))
    df = calculate_tfidf(df, word_map, candidates)

    tf_arr = []
    tf_arr.append([tokens])
    n =1
    for i in candidates:
        tf_arr.append(df[i]['sentences'])
        n += len(df[i]['sentences'])

    tf_arr = calculate_tfidf2(tf_arr, word_map, n)
    data['tf-idf'] = tf_arr[0][0]
    
    doc_tfidf = []
    for i in tf_arr[1:]:
        sumd = {}
        for j in i:
            sumd = sum_dicts([sumd, j])
        doc_tfidf.append(sumd)
    for i, candidate in enumerate(candidates):
        res[candidate] = cosine_similarity2(data['tf-idf'],doc_tfidf[i], len(unique_tokens))
    res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

    ans = next(iter(res)) 
    target = []
    max_val = res[ans]
    # for i ,v in enumerate(df[ans]['sentences']):
    #     print(f"sent {i}", v)

    # tf_arr = []
    # tf_arr.append(tokens)
  
    # for i in df[ans]['sentences']:
    #     tf_arr.append(i)

    # tf_arr = calculate_tfidf2(tf_arr, word_map)
    # data['tf-idf'] =  tf_arr[0]
    for v in tf_arr[candidates.index(ans)]:
        target.append(cosine_similarity2(data['tf-idf'],v, len(unique_tokens)))
    

    max_val_sent = 0
    max_inx = 0
    for i in range(len(target)):
        if max_val_sent < target[i]:
            max_inx = i
            max_val_sent = target[i]


    return ans, max_inx, max_val, max_val_sent

    

