from word_tokenizer import tokenizer 
from dataset_generator import count_dict
from tf_idf import tf_idf
from cosine_similarity import cosine_similarity
from difflib import SequenceMatcher
from tf_idf import calculate_tfidf
from termcolor import colored

CONTENT_FILE_PATTERN = "document_{index}.txt"

def similar_tokens(query, unique_tokens, threshold=0.8):
    similar_tokens = []

    for token in unique_tokens:
        similarity_ratio = SequenceMatcher(None, query, token).ratio()

        if similarity_ratio > threshold:
            similar_tokens.append(token)

    return similar_tokens


def search(sentence, candidates, df, unique_tokens, word_map):
    res={}

    sent = tokenizer(sentence)
    tokens = similar_tokens(sent, unique_tokens)

    data = {'sentences': tokens, 'count_dict': count_dict(tokens, tokens),'tf-idf': None}
    data['tf-idf'] = tf_idf(tokens, word_map, data)

    print(colored("!!!Calculating tf_idf of documents!!!", "red"))
    df = calculate_tfidf(df, word_map, candidates)
    
    for candidate in candidates:
        res[candidate] = cosine_similarity(data['tf-idf'],df[candidate]['tf-idf'])
    res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

    ans = next(iter(res)) 
    print(f"ans = {ans}")
    target = []

    for v in df[ans]['vectors']:
        target.append(cosine_similarity(data['tf-idf'],v))
    

    max_val = 0
    max_inx = 0
    for i in range(len(target)):
        if max_val < target[i]:
            max_inx = i
            max_val = target[i]


    return ans, max_inx, max_val

    

