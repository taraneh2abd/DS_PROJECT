from cosine_similarity import cosine_similarity
from termcolor import colored
from tqdm import tqdm
from dataset_generator import DatasetGenerator
def search(qcase, candidates, df,unique_tokens, word_map ):
    res={}

    # data['tf-idf'] = tf_idf(tokens, word_map, data)
    # print(tokens)
    # print(sent)
    print(colored("!!!Calculating tf_idf of query!!!", "red"))
    dataset_gerator = DatasetGenerator()
    qcase['sentence'] = dataset_gerator.similar_tokens(qcase, unique_tokens)
    qcase = dataset_gerator.calculate_tfidf2(qcase, df, word_map)

    for i, candidate in enumerate(candidates):
        res[candidate] = cosine_similarity(qcase['tf_idf'],df[candidate]['tf_idf'])
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
    for v in tqdm(df[ans]['vectors']):
        target.append(cosine_similarity(qcase['tf_idf'],v))
    

    max_val_sent = 0
    max_inx = 0
    for i in range(len(target)):
        if max_val_sent < target[i]:
            max_inx = i
            max_val_sent = target[i]


    return ans, max_inx, max_val, max_val_sent

    

