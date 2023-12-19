import numpy as np

def tf(doc, word):
    n = len(doc)
    occurance = len([token for token in doc if token == word])
    return occurance / n

def idf(word,doc):
    try:
        word_occurance = doc['count_dict'][word] + 1
    except:
        word_occurance = 1
    return np.log(len(doc['sentences']) / word_occurance)

def tf_idf(sentence, word_set, word_map, doc):
    vec = np.zeros((len(word_set)))
    for word in sentence:
        tf = tf(sentence, word)
        idf = idf(word, doc)
        vec[word_map[word]] = tf * idf
    return vec

def calculate_tfidf(df, unique_tokens, word_map):
    for index, row in df.iterrows():
        vector = np.zeros(len(unique_tokens))
        vectors = [] 
        for sent in row['sentences']:
            v = tf_idf(sent, unique_tokens, word_map, row)
            vectors.append(v)
            vector += v
        row['vectors'] = vectors
        row['tf-idf'] = vector
    return df
