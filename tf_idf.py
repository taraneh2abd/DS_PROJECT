import numpy as np
from tqdm import tqdm

def term_frequency(doc, word):
    n = len(doc)
    occurrence = len([token for token in doc if token == word])
    return occurrence / n

def inverse_document_frequency(word, doc):
    try:
        word_occurrence = doc['count_dict'][word] + 1
    except KeyError:
        word_occurrence = 1
    return np.log(len(doc['sentences']) / word_occurrence)

def tf_idf(sentence,word_map, doc):
    vec = {}
    for word in sentence:
        term_freq = term_frequency(sentence, word)
        idf_val = inverse_document_frequency(word, doc)
        vec[word_map[word]] = term_freq * idf_val
    return vec

def calculate_tfidf(df, word_map, candidates):
    print("Calculating TF-IDF ...")
    for candidate in candidates:
        vector = {}
        vectors = [] 
        row = df[candidate]
        for sent in row['sentences']:
            v = tf_idf(sent, word_map, row)

            for key, value in v.items():
                if key in vector:
                    vector[key] += value
                else:
                    vector[key] = value
            vectors.append(v)
            
        df[candidate]['tf-idf'] = vector
        df[candidate]['vectors'] = vectors
    
    return df
