import numpy as np

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

def tf_idf(sentence, word_set, word_map, doc):
    vec = np.zeros((len(word_set)))
    for word in sentence:
        term_freq = term_frequency(sentence, word)
        idf_val = inverse_document_frequency(word, doc)
        vec[word_map[word]] = term_freq * idf_val
    return vec

def calculate_tfidf(df, unique_tokens, word_map, canditates):
    
    for candidate in canditates:
        vector = np.zeros(len(unique_tokens))
        vectors = [] 
        row = df.loc[canditates]
        for sent in row['sentences']:
            v = tf_idf(sent, unique_tokens, word_map, row)
            vectors.append(v)
            vector += v
            
        df.iloc[candidate, df.columns.get_loc('tf-idf')] = vector
        df.iloc[candidate, df.columns.get_loc('vectors')] = vectors
    
    return df
