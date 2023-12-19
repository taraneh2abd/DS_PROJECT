import numpy as np

def tf(doc, word):
    n = len(doc)
    occurance = len([token for token in doc if token == word])
    return occurance / n

def idf(word, word_count, total_docs):
    try:
        word_occurance = word_count[word] + 1
    except:
        word_occurance = 1
    return np.log(total_docs / word_occurance)

def tf_idf(sentence, word_set, word_index):
    vec = np.zeros((len(word_set),))
    for word in sentence:
        tf = tf(sentence, word)
        idf = idf(word)
        vec[word_index[word]] = tf * idf
    return vec