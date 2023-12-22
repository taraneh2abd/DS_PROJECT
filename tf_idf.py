import math
from collections import Counter

class TFIDFVectorizer:
    def __init__(self, word_map):
        self.inverse_document_frequency = None
        self.vocab = None
        self.word_map = word_map

    def fit(self, documents):
        
        self.inverse_document_frequency = self.calculate_inverse_document_frequency(documents)
        self.vocab = set(self.inverse_document_frequency.keys())

    def transform(self, documents, data_type):
       
        tfidf_values = {}
        if data_type == 'documents':

            for document in documents:
                doc_tfidf = []
                for sentence in document['sentences']:
                    tf_arr = {}
                    term_frequency = self.calculate_term_frequency(sentence)
                    tfidf_sentence = {term: tf * self.inverse_document_frequency[term] for term, tf in term_frequency.items()}
                    for key, val in tfidf_sentence.items():
                        tf_arr[self.word_map[key]] = val
                    doc_tfidf.append(tf_arr)
                tfidf_values[document["document_id"]] = doc_tfidf
        
        else:

           
            tf_arr = {}
            term_frequency = self.calculate_term_frequency(documents['sentence'])
            tfidf_sentence = {term: tf * self.inverse_document_frequency[term] for term, tf in term_frequency.items()}
            for key, val in tfidf_sentence.items():
                tf_arr[self.word_map[key]] = val
            return tf_arr

        return tfidf_values

    def fit_transform(self, documents, data_type):
       
        self.fit(documents)
        return self.transform(documents, data_type)

    def calculate_term_frequency(self, document):
        term_frequency = {}
        total_terms = len(document)

        for term in document:
            term_frequency[term] = term_frequency.get(term, 0) + 1 / total_terms
        return term_frequency

    def calculate_inverse_document_frequency(self, documents):
        document_count = len(documents)
        term_document_count = Counter()

        for document in documents:
            for sent in document['sentences']:
                unique_terms = set(sent)
                term_document_count.update(unique_terms)

        inverse_document_frequency = {}

        for term, count in term_document_count.items():
            inverse_document_frequency[term] = math.log(document_count / (count + 1))

        return inverse_document_frequency