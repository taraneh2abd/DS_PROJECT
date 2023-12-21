import json
from pathlib import Path
from termcolor import colored
from tf_idf import TFIDFVectorizer
from difflib import SequenceMatcher, get_close_matches
from word_tokenizer import WordTokenizer
from tqdm import tqdm
import numpy as np

class DatasetGenerator:

    def __init__(self, ds_path = "dataset", clean_ds_file = "dataset.json", content_file_pattern = "document_{index}.txt",
                 content_dir = "dataset/data", query_file = "data.json", clean_query = "clean_query.json"):
        self.CONTENT_FILE_PATTERN = content_file_pattern
        self.DS_PATH = Path(ds_path)
        self.CLEAN_DS_FILE = Path(clean_ds_file)
        self.CONTENT_DIR = Path(content_dir)
        self.QUERY_FILE = Path(query_file)
        self.CLEAN_QUERY = Path(clean_query)
        

    def sum_dicts(self, dicts):
        result_dict = {}

        for d in dicts:
            for key, value in d.items():
                result_dict[key] = result_dict.get(key, 0) + value

        return result_dict
    
    def count_dict(self, sentences, word_set):
        count_dict = {token: 0 for token in word_set}  

        for sentence in tqdm(sentences):
            for word in sentence:
                if word in count_dict:
                    count_dict[word] += 1

        return count_dict
    def save_json(self, data, data_type):
        if data_type == "data":
            with open(Path(self.DS_PATH / self.CLEAN_DS_FILE), 'w') as json_file:
                    json.dump(data, json_file, indent=4)
            return f"Clean data seved to {Path(self.DS_PATH / self.CLEAN_DS_FILE)}"
        else:
            with open(Path(self.DS_PATH / self.CLEAN_QUERY), 'w') as json_file:
                    json.dump(data, json_file, indent=4)
            return f"Query data seved to {Path(self.DS_PATH / self.CLEAN_QUERY)}"
    def load_json(self, path, file_name):
        data = json.loads(open(path, encoding="UTF8").read())

        return data, f"{file_name} loaded from {path}"
    
    def dataset_format_converter(self, CONTENT_FILE_PATTERN, CONTENT_DIR):
        data_list = []
        query = []
        unique_tokens = set()
        index = 0
        tokenizer =  WordTokenizer()
        word_map = {}
        if not Path(self.DS_PATH / self.CLEAN_DS_FILE).is_file():
            
            while Path(CONTENT_DIR /CONTENT_FILE_PATTERN.format(index = index)).is_file():
                sentences = []
                tokens = set()
                file_path = self.CONTENT_DIR / self.CONTENT_FILE_PATTERN.format(index=index)

                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        sent = tokenizer.tokenizer(line.strip())
                        sentences.append(sent)
                        tokens |= set(sent)
                
                unique_tokens |= tokens

                data_list.append({'document_id': index, 'unique_Tokens': tokens,'sentences': sentences,'vectors': None,'tf_idf': None})
                index += 1

            for i, word in enumerate(unique_tokens):
                word_map[word] = i
            self.tfidf_vectorizer =  TFIDFVectorizer(word_map)
            for item in data_list:
                for key, value in item.items():
                    if isinstance(value, set):
                        item[key] = list(value)
            query_data, message1 = self.load_json(Path(self.DS_PATH/self.QUERY_FILE), "data.json")
            print(colored(message1, "blue"))
            
            for index, case in enumerate(query_data):
                query.append({'case_id': index, 'sentence': case['query'],'tf_idf': None, "document_id": case['document_id'], 
                              "candidate_documents_id": case['candidate_documents_id'], "is_selected": case['is_selected']})
            
            for index, case in enumerate(query):
                query[index]['sentence'] = tokenizer.tokenizer(case['sentence'])

            data = self.calculate_tfidf(data_list)

            message2 = self.save_json(data, "data")
            print(colored(message2, "yellow"))

            message1 = self.save_json(query, "query")
            print(colored(message1, "yellow"))

            return data, query, unique_tokens, word_map
        else:
            data, message = self.load_json(Path(self.DS_PATH / self.CLEAN_DS_FILE), "dataset.json")

            for document in data:
                unique_tokens |= set(document['unique_Tokens'])
            self.unique_tokens= unique_tokens
            for i, word in enumerate(unique_tokens):
                word_map[word] = i
            self.tfidf_vectorizer =  TFIDFVectorizer(word_map)

            print(colored(message, "blue"))

            query_data, message1 = self.load_json(Path(self.DS_PATH/self.CLEAN_QUERY), "clean_query.json")

            return data, query_data, unique_tokens, word_map

        
    
    def calculate_tfidf(self, data):
        tfidf_results = self.tfidf_vectorizer.fit_transform(data, 'documents')
        
        for key, value in tqdm(tfidf_results.items()):
            data[key]['vectors'] = value
            sum_vec = {}
            for sentence in value:
                sum_vec = self.sum_dicts([sum_vec, sentence])
            data[key]['tf_idf'] = sum_vec
        
        return data
            

    def calculate_tfidf2(self, query_data, df,word_map):

        self.tfidf_vectorizer =  TFIDFVectorizer(word_map)
        self.tfidf_vectorizer.fit(df)
        query_data['tf_idf'] = self.tfidf_vectorizer.transform(query_data, 'query')
        
        return  query_data
    
    def similar_tokens(self, case,unique_tokens, threshold=0.8):
        similar_tokens = []
        max_val = 0
        similarity_ratio = 0
        for token in case['sentence']:
            max_val = 0
            seq = SequenceMatcher()
            words = get_close_matches(token, unique_tokens)
            # print(token, words)
            for word in words:
                seq.set_seqs(token, word)
                similarity_ratio = seq.ratio()*100
                # print(word, similarity_ratio)
                if similarity_ratio > max_val:
                    target = word
                    max_val = similarity_ratio
            if similarity_ratio > threshold:
                similar_tokens.append(target)

        return similar_tokens

    # def similar_tokens(self, query, self., threshold=0.8):
    #     result = {}

    #     for case in tqdm(query):
    #         case_id = case['case_id']
    #         tokens = case['sentence']

    #         similar_tokens = []
    #         for token in tokens:
    #             words = get_close_matches(token, unique_tokens, n=5, cutoff=threshold)
    #             if not words:
    #                 continue

    #             ratios = np.array([SequenceMatcher(None, token, word).ratio() for word in words])
    #             max_index = np.argmax(ratios)

    #             if ratios[max_index] > threshold:
    #                 similar_tokens.append(words[max_index])

    #         result[case_id] = similar_tokens

    #     return result


