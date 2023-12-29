#class of DatasetGenerator

import json
from pathlib import Path
from termcolor import colored
from tf_idf import TFIDFVectorizer
from difflib import SequenceMatcher, get_close_matches
from word_tokenizer import WordTokenizer
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class DatasetGenerator:

    def __init__(self, ds_path = "dataset", clean_ds_file = "dataset.json", content_file_pattern = "document_{index}.txt",unique_tokens = "unique_tokens.json",
                 content_dir = "dataset/data", query_file = "data.json", clean_query = "clean_query.json", word_map = "word_map.json", json_patt ="clean_data_{index}.json",
                 clean_data_folder = "clean_data"):
        self.CONTENT_FILE_PATTERN = content_file_pattern
        self.DS_PATH = Path(ds_path)
        self.CLEAN_DS_FILE = Path(clean_ds_file)
        self.CONTENT_DIR = Path(content_dir)
        self.QUERY_FILE = Path(query_file)
        self.CLEAN_QUERY = Path(clean_query)
        self.WORD_MAP =  Path(word_map)
        self.UNIQUE_TOKENS =  Path(unique_tokens)
        self.word_map = {}
        self.BATCH_SIZE = 100
        self.CONTENT_jSON_PATTERN = json_patt
        self.CLEAN_DS_FOLDER = Path(clean_data_folder)


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
    def save_json(self, data, path, name):
            
        with open(path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        return f"{name} seved to {path}"

    def load_json(self, path, file_name):
        try:
            with open(path, encoding="UTF8") as json_file:
                data = json.load(json_file)
            return data, f"{file_name} loaded from {path}"
        except Exception as e:
            return None, f"Error loading {file_name} from {path}: {str(e)}"
    
    def dataset_format_converter(self, CONTENT_FILE_PATTERN, CONTENT_DIR):
        logger.info("Script started.")
        data_list = []
        query = []
        unique_tokens = set()
        index = 0
        tokenizer =  WordTokenizer()

        if not Path(self.DS_PATH / self.CLEAN_DS_FILE).is_file():
            

            dd = Path(self.DS_PATH / self.CLEAN_DS_FOLDER / self.CONTENT_jSON_PATTERN.format(index=0))
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

                data_list.append({'document_id': index,'sentences': sentences,'vectors': None,'tf_idf': None})
                index += 1
            
            if isinstance(unique_tokens, set):
                uniques = list(unique_tokens)
            

            for i, word in enumerate(uniques):
                self.word_map[word] = i
            
            message = self.save_json(uniques,Path(self.DS_PATH / self.UNIQUE_TOKENS), "unique_tokens")
            print(colored(message, "yellow"))
            message = self.save_json(list(self.word_map),Path(self.DS_PATH / self.WORD_MAP), "word_map")
            print(colored(message, "yellow"))
        
            self.tfidf_vectorizer =  TFIDFVectorizer(self.word_map)
            for item in data_list:
                for key, value in item.items():
                    if isinstance(value, set):
                        item[key] = list(value)
        
            data, top_five, most_frequent = self.calculate_tfidf(data_list)

            message = self.save_json(data,Path(self.DS_PATH / self.CLEAN_DS_FILE), "Clean data")
            print(colored(message, "yellow"))

            message = self.save_json(top_five,Path(self.DS_PATH / "top_five.json"), "top five")
            print(colored(message, "yellow"))

            message = self.save_json(most_frequent,Path(self.DS_PATH / "most_frequen.json"), "most_frequen")
            print(colored(message, "yellow"))
            query_data, message = self.load_json(Path(self.DS_PATH/self.QUERY_FILE), "data.json")
            print(colored(message, "blue"))
            
            for index, case in enumerate(query_data):
                query.append({'case_id': index, 'sentence': case['query'],'tf_idf': None, "document_id": case['document_id'], 
                              "candidate_documents_id": case['candidate_documents_id'], "is_selected": case['is_selected']})
            
            for index, case in enumerate(query):
                query[index]['sentence'] = tokenizer.tokenizer(case['sentence'])


            message = self.save_json(query, Path(self.DS_PATH / self.CLEAN_QUERY), "Clean query")
            print(colored(message, "yellow"))


            return data, query, uniques
        else:
            data, message = self.load_json(Path(self.DS_PATH / self.CLEAN_DS_FILE), "dataset.json")

            self.word_map, message = self.load_json(Path(self.DS_PATH / self.WORD_MAP), "word_map.json") 
            print(colored(message, "blue"))

            unique_tokens, message = self.load_json(Path(self.DS_PATH / self.UNIQUE_TOKENS), "unique tokens.json") 
            print(colored(message, "blue"))


            self.tfidf_vectorizer =  TFIDFVectorizer(self.word_map)
            self.tfidf_vectorizer.fit(data)
            print(self.word_map)

            query_data, message = self.load_json(Path(self.DS_PATH/self.CLEAN_QUERY), "clean_query.json")
            print(colored(message, "blue"))

            logger.info("Script completed.")
            print(unique_tokens)
            return data, query_data, unique_tokens
        
    
    def calculate_tfidf(self, data):
        tfidf_results , tf_results= self.tfidf_vectorizer.fit_transform(data, 'documents')
        top = {}
        most = {}
        for key, value in tqdm(tfidf_results.items()):
            data[key]['vectors'] = value
            sum_vec = {}
            for sentence in value:
                sum_vec = self.sum_dicts([sum_vec, sentence])
            data[key]['tf_idf'] = sum_vec
            sorted_tfidf = dict(sorted(sum_vec.items(), key=lambda item: item[1], reverse=True))
            tfidf_list = list(sorted_tfidf)
            word_list = list(self.word_map)
            top[key]= [word_list[k] for k in tfidf_list[:5]]
        
        for key, value in tqdm(tf_results.items()):
            sum_vec = {}
            for sentence in value.values():
                sum_vec = self.sum_dicts([sum_vec, sentence])
            sorted_tf = dict(sorted(sum_vec.items(), key=lambda item: item[1], reverse=True))
            most[key] = list(sorted_tf.keys())
            

        
        return data, top, most
            

    def calculate_tfidf2(self, query_data ):

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
            for word in words:
                seq.set_seqs(token, word)
                similarity_ratio = seq.ratio()*100
                if similarity_ratio > max_val:
                    target = word
                    max_val = similarity_ratio
            if similarity_ratio > threshold:
                similar_tokens.append(target)

        return similar_tokens

    def convert_sets_to_lists(self, obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_sets_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_sets_to_lists(item) for item in obj]
        else:
            return obj