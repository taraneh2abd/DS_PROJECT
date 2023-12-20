import json
from pathlib import Path
import pandas as pd
from word_tokenizer import tokenizer 
from termcolor import colored
import pickle
from tf_idf import calculate_tfidf


DS_PATH = Path("dataset")
CLEAN_DS_FILE = Path("clean_data.csv")
UNIQUE_TOKENS_FILE = Path("unique_tokens.txt")

def count_dict(sentences, word_set):
    count_dict = {}
    for word in word_set:
        count_dict[word] = 0
    for sent in sentences:
        for word in sent:
            count_dict[word] += 1
    return count_dict

def dataset_format_converter(CONTENT_FILE_PATTERN, CONTENT_DIR):
    data_list = []
    unique_tokens = set()
    index = 0
    
    word_map = {}
    if not Path(CLEAN_DS_FILE).is_file() or not unique_tokens:
        while Path(CONTENT_DIR /CONTENT_FILE_PATTERN.format(index = index)).is_file():
            sentences = []
            filename = CONTENT_FILE_PATTERN.format(index = index)
            file_path = CONTENT_DIR / CONTENT_FILE_PATTERN.format(index=index)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                for line in file:
                    sentences.append(tokenizer(line.strip()))
            
            tokens = set(tokenizer(content))
            
            unique_tokens |= tokens

            data_list.append({'Filename': filename, 'unique Tokens of document': tokens, 'Content': content,
                            'sentences': sentences, 'count_dict': count_dict(sentences, tokens), 'vectors': None,'tf-idf': None})
            index += 1

        df = pd.DataFrame(data_list)
        df.to_csv(Path(DS_PATH / CLEAN_DS_FILE))
        print(colored(f"Clean data seved to dataset/clean_data.csv", "yellow"))
        with open(Path(DS_PATH / UNIQUE_TOKENS_FILE),'wb') as f:
            pickle.dump(unique_tokens, f)
        print(colored(f"Unique tokens seved to dataset/unique_tokens.txt", "yellow"))
        for i, word in enumerate(unique_tokens):
            word_map[word] = i

        df = calculate_tfidf(df, unique_tokens, word_map)
    else:
        df = pd.read_csv(Path(DS_PATH / CLEAN_DS_FILE))
        with open(Path(DS_PATH / UNIQUE_TOKENS_FILE),'rb') as f:
             unique_tokens = pickle.load(f)
        
        for i, word in enumerate(unique_tokens):
            word_map[word] = i
        
        print(colored(f"Clean data loaded from dataset/clean_data.csv", "blue"))
        print(colored(f"Unique tokens loaded from dataset/unique_tokens.txt", "blue"))
    return df , unique_tokens , word_map