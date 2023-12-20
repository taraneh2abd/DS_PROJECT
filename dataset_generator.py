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
    count_dict = {token: 0 for token in word_set}  

    for sentence in sentences:
        for word in sentence:
            if word in count_dict:
                count_dict[word] += 1

    return count_dict

def dataset_format_converter(CONTENT_FILE_PATTERN, CONTENT_DIR):
    data_list = []
    unique_tokens = set()
    index = 0
    
    word_map = {}
    if not Path(DS_PATH / CLEAN_DS_FILE).is_file():
        while Path(CONTENT_DIR /CONTENT_FILE_PATTERN.format(index = index)).is_file():
            sentences = []
            tokens = set()
            filename = CONTENT_FILE_PATTERN.format(index = index)
            file_path = CONTENT_DIR / CONTENT_FILE_PATTERN.format(index=index)

            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    sent = tokenizer(line.strip())
                    sentences.append(sent)
                    tokens |= set(sent)
            
            unique_tokens |= tokens

            data_list.append({'Filename': filename, 'unique Tokens of document': tokens,'sentences': sentences, 'count_dict': count_dict(sentences, tokens), 'vectors': None,'tf-idf': None})
            index += 1

        for i, word in enumerate(unique_tokens):
            word_map[word] = i

        df = pd.DataFrame(data_list)

        df.to_csv(Path(DS_PATH / CLEAN_DS_FILE))
        print(colored(f"Clean data seved to dataset/clean_data.csv", "yellow"))


    else:
        df = pd.read_csv(Path(DS_PATH / CLEAN_DS_FILE))

        for index, row in df.iterrows():
            unique_tokens |= set(row['unique Tokens of document'])
        
        for i, word in enumerate(unique_tokens):
            word_map[word] = i
        
        print(colored(f"Clean data loaded from dataset/clean_data.csv", "blue"))
    return df , unique_tokens , word_map