import json
from pathlib import Path
import pandas as pd
from word_tokenizer import tokenizer 

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
    unique_tokens_in_all = set()
    index = 0
    while Path(CONTENT_DIR /CONTENT_FILE_PATTERN.format(index = index)).is_file():
        sentences = []
        filename = CONTENT_FILE_PATTERN.format(index = index)
        file_path = CONTENT_DIR / CONTENT_FILE_PATTERN.format(index=index)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            for line in file:
                sentences.append(tokenizer(line.strip()))
        
        tokens = set(tokenizer(content))
        
        unique_tokens_in_all |= tokens

        data_list.append({'Filename': filename, 'unique Tokens of document': tokens, 'Content': content,
                        'sentences': sentences, 'count_dict': count_dict(sentences, tokens), 'vectors': None,'tf-idf': None})
        index += 1

    df = pd.DataFrame(data_list)

    return df , unique_tokens_in_all 