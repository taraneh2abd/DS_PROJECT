import json
from pathlib import path
import pandas as pd
from word_tokenizer import tokenizer 

def dataset_format_converter(CONTENT_FILE_PATTERN, CONTENT_DIR):
    data_list = []
    unique_tokens_in_all = set()
    index = 0
    while path.isfile(CONTENT_FILE_PATTERN.format(index = index)):
        sentences = []
        filename = CONTENT_FILE_PATTERN.format(index = index)
        file_path = CONTENT_DIR / CONTENT_FILE_PATTERN.format(index=index)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            for line in file:
                sentences.append(set(tokenizer(line.strip())))

        tokens = tokenizer(content)

        unique_tokens_in_all |= set(tokens) 

        data_list.append({'Filename': filename, 'unique Tokens of document': tokens, 'Content': content,
                        'unique Tokens of sentence': sentences})
        index += 1

    df = pd.DataFrame(data_list)

    return df , unique_tokens_in_all 