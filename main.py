import json
from pathlib import path
import pandas as pd
from word_tokenizer import tokenizer 

JSON_FILE = path("dataset") / "data.json"
CONTENT_FILE_PATTERN = "document_{index}.txt"
CONTENT_DIR = path("dataset/data")

query = json.loads(open(JSON_FILE, encoding="UTF8").read())

data_list = []
unique_tokens_in_all = set()
index = 0
while path.isfile(CONTENT_FILE_PATTERN.format(index = index)):

    filename = CONTENT_FILE_PATTERN.format(index = index)
    file_path = CONTENT_DIR / CONTENT_FILE_PATTERN.format(index=index)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    tokens = tokenizer(content)

    unique_tokens_in_all |= set(tokens) 

    data_list.append({'Filename': filename, 'unique Tokens of document': tokens, 'Content': content})
    index += 1
