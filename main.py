import json
from pathlib import path
import pandas as pd
from word_tokenizer import tokenizer 
from dataset_generator import dataset_format_converter
from tf_idf import calculate_tfidf

JSON_FILE = path("dataset") / "data.json"
CONTENT_FILE_PATTERN = "document_{index}.txt"
CONTENT_DIR = path("dataset/data")

query = json.loads(open(JSON_FILE, encoding="UTF8").read())

df , unique_tokens_in_all = dataset_format_converter(CONTENT_FILE_PATTERN, CONTENT_DIR)

word_map = {}
for i, word in enumerate(unique_tokens_in_all):
    word_map[word] = i
df = calculate_tfidf(df, unique_tokens_in_all, word_map)