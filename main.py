import json
from pathlib import Path
import pandas as pd
from word_tokenizer import tokenizer 
from dataset_generator import dataset_format_converter
from tf_idf import calculate_tfidf
from search import search

JSON_FILE = Path("dataset") / "data.json"
CONTENT_FILE_PATTERN = "document_{index}.txt"
CONTENT_DIR = Path("dataset/data")

query = json.loads(open(JSON_FILE, encoding="UTF8").read())

df , unique_tokens = dataset_format_converter(CONTENT_FILE_PATTERN, CONTENT_DIR)

word_map = {}
for i, word in enumerate(unique_tokens):
    word_map[word] = i
df = calculate_tfidf(df, unique_tokens, word_map)

for i, case in enumerate(query):
    print("*" * 45)
    print(f"Searching in document No: {case['document_id']}")
    indx, sent_id, max_val = search(case['query'], case['candidate_documents_id'], df, unique_tokens, word_map)
    print(f"Document_{indx} has the most similarity with query No.{i} with {max_val*100}% similarity")

    

    