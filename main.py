import json
from pathlib import Path
from dataset_generator import dataset_format_converter
from search import search
from termcolor import colored


JSON_FILE = Path("dataset") / "data.json"
CONTENT_FILE_PATTERN = "document_{index}.txt"
CONTENT_DIR = Path("dataset/data")

query = json.loads(open(JSON_FILE, encoding="UTF8").read())

print(colored("!!!Generating cleaned dataset!!!", "red"))
df , unique_tokens, word_map = dataset_format_converter(CONTENT_FILE_PATTERN, CONTENT_DIR)


for i, case in enumerate(query):
    print("*" * 45)
    print(f"Searching in document No: {case['document_id']}")
    indx, sent_id, max_val = search(case['query'], case['candidate_documents_id'], df, unique_tokens, word_map)
    print(f"Document_{indx} has the most similarity with query No.{i} with {max_val*100}% similarity")

    

    