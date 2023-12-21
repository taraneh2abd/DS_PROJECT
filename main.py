import json
from pathlib import Path
from search import search
from termcolor import colored
from dataset_generator import DatasetGenerator


JSON_FILE = Path("dataset") / "data.json"
CONTENT_FILE_PATTERN = "document_{index}.txt"
CONTENT_DIR = Path("dataset/data")

print(colored("!!!Generating cleaned dataset!!!", "red"))

dataset_generator = DatasetGenerator()

df , query , unique_tokens= dataset_generator.dataset_format_converter(CONTENT_FILE_PATTERN, CONTENT_DIR)
rejected = 0
passed = 0
for i, case in enumerate(query):
    print("*" * 45)
    print(f"Searching in documents:")
    indx, sent_id, max_val, max_val_sent = search(case, case['candidate_documents_id'], df, unique_tokens)
    if indx == case['document_id']:
        passed += 1
        color = "green"
    else: 
        color = "red"
        rejected += 1
    if  case["is_selected"][sent_id]:
        color2 = "green"
    else: 
        color2 = "red"
    print(colored(f"Document_{indx} has the most similarity with query No.{i} with {max_val*100}% similarity", color))
    print(colored(f"The sentence with the highest similarity to the query No.{i} is sentence number {sent_id} with {max_val_sent*100}% similarity.",color2))
print(f"The search engine yielded a success rate of {passed} out of {passed + rejected} queries.")

    

    