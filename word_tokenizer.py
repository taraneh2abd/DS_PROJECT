import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


stopwords = set(stopwords.words("english"))

def remove_punctuation(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

def tokenizer(sentence):
    lemmatizer = WordNetLemmatizer()
    lemmatizer = WordNetLemmatizer()
    sentence = remove_punctuation(sentence)
    tokens = word_tokenize(sentence.lower())  
    tokens = set(lemmatizer.lemmatize(token) for token in tokens if token not in stopwords)
    return list(tokens)