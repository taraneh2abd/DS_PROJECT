import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class WordTokenizer:
    
    def __init__(self, remove_punct = True, lemm = True, _stopwords = True):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self.stopwords = set(stopwords.words("english"))
        self.remove_punct = remove_punct
        self.lemmatize = lemm
        self._remove_stopwords = _stopwords
        

    def remove_punctuation(self, text):
        # text = re.sub(r'\d+', '', text)
        cleaned_text = re.sub(r'[^\w\s]', '', text)
        return cleaned_text

    def tokenizer(self, sentence):
        if self.remove_punct : 
            sentence = self.remove_punctuation(sentence)
        tokens = word_tokenize(sentence.lower()) 

        if self._remove_stopwords:
            tokens =self.remove_stopwords(tokens)  

        if self.lemmatize :
            tokens = self.lemmatizer(tokens)

        return tokens
    
    def remove_stopwords(self, tokens):

        return list(token for token in tokens if token not in self.stopwords and len(token) > 1)
    
    def lemmatizer(self, tokens):
        
        lemmatizer = WordNetLemmatizer()
        tokens = list(lemmatizer.lemmatize(token) for token in tokens)
        return tokens
