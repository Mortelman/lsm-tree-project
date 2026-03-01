import re
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

class TextPreprocessor:

    def __init__(self, language: str = 'russian'):
        self.language = language
        self.morph = pymorphy3.MorphAnalyzer()

        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words(language))
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def tokenize(self, text: str) -> List[str]:

        text = text.lower()
        tokens = word_tokenize(text, language=self.language)
        
        tokens = [token for token in tokens if token.isalpha()]
        
        return tokens
    
    def lemmatize(self, word: str) -> str:
        parsed = self.morph.parse(word)[0]
        return parsed.normal_form
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]
    
    def preprocess(self, text: str, remove_stopwords: bool = True) -> List[str]:

        tokens = self.tokenize(text)
        
        tokens = [self.lemmatize(token) for token in tokens]
        
        if remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        return tokens
    
    def add_stop_words(self, words: Set[str]) -> None:
        self.stop_words.update(words)
    
    def remove_custom_stop_words(self, words: Set[str]) -> None:
        self.stop_words.difference_update(words)