import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
import joblib
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words=set(stopwords.words("english"))
stemmer=PorterStemmer()
class TextPreprocessor():
    def __init__(self):
        self.vectorizer=TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2,max_df=0.8)
    def lowercase(self,text):
        return text.lower()
    def remove_url(self,text):
        return re.sub(r"http\S+|www\S+","",text)
    def remove_html(self,text):
        return BeautifulSoup(text,"html.parser").get_text()
    def remove_punctuation(self,text):
        return re.sub(r"[^\w\s]","",text)
    def remove_stopwords(self,text):
        words=word_tokenize(text)
        words=[word for word in words if word.lower()not in stop_words]
        return " ".join(words)
    def stemming(self,text):
        words=word_tokenize(text)
        words=[stemmer.stem(word) for word in words]
        return " ".join(words)
    def preprocess(self,text):
        text=self.lowercase(text)
        text=self.remove_url(text)
        text=self.remove_html(text)
        text=self.remove_punctuation(text)
        text=self.remove_stopwords(text)
        text=self.stemming(text)
        return text
    def fit_transform(self,texts):
        texts=[self.preprocess(text) for text in texts]
        return self.vectorizer.fit_transform(texts)
    def transform(self,texts):
        texts=[self.preprocess(text) for text in texts]
        return self.vectorizer.transform(texts)