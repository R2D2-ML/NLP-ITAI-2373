import os
import platform
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from src.data_processing.text_preprocessor import preprocess_text

NLP = spacy.load('en_core_web_sm')

DF = pd.read_csv("/Users/gamemaster/Documents/NLP/News_Analysis_ChatBot/data/raw/BBC News Train.csv")

DF['text_processed'] = DF['Text'].apply(preprocess_text)

TFIDF_VECTORIZER = TfidfVectorizer(
        max_features=5000,  # Limit vocabulary for computational efficiency
        ngram_range=(1, 2),  # Include unigrams and bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.8  # Ignore terms that appear in more than 80% of documents
    )

tfidf_matrix = TFIDF_VECTORIZER.fit_transform(DF['text_processed'])

COUNT_VECTORIZER = CountVectorizer(
                max_features=5000,  # Limit vocabulary size (same as TF-IDF for consistency)
                min_df=2,           # Ignore terms that appear in less than 2 documents
                max_df=0.8          # Ignore terms that appear in more than 80% of documents
            )

def clear_terminal():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')