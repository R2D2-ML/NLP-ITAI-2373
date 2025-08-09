import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from src.utils.global_ import COUNT_VECTORIZER
from src.utils.global_ import DF
# from src.data_processing.text_preprocessor import preprocess_text

# df = pd.read_csv("/Users/gamemaster/Documents/NLP/News_Analysis_ChatBot/data/raw/BBC News Train.csv")

# df['text_processed'] = df['Text'].apply(preprocess_text)

count_matrix = COUNT_VECTORIZER.fit_transform(DF['text_processed'])

# Define the number of topics
n_topics = 10  # You can experiment with different numbers of topics

# Initialize LDA model
LDA = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method='online' # 'online' is faster for large datasets
)

# Fit LDA model to the document-term matrix
print(f"🧠 Training LDA model with {n_topics} topics...")
LDA.fit(count_matrix)