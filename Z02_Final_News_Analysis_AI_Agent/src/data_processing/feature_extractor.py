import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.global_ import TFIDF_VECTORIZER

# Create TF-IDF vectorizer
def create_tfidf(df):
    # tfidf_vectorizer = TfidfVectorizer(
    #     max_features=5000,  # Limit vocabulary for computational efficiency
    #     ngram_range=(1, 2),  # Include unigrams and bigrams
    #     min_df=2,  # Ignore terms that appear in less than 2 documents
    #     max_df=0.8  # Ignore terms that appear in more than 80% of documents
    # )

    # Fit and transform the processed text
    print("🔢 Creating TF-IDF features...")
    tfidf_matrix = TFIDF_VECTORIZER.fit_transform(df['text_processed'])
    feature_names = TFIDF_VECTORIZER.get_feature_names_out()

    print(f"✅ TF-IDF matrix created!")
    print(f"📊 Shape: {tfidf_matrix.shape}")
    print(f"📝 Vocabulary size: {len(feature_names)}")
    print(f"🔢 Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")

    # Convert to DataFrame for easier analysis
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    tfidf_df['Category'] = df['Category'].values

    return tfidf_df