import numpy as np
# import joblib
from src.data_processing.text_preprocessor import preprocess_text
from src.utils.global_ import NLP
# from utils.global_ import TFIDF_VECTORIZER
from src.utils.global_ import COUNT_VECTORIZER
# from .topic_modeler import LDA
from .en_extractor import extract_entities
from .sentiment_analyzer import analyze_sentiment

# Load the model
# MODEL = joblib.load('data/models/best_classifier.pkl')



class NewsBotIntelligenceSystem:
    """
    Complete NewsBot Intelligence System

    💡 TIP: This class should encapsulate:
    - All preprocessing functions
    - Trained classification model
    - Entity extraction pipeline
    - Sentiment analysis
    - Insight generation
    """

    def __init__(self, classifier, vectorizer, topic_model=None):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.nlp = NLP  # spaCy model
        self.topic_model = topic_model

        if topic_model:
            self.count_vectorizer = COUNT_VECTORIZER

    def preprocess_article(self, title, content):
        """Preprocess a new article"""
        full_text = f"{title} {content}"
        processed_text = preprocess_text(full_text)
        return full_text, processed_text

    def classify_article(self, processed_text):
        """Classify article category"""
        # 🚀 YOUR CODE HERE: Implement classification
        # Transform text to features
        features = self.vectorizer.transform([processed_text])

        # Add dummy features for sentiment and length (in production, calculate these)
        # In a real application, you would calculate sentiment and length features here
        dummy_features = np.zeros((1, 6))  # 4 sentiment + 2 length features (removed title_length as not used consistently)
        features_combined = np.hstack([features.toarray(), dummy_features])

        # Predict category and probability
        prediction = self.classifier.predict(features_combined)[0]
        probabilities = self.classifier.predict_proba(features_combined)[0]

        # Get class probabilities
        class_probs = dict(zip(self.classifier.classes_, probabilities))

        return prediction, class_probs

    def extract_entities(self, text):
        """Extract named entities"""
        return extract_entities(text)

    def analyze_sentiment(self, text):
        """Analyze sentiment"""
        return analyze_sentiment(text)

    def predict_topic(self, processed_text):
        """Predict dominant topic for a new article"""
        if not self.topic_model or not self.count_vectorizer:
            return None, None

        # Transform text using the fitted count vectorizer
        text_count_matrix = self.count_vectorizer.transform([processed_text])

        # Get topic distribution
        topic_distribution = self.topic_model.transform(text_count_matrix)[0]

        # Find the dominant topic
        dominant_topic_idx = np.argmax(topic_distribution)
        dominant_topic_prob = np.max(topic_distribution)

        return dominant_topic_idx, dominant_topic_prob


    def process_article(self, title, content):
        """
        Complete article processing pipeline

        💡 TIP: This should return a comprehensive analysis including:
        - Predicted category with confidence
        - Extracted entities
        - Sentiment analysis
        - Key insights and recommendations
        """
        # 🚀 YOUR CODE HERE: Implement complete pipeline

        # Step 1: Preprocess
        full_text, processed_text = self.preprocess_article(title, content)

        # Step 2: Classify
        category, category_probs = self.classify_article(processed_text)

        # Step 3: Extract entities
        entities = self.extract_entities(full_text)

        # Step 4: Analyze sentiment
        sentiment = self.analyze_sentiment(full_text)

        # Step 5: Predict topic
        dominant_topic_idx, dominant_topic_prob = self.predict_topic(processed_text)

        # Step 6: Generate insights
        insights = self.generate_insights(category, entities, sentiment, category_probs, dominant_topic_idx, dominant_topic_prob)

        return {
            'title': title,
            'content': content[:200] + '...' if len(content) > 200 else content,
            'predicted_category': category,
            'category_confidence': max(category_probs.values()),
            'category_probabilities': category_probs,
            'dominant_topic': dominant_topic_idx,
            'dominant_topic_confidence': dominant_topic_prob,
            'entities': entities,
            'sentiment': sentiment,
            'insights': insights
        }

    def generate_insights(self, category, entities, sentiment, category_probs, dominant_topic_idx, dominant_topic_prob):
        """Generate actionable insights"""
        insights = []
        topic_labels = {0: "Entertainment", 1: "Politics", 2: "Business", 3: "Tech", 4: "Tech", 5: "Tech", 6: "Tech", 7: "Entertainment", 8: "Sports", 9: "Sports" }

        # Classification insights
        confidence = max(category_probs.values())
        if confidence > 0.1:
            insights.append(f"✅ High confidence {category} classification ({confidence:.2%})")
        else:
            insights.append(f"⚠️ Uncertain classification - consider manual review")

        # Sentiment insights
        if sentiment['compound'] > 0.1:
            insights.append(f"😊 Positive sentiment detected ({sentiment['compound']:.3f})")
        elif sentiment['compound'] < -0.1:
            insights.append(f"😞 Negative sentiment detected ({sentiment['compound']:.3f})")
        else:
            insights.append(f"😐 Neutral sentiment ({sentiment['compound']:.3f})")

        # Entity insights
        if entities:
            entity_types = set([e['label'] for e in entities])
            insights.append(f"🔍 Found {len(entities)} entities of {len(entity_types)} types")

            # Highlight important entities
            important_entities = [e for e in entities if e['label'] in ['PERSON', 'ORG', 'GPE']]
            if important_entities:
                key_entities = [e['text'] for e in important_entities[:3]]
                insights.append(f"🎯 Key entities: {', '.join(key_entities)}")
        else:
            insights.append("ℹ️ No named entities detected")

        # Topic insights
        if dominant_topic_idx is not None:
            insights.append(f"📊 Dominant topic: {topic_labels[dominant_topic_idx]} ({dominant_topic_prob:.2%} confidence)")
        else:
            insights.append("ℹ️ No dominant topic detected")


        return insights

# Initialize the complete system
# newsbot = NewsBotIntelligenceSystem(
#     classifier=MODEL,
#     vectorizer=TFIDF_VECTORIZER,
#     sentiment_analyzer=SentimentIntensityAnalyzer(),
#     topic_model=LDA # Pass the trained LDA model
# )
