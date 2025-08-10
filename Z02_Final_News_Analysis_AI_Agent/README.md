# 📰 News Analysis ChatBot

Welcome to the **News Analysis ChatBot**, an intelligent agent designed to analyze news articles, extract insights, and provide actionable recommendations. This project combines the power of natural language processing (NLP), machine learning, and conversational AI to deliver a seamless news analysis experience.

---

# Must Download
```bash
!python -m spacy download en_core_web_sm
```

## 🚀 Features

- **News Categorization**: Automatically classify news articles into categories like Politics, Business, Sports, and more.
- **Sentiment Analysis**: Understand the emotional tone of articles (positive, neutral, or negative).
- **Entity Extraction**: Identify key entities such as people, organizations, and locations.
- **Topic Modeling**: Discover dominant topics within articles using Latent Dirichlet Allocation (LDA).
- **Interactive Workflow**: Engage with the chatbot to analyze articles and explore insights interactively.
- **Search Integration**: Fetch and summarize news articles from the web using Google Search.

---

## 🛠️ Project Structure

```

```



---

## 🧠 How It Works

1. **Preprocessing**: Articles are cleaned, tokenized, and lemmatized using the `text_preprocessor`.
2. **Classification**: Articles are categorized using a pre-trained classifier.
3. **Entity Extraction**: Named entities are identified using spaCy.
4. **Sentiment Analysis**: Sentiment scores are calculated using VADER.
5. **Topic Modeling**: Dominant topics are extracted using LDA.
6. **Interactive Insights**: Users can interact with the chatbot to explore insights and recommendations.

---

## 🛑 Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt

- Download the spaCy language model:
  ```bash
  python -m spacy download en_core_web_sm

---

## ▶️ Getting Started

1. Clone the repository:
  ```bash
  git clone https://github.com/your-repo/news-analysis-chatbot.git
  cd news-analysis-chatbot
  ```
2. Set up your .env file with API keys:
  ```bash
  GOOGLE_API_KEY=your-google-api-key
  GOOGLE_CSE_ID=your-google-cse-id
  OPENAI_API_KEY=your-openai-api-key
  ```
3. Run the Agent:
  ```bash
  python main.py
  ```
4. Interact with the chatbot:

- What can I do for you?: find the latest news on AI advancements.

---

📊 Example Insights
- Predicted Category: 🏷️ Technology (95% confidence)
- Sentiment: 😊 Positive (compound score: 0.85)
- Entities: 🔍 Found 3 entities (e.g., "OpenAI" as ORG, "GPT-4" as PRODUCT)
- Dominant Topic: 📊 AI Research (80% confidence)
- Recommendations: ✅ High confidence classification, Positive sentiment detected.

---

🛠️ Technologies Used
- Natural Language Processing: spaCy, NLTK
- Machine Learning: scikit-learn, joblib
- Visualization: matplotlib, seaborn, wordcloud
- Web Integration: Google Search API, LangChain
- Data Handling: pandas
- Prompt Engineering: OpenAI GPT

---

🌟 Acknowledgments
spaCy for NLP capabilities.
NLTK for sentiment analysis.
scikit-learn for machine learning utilities.
LangChain for LLM-based workflows.

---

📧 Contact
For questions or feedback, reach out to fullstackvon@gmail.com.

Happy analyzing! 🎉

---

