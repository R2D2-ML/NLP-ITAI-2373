
# Module 7: Sentiment and Emotion Analysis Lab 📚 

## ITAI 2373 - Natural Language Processing

## Lab Overview

Welcome to the Sentiment and Emotion Analysis lab! In this hands-on session, I learned how to build a complete emotion detection system that works with both text and speech. This lab connects directly to previous Module concepts and gives me practical experience with real-world emotion analysis.

## Setup and Installation

To run this notebook, you will need to install the following libraries:
```bash
!pip install vaderSentiment textblob librosa soundfile
!python -m textblob.download_corpora
```

### What was built:

1.  **Text Sentiment Analyzer** using VADER and TextBlob
2.  **Machine Learning Classifier** with scikit-learn
3.  **Speech Emotion Detector** using audio features
4.  **Multimodal System** combining text and speech analysis

## Learning Objectives

By the end of this lab I was able to:

-   Understand the differences between rule-based and ML approaches to sentiment analysis
-   Build and evaluate multiple sentiment analysis systems
-   Extract and analyze emotional features from speech
-   Create a multimodal emotion detection system
-   Critically evaluate bias and fairness in emotion detection systems

## Datasets Used

*   **Part 1 (Text Sentiment):** Sample data with linguistic challenges (created in notebook)
*   **Part 2 (ML Text Sentiment):** Illustrative text dataset (created in notebook)
*   **Part 3 (Speech Emotion):** Simulated emotional audio samples (created in notebook)
*   **Part 4 (Multimodal):** Multimodal dataset combining text and simulated audio features (created in notebook)

## Lab Structure

This lab is divided into the following parts:

*   **Part 0: Setup and Installation:** Install required libraries (vaderSentiment, textblob, librosa, soundfile)
*   **Part 1: Text Sentiment Analysis:** Rule-based methods (VADER, TextBlob), analysis and comparison
*   **Part 2: Machine Learning Approach:** TF-IDF feature extraction, Logistic Regression, RandomForestClassifier, model evaluation and interpretation
*   **Part 3: Speech Emotion Detection:** Simulating emotional audio, extracting audio features (pitch, energy, tempo, spectral centroid) using librosa, visualizing features, training and evaluating an audio classifier
*   **Part 4: Multimodal Emotion Analysis:** Creating multimodal data, fusing text (TF-IDF) and audio features (scaling), training and comparing Text Only, Audio Only, and Multimodal (Fused) models

## Connections to Previous Modules

This lab builds upon concepts from previous modules:

*   **Module 1 (Introduction to NLP):** Applying NLP to understand emotions, real-world applications.
*   **Module 2 (Text Preprocessing):** Using tokenization, normalization, cleaning techniques for text analysis.
*   **Module 3 (Audio and Preprocessing):** Extracting emotional features from speech signals, combining text and audio.
*   **Module 4 (Text Representation):** Using TF-IDF for ML, comparing rule-based vs. ML features.
*   **Module 5 (Part-of-Speech Tagging):** Identifying emotional intensity words.
*   **Module 6 (Syntax and Parsing):** Understanding emotional targets, sentiment scope, and negation.

## Conclusion and Key Takeaways

- Integration of concepts from all modules is powerful.
- Different approaches (rule-based, ML, multimodal) have different strengths.
- Context (including speech cues) matters in emotion detection.
- Bias is a significant ethical risk and must be addressed.
- Real-world deployment requires balancing factors like accuracy, complexity, and cost.
