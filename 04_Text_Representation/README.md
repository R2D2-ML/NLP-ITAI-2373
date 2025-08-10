
# 📚 Module 04: Text Representation - From Words to Numbers

## ITAI 2373 - Natural Language Processing

## Overview

This lab explores how computers transform human language into mathematical representations that machines can understand and process. It guided me through a journey from basic word counting to sophisticated embedding techniques used in modern AI systems.

---

## Setup and Installation

To run this notebook, you will need to install the following libraries:
```bash
!pip install vaderSentiment textblob librosa soundfile
!python -m textblob.download_corpora
```
---

## Learning Objectives

By completing this lab, I am able to:

*   Explain why text must be converted to numbers for machine learning
*   Implement Bag of Words and TF-IDF representations from scratch
*   Apply N-gram analysis to capture word sequences
*   Explore word embeddings and their semantic properties
*   Compare different text representation methods
*   Build a simple text classification system

---

## Datasets Used

*   Sample Movie Reviews (a small, custom dataset for demonstration)
*   NLTK Movie Reviews Dataset (for the text classification task)

---

## Lab Structure (5 Parts)

1.  **Part 1-2: Foundations & Sparse Representations**
    *   Why text needs numerical representation
    *   Text preprocessing and tokenization
    *   Bag of Words (BOW) from scratch
    *   Limitations of sparse representations
2.  **Part 3: TF-IDF & N-grams - Weighted Representations**
    *   TF-IDF weighting concept and implementation
    *   N-gram analysis for capturing sequences
    *   Document similarity using cosine similarity
    *   Comparison of methods
3.  **Part 4: Dense Representations - Word Embeddings**
    *   Distributional hypothesis
    *   Pre-trained word embeddings (GloVe)
    *   Semantic relationships and word arithmetic
    *   Sparse vs Dense representations comparison
4.  **Part 5: Integration & Real-World Applications**
    *   Building a text classification system
    *   Comparing representations on a real task
    *   Exploring real-world applications
    *   Ethical considerations

---

## Conclusion and Key Takeaways

- I have a strong foundational understanding of various text representation techniques, their strengths and weaknesses, and how they are applied in real-world NLP tasks.
- I now have practical experience implementing and comparing various methods.
