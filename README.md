# Fake News Detector

Automatically classifies news articles as *fake* or *real* using Machine Learning and NLP.

This is a work-in-progress project to build a robust Fake News Detection system using real datasets, feature engineering, and classical ML models.

---

## 📰 Problem Statement

Fake news spreads rapidly online and can drastically affect public opinion and decision-making. This project aims to **detect fake news articles** based on their textual content using machine learning techniques.

---

## 🛠 Approach

### 📌 Data
- Splitting the dataset into `train.csv`, `valid.csv`, and `test.csv`  
- Labels encoded for binary classification

### 📌 Text Preprocessing
- Lowercasing
- Removing noise (URLs, special characters)
- Tokenization + lemmatization
- Vectorization using TF-IDF

### 📌 Models Tried
| Model | Status | Notes |
|-------|--------|-------|
| Logistic Regression | Implemented | ~23% validation accuracy so far |
| Naive Bayes | Implemented | ~23% validation accuracy so far |
| Others planned | Pending | Want to try SVM, Random Forest, Transformers |

> Current accuracy indicates the model is underfitting or feature representation needs improvement.

---

## 📦 Code Structure

| File | Description |
|------|-------------|
| `fake-news-detector.ipynb` | Complete notebook with data prep, models, evaluation |
| `app.py` | Flask app to serve model predictions |
| `best_model.pkl` | Pickled trained ML model |
| `tfidf_vectorizer.pkl` | Vectorizer used for converting text to features |
| `label_encoder.pkl` | Encodes target labels |
| `train.csv`, `valid.csv`, `Test.csv` | Dataset splits |

---

## 🚧 Challenges & Current Status

- **Low accuracy (~23%) with baseline models**
- Fixing TF-IDF consistency across train/valid/test feature spaces
- Next steps: Model tuning, feature improvements
- Plan to explore algorithms better suited for text (SVM, deep learning)

---

## 🚀 Future Work

- Better feature extraction (word embeddings / transformers)
- Hyperparameter optimization
- Deployment improvements
- User interface enhancements

---

## 📌 How to Run

1. Clone the repo  
   `git clone https://github.com/kavya05-cell/Fake-News-Detector.git`
2. Install dependencies  
   `pip install -r requirements.txt`
3. Run Jupyter Notebook or Flask app  
   `python app.py`

---

## 🧠 Notes

This project is being developed in **public progress** — updates will be added regularly as models improve.

---

## 🏷 Topics

Fake news detection, NLP, Machine Learning, TF-IDF, Text Classification
