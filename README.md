# 🧠 Mental Health Text Analyzer

A machine learning web app that detects mental health signals in text using NLP.
Trained on 52,000+ real Reddit posts across 7 mental health categories.

🔗 **Live Demo:** https://mental-health-detector-qwerty.streamlit.app/

---

## What it does

Type or paste any text → the model predicts which mental health category
it most closely resembles, with a confidence score for each category.

**Categories:** Normal · Depression · Anxiety · Stress · Bipolar · Suicidal · Personality Disorder

---

## How it works

```
Raw text → TF-IDF Vectorization → Logistic Regression → Prediction + Confidence scores
```

1. Text is cleaned and vectorized using TF-IDF (top 10,000 features)
2. A Logistic Regression model trained on 42,144 samples predicts the category
3. Confidence scores are shown for all 7 categories

---

## Model Performance

| Metric           | Score |
| ---------------- | ----- |
| Overall Accuracy | 76%   |
| Best F1 (Normal) | 0.89  |
| Macro F1 Average | 0.72  |

> Note: Stress and Personality Disorder scored lower due to fewer training samples —
> a real-world class imbalance problem handled using `class_weight="balanced"`.

---

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (Logistic Regression, TF-IDF)
- **Frontend:** Streamlit
- **Data:** Reddit Mental Health Dataset (Kaggle, 52,000+ posts)
- **Deployment:** Streamlit Cloud

---

## Run locally

```bash
git clone https://github.com/shreyajen/mental-health-detector.git
cd mental-health-detector
pip install -r requirements.txt
python train.py        # trains and saves the model
streamlit run app.py   # launches the app
```

---

## Project structure

```
mental-health-detector/
├── app.py               # Streamlit frontend
├── train.py             # Model training + evaluation
├── explore.py           # Data exploration
├── model.pkl            # Trained model
├── vectorizer.pkl       # Saved TF-IDF vectorizer
├── confusion_matrix.png # Evaluation visualization
└── requirements.txt
```

---

## Disclaimer

This is an educational ML project — not a medical tool.
Please seek professional help if you are experiencing mental health issues.
