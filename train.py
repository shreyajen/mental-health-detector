import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv("Combined Data.csv")

# ── 2. Clean data ─────────────────────────────────────────
df = df.drop(columns=["Unnamed: 0"])
df = df.dropna(subset=["statement"])
print(f"Rows after cleaning: {len(df)}")

# ── 3. Split input and output ─────────────────────────────
X = df["statement"]   # text posts
y = df["status"]      # labels

# ── 4. Split into train and test sets ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ── 5. Convert text to numbers using TF-IDF ───────────────
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── 6. Train the model ────────────────────────────────────
print("Training model... (may take a minute)")
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# ── 7. Evaluate ───────────────────────────────────────────
y_pred = model.predict(X_test_vec)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# ── 8. Confusion matrix ───────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", 
            xticklabels=model.classes_, 
            yticklabels=model.classes_,
            cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as confusion_matrix.png")

# ── 9. Save model and vectorizer ──────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")
print("\nDone!")