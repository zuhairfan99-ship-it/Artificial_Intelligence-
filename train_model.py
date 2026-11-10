"""Train and evaluate the email spam classifier."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emails.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"

def main():
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["text", "label"])
    data["label"] = data["label"].str.lower().str.strip()
    data = data[data["label"].isin(["spam", "ham"])]
    if data["label"].nunique() < 2:
        raise ValueError("Dataset must contain both 'spam' and 'ham' labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"].astype(str), data["label"], test_size=0.20,
        random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), max_features=10000,
                                  strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Dataset rows: {len(data)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Not Spam", "Spam"]))
    print("Confusion matrix (labels: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
