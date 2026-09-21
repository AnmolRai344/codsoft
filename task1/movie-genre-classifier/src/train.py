import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

from preprocess import clean_text

train_path = "../data/train_data.txt"
df = pd.read_csv(train_path, sep=":::", engine="python",
                  names=["ID", "TITLE", "GENRE", "DESCRIPTION"])

df["GENRE"] = df["GENRE"].str.strip()
df["DESCRIPTION"] = df["DESCRIPTION"].astype(str)

print("Cleaning text...")
df["clean_plot"] = df["DESCRIPTION"].apply(clean_text)

X = df["clean_plot"]
y = df["GENRE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Vectorizing...")
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

results = {}
best_model = None
best_acc = 0
best_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")

joblib.dump(best_model, "genre_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Saved model and vectorizer.")