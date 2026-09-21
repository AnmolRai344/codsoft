import joblib
from preprocess import clean_text

model = joblib.load("genre_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def predict_genre(plot_summary: str) -> str:
    cleaned = clean_text(plot_summary)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

if __name__ == "__main__":
    sample_plot = """
    A group of astronauts travel through a wormhole in search of a
    new habitable planet for humanity as Earth becomes uninhabitable.
    """

    genre = predict_genre(sample_plot)
    print(f"Predicted genre: {genre}")