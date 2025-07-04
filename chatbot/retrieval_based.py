from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

df = pd.read_csv("data/qa_pairs.csv")
questions = df["Question"].tolist()
answers = df["Answer"].tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# chatbot/retrieval_based.py
def retrieval_bot(user_input):
    user_vector = vectorizer.transform([user_input])  # Transform user input
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    
    best_score = similarities.max()
    best_index = similarities.argmax()

    # Optional threshold
    if best_score > 0.6:
        return df["Answer"][best_index]
    else:
        return None


 
