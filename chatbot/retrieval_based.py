import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/qa_pairs.csv")  # Should have columns: Question, Answer

questions = df['Question'].tolist()
answers = df['Answer'].tolist()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def retrieval_bot(user_input):
    input_vec = vectorizer.transform([user_input])
    scores = cosine_similarity(input_vec, X)
    max_score = scores.max()
    if max_score < 0.3:
        return None  # No close match found
    index = scores.argmax()
    return answers[index]
 
