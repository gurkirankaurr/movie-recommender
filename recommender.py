import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "945e4d253cca3f197069d2694ee26a9b"   # ← paste your TMDB v3 API key here

movies = pd.read_csv("movies.csv")
movies["combined"] = movies["title"] + " " + movies["genres"]

vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(movies["combined"])
similarity = cosine_similarity(vectors)

def get_description(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        r = requests.get(url, timeout=5).json()

        if r.get("results"):
            movie_id = r["results"][0]["id"]
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
            details = requests.get(details_url, timeout=5).json()
            return details.get("overview", "No description available.")

        return "No description available."

    except:
        return "Description not available (API rate limit). Try again later."

def recommend(movie_name):
    idx = movies[movies["title"] == movie_name].index[0]
    distances = list(enumerate(similarity[idx]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sorted_movies]

print("🎬 Movie Recommender System Ready!")
print("Type 'exit' to quit.\n")

while True:
    movie = input("Enter movie title: ")
    if movie.lower() == "exit":
        break

    if movie not in movies["title"].values:
        print("Movie not found. Try again.\n")
        continue

    recs = recommend(movie)
    print("\nRecommended Movies:\n")
    for m in recs:
        print("•", m)

    print("\nMovie Description:\n", get_description(movie), "\n")
