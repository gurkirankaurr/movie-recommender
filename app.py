import streamlit as st
import pandas as pd
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
#  INSERT YOUR TMDB API KEY ↓
# ---------------------------
TMDB_API_KEY = "945e4d253cca3f197069d2694ee26a9b"
# ---------------------------

# Load Dataset
movies = pd.read_csv("movies.csv")
movies["combined"] = movies["title"] + " " + movies["genres"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(movies["combined"])
similarity = cosine_similarity(vectors)

@st.cache_data(show_spinner=False)
def get_description(movie_title):
    # Remove year e.g. "Toy Story (1995)" → "Toy Story"
    clean_title = re.sub(r"\s*\(\d{4}\)$", "", movie_title)

    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean_title}"
        r = requests.get(url, timeout=5).json()

        if r.get("results"):
            movie_id = r["results"][0]["id"]
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
            details = requests.get(details_url, timeout=5).json()
            return details.get("overview", "Description not available.")

        return "Description not available."

    except:
        return "Description not available (API limit reached). Try later."

def recommend(movie_name):
    idx = movies[movies["title"] == movie_name].index[0]
    distances = list(enumerate(similarity[idx]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sorted_movies]

# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select a Movie:", movies["title"].values)

if st.button("Get Recommendations"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")

    for movie in recommendations:
        st.write(f"✅ **{movie}**")
        st.caption(get_description(movie))   # show description
        st.write("---")
