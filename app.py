import streamlit as st
import pandas as pd
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# ---------------------------
TMDB_API_KEY = "945e4d253cca3f197069d2694ee26a9b"
# ---------------------------

# Load dataset
movies = pd.read_csv("movies.csv")
movies["combined"] = movies["title"] + " " + movies["genres"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(movies["combined"])
similarity = cosine_similarity(vectors)

# ---------------- API Functions ----------------
@st.cache_data(show_spinner=False)
def get_description(movie_title):
    clean_title = re.sub(r"\s*\(\d{4}\)$", "", movie_title)
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean_title}"
        r = requests.get(url, timeout=5).json()
        if r.get("results"):
            movie_id = r["results"][0]["id"]
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
            details = requests.get(details_url, timeout=5).json()
            description = details.get("overview", "Description not available.")
            poster_path = details.get("poster_path", None)
            genres = ", ".join([g['name'] for g in details.get("genres", [])])
            return description, poster_path, genres, movie_id
        return "Description not available.", None, "", None
    except:
        return "Description not available (API limit reached). Try later.", None, "", None

@st.cache_data(show_spinner=False)
def get_trailer_url(movie_id):
    if not movie_id:
        return None
    try:
        videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        videos = requests.get(videos_url, timeout=5).json().get("results", [])
        for v in videos:
            if v["type"] == "Trailer" and v["site"] == "YouTube":
                return f"https://www.youtube.com/watch?v={v['key']}"
        return None
    except:
        return None

def recommend(movie_name, top_n=6):
    try:
        idx = movies[movies["title"] == movie_name].index[0]
    except IndexError:
        return []
    distances = list(enumerate(similarity[idx]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [movies.iloc[i[0]].title for i in sorted_movies]

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# CSS Styling
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                      url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
.movie-card {
    border-radius: 15px;
    box-shadow: 2px 4px 12px rgba(0,0,0,0.5);
    padding: 10px;
    margin-bottom: 20px;
    background-color: rgba(0,0,0,0.6);
    transition: transform 0.2s;
}
.movie-card:hover {
    transform: scale(1.03);
}
.stTextInput > div > input {
    background-color: rgba(255,255,255,0.9);
    color: black;
}
button {
    background-color:#2a9d8f;
    color:white;
    font-size:22px;
    padding:15px 40px;
    border-radius:12px;
    border:none;
    cursor:pointer;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.session_state.page

# ---------------- HOME PAGE ----------------
if page == "Home":
    # Hero section
    st.markdown("""
    <div style="
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        height: 50vh;
        color: white;
    ">
        <h1 style='font-size:60px; margin-bottom:20px;'>🎬 Movie Recommender</h1>
        <p style='font-size:22px; margin-bottom:10px;'>Discover movies similar to your favorites!</p>
    </div>
    """, unsafe_allow_html=True)

    # Properly centered Start Exploring button using columns
    col1, col2, col3 = st.columns([2.3,2,1])
    with col2:
        if st.button("Start Exploring"):
            st.session_state.page = "Recommendations"

# ---------------- RECOMMENDATIONS PAGE ----------------
elif page == "Recommendations":
    st.markdown("<h2 style='text-align:center;'>Find Your Next Movie</h2>", unsafe_allow_html=True)

    top_n = st.slider("Number of Recommendations", 3, 9, 6)
    random_movie_btn = st.button("🎲 Random Movie Suggestion")

    search_movie = st.text_input("Start typing a movie title:")
    matches = movies[movies['title'].str.contains(search_movie, case=False, na=False)]
    selected_movie = st.selectbox("Select a Movie:", matches["title"].values) if not matches.empty else None

    if random_movie_btn:
        selected_movie = random.choice(movies["title"].values)
        st.info(f"Randomly selected movie: **{selected_movie}**")

    if selected_movie and st.button("Get Recommendations"):
        recommendations = recommend(selected_movie, top_n=top_n)
        if not recommendations:
            st.warning("Movie not found in dataset. Try another title.")
        else:
            st.subheader("Recommended Movies:")
            cols = st.columns(3)
            for idx, movie in enumerate(recommendations):
                description, poster_path, genres, movie_id = get_description(movie)
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Image"
                tmdb_url = f"https://www.themoviedb.org/movie/{movie_id}" if movie_id else f"https://www.themoviedb.org/search?query={movie.replace(' ','+')}"
                trailer_url = get_trailer_url(movie_id)

                col = cols[idx % 3]
                with col:
                    st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"### {movie}")
                    if genres:
                        st.markdown(f"**Genres:** {genres}")
                    if description:
                        st.write(description[:120] + "..." if len(description) > 120 else description)
                        with st.expander("Read More"):
                            st.write(description)
                    st.markdown(f"[🔗 More Info]({tmdb_url})", unsafe_allow_html=True)
                    if trailer_url:
                        st.markdown(f"[🎬 Watch Trailer]({trailer_url})", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("About This App"):
        st.session_state.page = "About"

# ---------------- ABOUT PAGE ----------------
elif page == "About":
    st.markdown("<h2 style='text-align:center;'>About This App</h2>", unsafe_allow_html=True)
    st.markdown("""
    - Built with **Python**, **Streamlit**, and **scikit-learn**.  
    - Uses **TMDb API** to fetch movie posters, descriptions, and genres.  
    - Provides movie recommendations based on **cosine similarity** of genres and titles.  
    - Developed by **Gurkiran Kaur**.
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>GitHub: <a href='https://github.com/gurkirankaurr'>https://github.com/gurkirankaurr</a></p>", unsafe_allow_html=True)

    if st.button("Back to Home"):
        st.session_state.page = "Home"

# ---------------- FOOTER ----------------
st.markdown("<p style='text-align:center; color:#ccc; margin-top:50px;'>© 2025 Gurkiran Kaur</p>", unsafe_allow_html=True)
