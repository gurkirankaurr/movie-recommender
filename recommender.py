import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load Data
df = pd.read_csv("movies.csv")
df['genres'] = df['genres'].fillna('')
df['content'] = df['title'].str.lower() + " " + df['genres'].str.replace('|', ' ')

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['content'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(movie_title, top_n=5):
    movie_title = movie_title.lower()
    if movie_title not in df['title'].str.lower().values:
        print("Movie not found! Please try again.\n")
        return
    
    idx = df[df['title'].str.lower() == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    print("\nTop Recommendations:")
    for i, (movie_index, score) in enumerate(sim_scores, start=1):
        print(f"{i}. {df.loc[movie_index, 'title']}  —  {df.loc[movie_index, 'genres']}")
    print()

print("🎬 Movie Recommender System Ready!")
print("Type a movie name exactly from the list, e.g., Toy Story (1995)")
print("Type 'exit' to quit.\n")

while True:
    movie = input("Enter movie title: ")
    if movie.lower() == "exit":
        break
    recommend(movie)
