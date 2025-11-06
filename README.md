# 🎬 Content-Based Movie Recommendation System

A simple movie recommender system that suggests similar movies based on **genres** and **title keywords**, using **TF-IDF vectorization** and **cosine similarity**.

### 🚀 How It Works
1. Each movie is represented as text (title + genres).
2. TF-IDF converts this text into numerical feature vectors.
3. Cosine similarity measures closeness between movies.
4. The system recommends the top-N most similar movies.

### 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn (TF-IDF + cosine similarity)

### ▶️ How to Run
```bash
py -m venv venv
.\venv\Scripts\activate
pip install pandas scikit-learn numpy
py recommender.py
