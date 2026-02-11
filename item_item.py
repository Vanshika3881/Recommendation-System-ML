import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings
ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Load movies
movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None
)

movies = movies[[0, 1]]
movies.columns = ["movie_id", "title"]

# Merge
data = pd.merge(ratings, movies, on="movie_id")

print("Dataset Loaded Successfully!")
print("Merged Shape:", data.shape)

# Create movie-user matrix
movie_user_matrix = data.pivot_table(index="title", columns="user_id", values="rating").fillna(0)

print("Movie-User Matrix Shape:", movie_user_matrix.shape)

# Compute cosine similarity between movies
movie_similarity = cosine_similarity(movie_user_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)

# Function to recommend similar movies
def recommend_similar_movies(movie_name, top_n=10):
    if movie_name not in movie_similarity_df.index:
        return "Movie not found! Please enter correct movie name."

    scores = movie_similarity_df[movie_name].sort_values(ascending=False)
    return scores.iloc[1:top_n+1]

# Interactive input
movie = input("\nEnter a movie name (example: Toy Story (1995)): ")

print("\nTop Similar Movies:\n")
print(recommend_similar_movies(movie, top_n=10))
