import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings data
ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Load movie names
movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None
)

movies = movies[[0, 1]]
movies.columns = ["movie_id", "title"]

# Merge ratings with movie titles
data = pd.merge(ratings, movies, on="movie_id")

print("Dataset Loaded Successfully!")
print("Ratings:", ratings.shape)
print("Movies:", movies.shape)
print("Merged:", data.shape)


# Create user-item matrix
user_movie_matrix = data.pivot_table(index="user_id", columns="title", values="rating").fillna(0)

print("\nUser-Movie Matrix Shape:", user_movie_matrix.shape)


# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)


# Recommend movies for a given user
def recommend_movies(user_id, top_n=5):
    if user_id not in user_movie_matrix.index:
        return "User not found!"

    # Get similarity scores
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]  # top 5 similar users

    # Weighted rating sum
    weighted_ratings = np.zeros(user_movie_matrix.shape[1])
    similarity_sum = np.zeros(user_movie_matrix.shape[1])

    for sim_user, sim_score in similar_users.items():
        user_ratings = user_movie_matrix.loc[sim_user].values
        weighted_ratings += sim_score * user_ratings
        similarity_sum += sim_score

    # Avoid division by zero
    predicted_ratings = weighted_ratings / (similarity_sum + 1e-9)

    # Movies already watched by user
    watched = user_movie_matrix.loc[user_id]
    predicted_ratings[watched > 0] = 0

    recommendations = pd.Series(predicted_ratings, index=user_movie_matrix.columns)
    recommendations = recommendations.sort_values(ascending=False).head(top_n)

    return recommendations


# Test recommendation
user_id = int(input("\nEnter User ID (1-943): "))
print("\nTop Movie Recommendations:\n")
print(recommend_movies(user_id, top_n=10))
