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

# USER-BASED MATRIX
user_movie_matrix = data.pivot_table(index="user_id", columns="title", values="rating").fillna(0)
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# ITEM-BASED MATRIX
movie_user_matrix = data.pivot_table(index="title", columns="user_id", values="rating").fillna(0)
movie_similarity = cosine_similarity(movie_user_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)


# USER-BASED RECOMMENDATION
def recommend_movies_for_user(user_id, top_n=10):
    if user_id not in user_movie_matrix.index:
        return "User not found!"

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]

    weighted_ratings = np.zeros(user_movie_matrix.shape[1])
    similarity_sum = np.zeros(user_movie_matrix.shape[1])

    for sim_user, sim_score in similar_users.items():
        user_ratings = user_movie_matrix.loc[sim_user].values
        weighted_ratings += sim_score * user_ratings
        similarity_sum += sim_score

    predicted_ratings = weighted_ratings / (similarity_sum + 1e-9)

    watched = user_movie_matrix.loc[user_id]
    predicted_ratings[watched > 0] = 0

    recommendations = pd.Series(predicted_ratings, index=user_movie_matrix.columns)
    return recommendations.sort_values(ascending=False).head(top_n)


# ITEM-BASED RECOMMENDATION
def recommend_similar_movies(movie_name, top_n=10):
    if movie_name not in movie_similarity_df.index:
        return "Movie not found! Please enter correct movie name."

    scores = movie_similarity_df[movie_name].sort_values(ascending=False)
    return scores.iloc[1:top_n+1]


# MENU
while True:
    print("\n===== Recommendation System =====")
    print("1. Recommend Movies for a User (User-Based Collaborative Filtering)")
    print("2. Recommend Similar Movies (Item-Based Collaborative Filtering)")
    print("3. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        user_id = int(input("Enter User ID (1-943): "))
        print("\nTop Recommended Movies:\n")
        print(recommend_movies_for_user(user_id, 10))

    elif choice == "2":
        movie = input("Enter movie name (example: Toy Story (1995)): ")
        print("\nTop Similar Movies:\n")
        print(recommend_similar_movies(movie, 10))

    elif choice == "3":
        print("Exiting...")
        break

    else:
        print("Invalid choice! Try again.")
