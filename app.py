import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "data/ml-100k/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        "data/ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        header=None
    )

    movies = movies[[0, 1]]
    movies.columns = ["movie_id", "title"]

    data = pd.merge(ratings, movies, on="movie_id")
    return data


@st.cache_data
def build_matrices(data):
    user_movie_matrix = data.pivot_table(index="user_id", columns="title", values="rating").fillna(0)
    movie_user_matrix = data.pivot_table(index="title", columns="user_id", values="rating").fillna(0)

    user_similarity = cosine_similarity(user_movie_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    movie_similarity = cosine_similarity(movie_user_matrix)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)

    return user_movie_matrix, movie_similarity_df, user_similarity_df


# -------------------- RECOMMENDATION FUNCTIONS --------------------
def recommend_movies_for_user(user_id, user_movie_matrix, user_similarity_df, top_n=10):
    if user_id not in user_movie_matrix.index:
        return None

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
    recommendations = recommendations.sort_values(ascending=False).head(top_n)

    return recommendations.reset_index().rename(columns={"title": "Movie", 0: "Predicted Rating"})


def recommend_similar_movies(movie_name, movie_similarity_df, top_n=10):
    if movie_name not in movie_similarity_df.index:
        return None

    scores = movie_similarity_df[movie_name].sort_values(ascending=False).iloc[1:top_n+1]
    return scores.reset_index().rename(columns={"title": "Similar Movie", movie_name: "Similarity Score"})


# -------------------- UI HEADER --------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>
    <h4 style='text-align: center; color: gray;'>
    Collaborative Filtering (User-Based + Item-Based) | MovieLens 100K Dataset
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("")

# -------------------- LOAD + BUILD --------------------
data = load_data()
user_movie_matrix, movie_similarity_df, user_similarity_df = build_matrices(data)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Recommendation Options")
option = st.sidebar.radio(
    "Select Mode",
    ["👤 User-Based Recommendation", "🎥 Similar Movies (Item-Based)"]
)

top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.info("📌 Dataset: MovieLens 100K\n\nUsers: 943 | Movies: 1682 | Ratings: 100,000")

# -------------------- METRICS --------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", "943")
col2.metric("Total Movies", "1682")
col3.metric("Total Ratings", "100,000")

st.write("---")

# -------------------- MAIN CONTENT --------------------
if option == "👤 User-Based Recommendation":
    st.subheader("👤 User-Based Collaborative Filtering")
    st.write("This recommends movies for a user based on ratings of similar users.")

    user_id = st.number_input("Enter User ID (1 - 943)", min_value=1, max_value=943, value=5)

    if st.button("🎯 Recommend Movies"):
        recs = recommend_movies_for_user(user_id, user_movie_matrix, user_similarity_df, top_n=top_n)

        if recs is None:
            st.error("User not found!")
        else:
            st.success(f"Top {top_n} recommendations for User {user_id}")
            st.dataframe(recs, use_container_width=True)

            st.write("📊 Predicted Rating Distribution")
            st.bar_chart(recs.set_index("Movie")["Predicted Rating"])

elif option == "🎥 Similar Movies (Item-Based)":
    st.subheader("🎥 Item-Based Collaborative Filtering")
    st.write("This finds similar movies based on cosine similarity between rating vectors.")

    movie_list = sorted(movie_similarity_df.index.tolist())
    movie_name = st.selectbox("Select a Movie", movie_list)

    if st.button("🔍 Find Similar Movies"):
        recs = recommend_similar_movies(movie_name, movie_similarity_df, top_n=top_n)

        if recs is None:
            st.error("Movie not found!")
        else:
            st.success(f"Top {top_n} movies similar to: {movie_name}")
            st.dataframe(recs, use_container_width=True)

            st.write("📊 Similarity Score Distribution")
            st.bar_chart(recs.set_index("Similar Movie")["Similarity Score"])

st.write("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed by Vanshika Arora | Streamlit ML Demo</p>",
    unsafe_allow_html=True
)
