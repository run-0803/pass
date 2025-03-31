import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dot, Input
from sklearn.metrics import mean_squared_error

# Load ratings dataset
data_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
column_names = ['userId', 'movieId', 'rating', 'TimeStamp']
data = pd.read_csv(data_url, sep='\t', names=column_names, header=None)

# Adjust IDs to be zero-indexed
data['userId'] = data['userId'] - 1
data['movieId'] = data['movieId'] - 1

# Load movie metadata (u.item)
item_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
movie_columns = ['movieId', 'title'] + [f'col_{i}' for i in range(22)]
movies = pd.read_csv(item_url, sep='|', encoding='latin-1', names=movie_columns, usecols=[0, 1])
movies['movieId'] = movies['movieId'] - 1

# Define embedding size and number of unique users/movies
num_users = data['userId'].nunique()
num_movies = data['movieId'].nunique()
embedding_size = 50

# Model architecture
user_input = Input(shape=[1], dtype=tf.int32, name='user')
movie_input = Input(shape=[1], dtype=tf.int32, name='movie')

# Give the movie embedding layer an explicit name so we can retrieve it later
user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
movie_embedding = Embedding(num_movies, embedding_size, name='movie_embedding')(movie_input)

user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)

dot_prod = Dot(axes=1)([user_vec, movie_vec])

model = Model(inputs=[user_input, movie_input], outputs=dot_prod)
model.compile(optimizer='adam', loss='mean_squared_error')

# Prepare training data
user_data = data['userId'].values
movie_data = data['movieId'].values
rating_data = data['rating'].values

# Train the model
model.fit([user_data, movie_data], rating_data, epochs=5, batch_size=64, verbose=1)

# ---- New Code for Custom Recommendations Based on Watched Movies ----

def get_user_profile(watched_movie_ids):
    """
    Given a list/array of watched movie IDs (zero-indexed),
    compute an aggregate user profile embedding by averaging their movie embeddings.
    """
    # Retrieve the movie embedding weights
    movie_emb_layer = model.get_layer('movie_embedding')
    movie_emb_weights = movie_emb_layer.get_weights()[0]  # shape: (num_movies, embedding_size)
    
    # Get embeddings for the watched movies and compute the mean
    watched_embeddings = movie_emb_weights[watched_movie_ids]
    user_profile = np.mean(watched_embeddings, axis=0)
    return user_profile

def recommend_movies_from_history(watched_movie_ids, num_rec=5):
    """
    Recommend movies based on a list of watched movie IDs.
    Returns movie IDs and titles that the user hasn't seen.
    """
    user_profile = get_user_profile(watched_movie_ids)  # shape: (embedding_size,)
    
    # Retrieve all movie embeddings
    movie_emb_layer = model.get_layer('movie_embedding')
    movie_emb_weights = movie_emb_layer.get_weights()[0]  # shape: (num_movies, embedding_size)
    
    # Compute similarity scores (using dot product here)
    scores = movie_emb_weights.dot(user_profile)
    
    # Exclude already watched movies by setting their score to -inf
    scores[watched_movie_ids] = -np.inf
    
    # Get top recommendations (indices of movies)
    top_movie_indices = np.argsort(scores)[-num_rec:][::-1]
    
    # Retrieve corresponding movie titles
    recommended = movies[movies['movieId'].isin(top_movie_indices)]
    # Sort in the same order as top_movie_indices
    recommended = recommended.set_index('movieId').loc[top_movie_indices].reset_index()
    # Adjust movieId back to 1-indexed for display (if desired)
    recommended['movieId'] = recommended['movieId'] + 1
    return recommended[['movieId', 'title']]

# Example: The user has watched movies with IDs 10, 20, and 30 (0-indexed; adjust as needed)
watched_movies = np.array([1])
recommended_df = recommend_movies_from_history(watched_movies, num_rec=5)
print("\nRecommended Movies Based on Watched History:")
print(recommended_df)