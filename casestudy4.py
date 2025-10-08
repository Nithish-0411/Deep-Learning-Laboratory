# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Simulated User-Movie Ratings Matrix
# Rows → Users, Columns → Movies, Values → Ratings (0–5)
ratings = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 0],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 0],
    [0, 1, 5, 4, 0]
])

users = ['User1', 'User2', 'User3', 'User4', 'User5']
movies = ['MovieA', 'MovieB', 'MovieC', 'MovieD', 'MovieE']

ratings_df = pd.DataFrame(ratings, index=users, columns=movies)
print("🎬 Original Ratings Matrix:\n")
print(ratings_df)

# 2. Normalize ratings (to 0–1 range)
ratings_norm = ratings / 5.0

# 3. Split train-test
x_train, x_test = train_test_split(ratings_norm, test_size=0.4, random_state=42)

# 4. Build RBM using Keras (simplified structure)
num_visible = ratings_norm.shape[1]  # 5 movies
num_hidden = 3                       # latent features

model = Sequential()
model.add(Dense(num_hidden, activation='relu', input_dim=num_visible))
model.add(Dense(num_visible, activation='sigmoid'))  # reconstruct movie preferences

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# 5. Train the RBM
model.fit(x_train, x_train, epochs=300, batch_size=2, verbose=0)

# 6. Predict for test users
predictions = model.predict(x_test)

# 7. Display recommendations
print("\n🔍 Predicted Ratings (normalized 0–1):")
pred_df = pd.DataFrame(predictions, columns=movies)
print(pred_df.round(2))

# 8. Suggest top movies for one user (example)
user_index = 0
original_ratings = x_test[user_index]
predicted_ratings = predictions[user_index]

unseen_movies = np.where(original_ratings == 0)[0]
recommended_movies = [movies[i] for i in unseen_movies[np.argsort(predicted_ratings[unseen_movies])[::-1]]]

print(f"\n🎯 Recommended Movies for Test User {user_index+1}: {recommended_movies[:2]}")
