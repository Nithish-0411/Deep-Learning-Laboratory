import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
import networkx as nx

# 1️⃣ Sample Twitter-like data
tweets = [
    "I love the new product! It's fantastic.",
    "This update is terrible, I hate it.",
    "The new feature is okay, nothing special.",
    "Absolutely wonderful performance, great work team!",
    "Disappointed with the new design.",
    "The product is good but could be better.",
    "Horrible experience, not satisfied at all.",
    "Excellent improvement, much faster now!",
    "Not bad, but still needs fixes.",
    "I’m happy with the new update!"
]

# Sentiment labels (1 = Positive, 0 = Negative)
labels = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 1])

# 2️⃣ Text preprocessing
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)
padded = pad_sequences(sequences, padding='post', maxlen=10)

# 3️⃣ Build RNN (LSTM) Model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 4️⃣ Train model
history = model.fit(padded, labels, epochs=10, verbose=0)

# 5️⃣ Predict Sentiments
predictions = (model.predict(padded) > 0.5).astype("int32").flatten()

# Display results
results = pd.DataFrame({
    "Tweet": tweets,
    "Predicted Sentiment": ["Positive" if p == 1 else "Negative" for p in predictions]
})
print(results)

# 6️⃣ Build Network Graph (relationships by shared words)
G = nx.Graph()
for i, tweet in enumerate(tweets):
    G.add_node(i, sentiment="Positive" if predictions[i] == 1 else "Negative", text=tweet)

# Connect tweets sharing at least one keyword
for i in range(len(tweets)):
    for j in range(i + 1, len(tweets)):
        words_i = set(tweets[i].lower().split())
        words_j = set(tweets[j].lower().split())
        if len(words_i.intersection(words_j)) > 0:
            G.add_edge(i, j)

# 7️⃣ Visualize Network
pos = nx.spring_layout(G, seed=42)
colors = ['lightgreen' if G.nodes[n]['sentiment'] == 'Positive' else 'lightcoral' for n in G.nodes]

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1000, font_size=8, font_weight='bold')
plt.title("Twitter Sentiment Network Graph")
plt.show()