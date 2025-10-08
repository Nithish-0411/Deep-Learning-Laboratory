# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

# 1. Simulate transaction data
# Normal transactions: around 0 mean, Fraudulent: shifted distribution
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 10))
fraud_data = np.random.normal(4, 1, (50, 10))

data = np.vstack((normal_data, fraud_data))
labels = np.hstack((np.zeros(1000), np.ones(50)))  # 0=Normal, 1=Fraud

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

# 2. Build Autoencoder
input_dim = x_train.shape[1]
encoding_dim = 5  # compressed representation

input_layer = Input(shape=(input_dim,))
encoder = Dense(8, activation="relu")(input_layer)
encoder = Dense(encoding_dim, activation="relu")(encoder)
decoder = Dense(8, activation="relu")(encoder)
decoder = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# 3. Train the autoencoder only on normal data
x_train_normal = x_train[y_train == 0]
autoencoder.fit(x_train_normal, x_train_normal, epochs=30, batch_size=16, validation_split=0.1, verbose=0)

# 4. Reconstruction error
reconstructions = autoencoder.predict(x_test)
mse = np.mean(np.power(x_test - reconstructions, 2), axis=1)

# Set threshold for anomaly detection
threshold = np.percentile(mse, 95)
predictions = (mse > threshold).astype(int)

# 5. Evaluate
from sklearn.metrics import confusion_matrix, classification_report

print("Reconstruction error threshold:", round(threshold, 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, digits=4))

# 6. Plot reconstruction error
plt.figure(figsize=(6,4))
plt.hist(mse[y_test==0], bins=30, alpha=0.6, label='Normal')
plt.hist(mse[y_test==1], bins=30, alpha=0.6, label='Fraud')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("Reconstruction Error Distribution")
plt.xlabel("MSE Loss")
plt.ylabel("Frequency")
plt.legend()
plt.show()
