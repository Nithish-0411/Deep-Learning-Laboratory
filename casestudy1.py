# Import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0–255) to (0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Build Multi-Layer Perceptron (MLP)
model = Sequential([
    Flatten(input_shape=(28, 28)),     # Flatten 2D image to 1D
    Dense(128, activation='relu'),     # Hidden layer 1
    Dense(64, activation='relu'),      # Hidden layer 2
    Dense(10, activation='softmax')    # Output layer (10 classes)
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 5. Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 6. Predict on first 5 samples
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample Predictions:")
for i in range(5):
    print(f"Image {i+1} → Predicted: {predicted_labels[i]}, Actual: {y_test[i]}")
