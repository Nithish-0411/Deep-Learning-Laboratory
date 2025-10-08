# Import libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Define XOR inputs and outputs
X = np.array([[0,0],[0,1],[1,0],[1,1]])   # Input pairs
y = np.array([[0],[1],[1],[0]])           # Expected XOR output

# 2. Build the Multilayer Perceptron model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # Hidden layer with 4 neurons
model.add(Dense(1, activation='sigmoid'))            # Output layer (binary output)

# 3. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the model
model.fit(X, y, epochs=500, verbose=0)

# 5. Evaluate and predict
loss, acc = model.evaluate(X, y, verbose=0)
print(f"✅ Model Accuracy: {acc:.4f}")

predictions = model.predict(X)
predicted = (predictions > 0.5).astype(int)

# 6. Display results
print("\nInput\tPredicted Output\tExpected Output")
for i in range(len(X)):
    print(f"{X[i]}\t\t{predicted[i][0]}\t\t\t{y[i][0]}")
