import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1️⃣ Load dataset (MNIST digits for demo)
# EMNIST can be used similarly for full letters+digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2️⃣ Preprocessing
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# One-hot encoding labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3️⃣ Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 4️⃣ Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5️⃣ Train the Model
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

# 6️⃣ Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

# 7️⃣ Make Prediction on a test sample
index = np.random.randint(0, len(X_test))
sample_image = X_test[index].reshape(1, 28, 28, 1)
prediction = np.argmax(model.predict(sample_image))

plt.imshow(X_test[index].reshape(28,28), cmap='gray')
plt.title(f"Predicted Digit: {prediction}")
plt.show()