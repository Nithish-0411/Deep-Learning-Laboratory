# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Sample Spam Dataset
data = {
    'text': [
        'Win a free iPhone now', 'Limited offer just for you', 'Hi friend, how are you?',
        'Your account has been updated', 'Get cheap loans instantly', 'Meeting at 5 pm tomorrow',
        'Congratulations! You won a lottery', 'Your OTP is 123456', 'Earn money from home',
        'Project discussion schedule confirmed'
    ],
    'label': ['spam', 'spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# 2. Text Preprocessing: Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define Logistic Regression Model with Regularization
lr = LogisticRegression(solver='liblinear')

# 4. Define Hyperparameter Grid for Tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10],     # Regularization strength
    'penalty': ['l1', 'l2'],     # Type of regularization
}

# 5. Perform Grid Search with Cross Validation
grid = GridSearchCV(lr, param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

# 6. Evaluate Best Model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("✅ Best Parameters:", grid.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
