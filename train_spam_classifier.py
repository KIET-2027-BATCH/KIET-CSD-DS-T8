import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle

# Data Collection & Pre-Processing
raw_mail_data = pd.read_csv(r'C:\Users\heman\Downloads\spam_mail_prediction_ML-main\mail_data.csv')
print(raw_mail_data.head())  # Display first few rows

# Replace the null values with an empty string
mail_data = raw_mail_data.fillna('')

# Label Encoding: spam = 0, ham = 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate the data
X = mail_data['Message']
Y = mail_data['Category'].astype('int')  # Ensure labels are integers

# Shuffle the data before splitting
X, Y = shuffle(X, Y, random_state=3)

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature Extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Model Training with class weight balance (optional for imbalance)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_features, Y_train)

# Save model and vectorizer
with open('spam_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(feature_extraction, vectorizer_file)

# Evaluation
training_predictions = model.predict(X_train_features)
test_predictions = model.predict(X_test_features)

training_accuracy = accuracy_score(Y_train, training_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)

print(f'Accuracy on training data: {training_accuracy}')
print(f'Accuracy on test data: {test_accuracy}')

# Extra Evaluation Metrics
print("\nClassification Report:\n", classification_report(Y_test, test_predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_test, test_predictions))

# Optional: Inference function for new mails
def predict_mail(message):
    input_data = feature_extraction.transform([message])
    prediction = model.predict(input_data)
    return "Ham" if prediction[0] == 1 else "Spam"

# Example usage
sample_message = "Congratulations! You've won a $1000 Walmart gift card. Click to claim now!"
print(f'\nSample Message Prediction: {predict_mail(sample_message)}')
