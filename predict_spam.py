import pickle

# Load the trained model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    feature_extraction = pickle.load(vectorizer_file)

# Function to predict whether a mail is spam or ham
def predict_mail(input_mail):
    input_features = feature_extraction.transform([input_mail])
    prediction = model.predict(input_features)
    return 'Ham' if prediction[0] == 1 else 'Spam'

# Sample input
sample_input = "I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have been wonderful and a blessing at all times."

# Prediction
result = predict_mail(sample_input)
print(f"Prediction: {result} mail")
