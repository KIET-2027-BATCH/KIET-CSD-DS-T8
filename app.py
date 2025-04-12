import pickle
from flask import Flask, render_template, request, redirect, url_for, session

# Load the trained model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure random key

# Enable template auto-reloading
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Route for the home page
@app.route('/')
def home():
    # Get prediction result and input email from session (then clear)
    result = session.pop('result', None)
    email = session.pop('email', '')
    return render_template('index.html', result=result, email=email)

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_mail = request.form.get('email', '').strip()

    if not input_mail:
        session['result'] = "⚠️ Error: Please provide the email text."
        session['email'] = ''
    else:
        try:
            # Transform input & make prediction
            input_features = tfidf_vectorizer.transform([input_mail])
            prediction = model.predict(input_features)
            result = 'Ham' if prediction[0] == 1 else 'Spam'

            session['result'] = result
            session['email'] = input_mail

        except Exception as e:
            session['result'] = f"⚠️ Error: {str(e)}"
            session['email'] = input_mail

    return redirect(url_for('home'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
