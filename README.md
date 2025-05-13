# Amazon Product Reviews Sentiment Analysis

This project involves analyzing product reviews from Amazon using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The goal is to classify reviews as positive, negative, or neutral based on the sentiment expressed in the review text. This is done using Logistic Regression, Deep Learning (LSTM), and NLP techniques to process and analyze the review text.

Project Overview
The project analyzes a dataset of Amazon product reviews. The key objectives are:

Clean and preprocess the review text using NLP techniques.

Train a model to predict the sentiment of each review.

Evaluate the model’s performance using various metrics such as accuracy, precision, recall, and F1-score.

Build a web app (using Flask or FastAPI) to allow users to input product reviews and get real-time sentiment predictions.

Technologies Used
Python

pandas (for data manipulation)

nltk (for text processing)

scikit-learn (for machine learning models)

TensorFlow/Keras (for deep learning models like LSTM)

Flask/FastAPI (for the web application)

ngrok (for exposing the local server to the internet)

Jupyter/Google Colab (for data analysis and training the model)

Dataset
The dataset used for this project consists of product reviews from Amazon. It includes the following columns:

id: Unique identifier for the review.

asins: Product identifier.

brand: Brand of the product.

categories: Product category.

colors: Available colors for the product.

reviews.text: Text of the product review.

reviews.rating: Rating given by the user.

reviews.date: Date the review was posted.

reviews.userCity: User's city.

reviews.username: Name of the reviewer.

reviews.doRecommend: Whether the user recommends the product (Boolean).

reviews.numHelpful: Number of helpful votes the review received.

Getting Started
1. Clone the Repository
To get started with the project, clone the repository to your local machine:

bash
Copy
Edit
git clone https://github.com/yourusername/amazon-product-reviews-sentiment-analysis.git
cd amazon-product-reviews-sentiment-analysis
2. Install Dependencies
You can install the required Python dependencies using pip:

bash
Copy
Edit
pip install -r requirements.txt
Here’s the list of libraries used in this project:

flask (for building the web app)

fastapi (optional, if using FastAPI for the web app)

tensorflow (for building the LSTM model)

scikit-learn (for machine learning models)

pandas (for data manipulation)

nltk (for text preprocessing)

pyngrok (for exposing the local Flask/FastAPI app via ngrok)

requests (for making API requests, if needed)

3. Data Preparation
Ensure you have the dataset (Amazon_product_reviews.csv) available in your project folder.

The dataset can be downloaded from Amazon Product Reviews Dataset.

python
Copy
Edit
import pandas as pd
df = pd.read_csv('Amazon_product_reviews.csv')
4. Preprocessing the Data
Run the following code to preprocess the data by:

Removing null values.

Tokenizing and cleaning the review text.

Converting the text into numerical data (TF-IDF, Word2Vec, etc.).

python
Copy
Edit
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocessing function
def clean_text(text):
    # Tokenize, remove stopwords, and perform other cleaning tasks
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(cleaned_words)

df['cleaned_text'] = df['reviews.text'].apply(clean_text)
5. Train the Model
You can train either a Logistic Regression model or a Deep Learning LSTM model to classify the sentiment of the reviews.

Logistic Regression Example:
python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Vectorize the review text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)  # Positive/Negative classification

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
LSTM Example:
python
Copy
Edit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Preprocessing: Convert text to sequences, pad sequences, etc.

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model
model.fit(X_train, y_train, epochs=5, batch_size=64)
6. Web Application (Flask/FastAPI)
Once you have trained your model, you can deploy it as a web service using Flask or FastAPI.

Flask Example:
python
Copy
Edit
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review_text = data['review_text']
    cleaned_text = clean_text(review_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    
    sentiment = 'positive' if prediction == 1 else 'negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
7. Expose the Flask/FastAPI App with Ngrok
Use ngrok to expose your local server to the internet:

python
Copy
Edit
from pyngrok import ngrok

# Expose the Flask app to the internet
public_url = ngrok.connect(5000)  # or 8000 for FastAPI
print(f' * ngrok tunnel "http://127.0.0.1:5000" -> "http://{public_url}"')
Project Evaluation
Accuracy: Evaluate the model using accuracy, precision, recall, and F1-score.

Model Comparison: Compare results between Logistic Regression and the LSTM-based deep learning model.

Deployment: Use Flask/FastAPI to deploy your model and test it with real user input.

Future Improvements
Sentiment categories: Instead of just positive and negative, add neutral sentiment for a more detailed classification.

Other Models: Experiment with more advanced models such as BERT, GPT, etc., for better text understanding.

User Interface: Build a user-friendly interface for the sentiment analysis tool.

