# AI-POWERED SPAM DETECTION FOR THUNDERBIRD
# "This project was developed as part of the Graduation Project course."

This project is a Thunderbird extension powered by machine learning models to detect and classify spam and phishing emails in real time. It analyzes both the email content and embedded URLs, offering enhanced email security and clean inbox management.

# Key Features

- Real-time spam & phishing detection
- Text-based classification using ML models (Naive Bayes, SVM, Logistic Regression, etc.)
- URL-based phishing detection
- Dual spam score (text + URL)
- User-selectable ML algorithms
- Flask-based backend API integration
- Lightweight, secure, and easy to use

# Technologies Used

- Python, Flask, Scikit-learn
- Thunderbird WebExtension API
- HTML / CSS / JavaScript
- TF-IDF for feature extraction
- Machine Learning algorithms

# How It Works

1. The user opens an email in Thunderbird.
2. The extension sends email text and any embedded URLs to a Flask API.
3.  The API uses ML models to analyze content and returns:
   - A spam score (0–100)
   - A prediction label: `safe`, `suspicious`, or `spam`.
4. Scores are shown instantly within the Thunderbird interface.
5. The user can switch models from dropdowns to compare results.

# Models and Datasets

- Text Models trained on: [Enron Spam Dataset](https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data)
- URL Models trained on: [Spam URL Prediction Dataset](https://www.kaggle.com/datasets/shivamb/spam-url-prediction)

Models:
- Text: `Naive Bayes`, `SVM`, `Logistic Regression`, `Random Forest`, `XGBoost`
- URL: `Naive Bayes`, `Decision Tree`, `SVM`

# Setup & Run

Prepare the Python environment to run the application. Then start the Flask API (app.py). Add the folder as an add-on from the Add-ons tab of Thunderbird and run it.


