"""
train_model.py
----------------
Trains a simple text-to-star-rating regression model on customer reviews.

Pipeline:
    1. Load review text + star rating pairs from data/reviews.csv
    2. Convert review text into numeric features using TF-IDF
    3. Train a Linear Regression model to predict the star rating
    4. Save the fitted vectorizer + model together as model.pkl

Run:
    python train_model.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import pickle

# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
data = pd.read_csv('data/reviews.csv')
X = data['review']
y = data['stars']

# ---------------------------------------------------------
# 2. Convert review text into numeric TF-IDF features
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=1000)
X_vectorized = vectorizer.fit_transform(X)

# ---------------------------------------------------------
# 3. Train the regression model
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_vectorized, y)

# ---------------------------------------------------------
# 4. Save vectorizer + model together
# ---------------------------------------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("✅ Model trained and saved as model.pkl")
