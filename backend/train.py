
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import nltk
from nltk.stem import WordNetLemmatizer
import re
from textblob import TextBlob

# Download NLTK data (first run only)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Lowercase, remove punctuation, lemmatize."""
    number_map = {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't'}
    text = ''.join(number_map.get(c, c) for c in text)
    text = str(TextBlob(text).correct())  # autocorrect misspellings
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def load_intents(path='intents.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['intents']

def prepare_data(intents):
    X, y = [], []
    for intent in intents:
        for pattern in intent['patterns']:
            X.append(preprocess(pattern))
            y.append(intent['tag'])
    return X, y

def train_model(X, y):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=5000,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=500,
            C=5.0,
            solver='lbfgs'
        ))
    ])
    
    # Cross-validate to check accuracy
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
    
    # Fit on full data
    pipeline.fit(X, y)
    return pipeline

def save_model(pipeline, intents, model_path='gymbuddy_model.pkl', intents_path='intents_data.pkl'):
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    with open(intents_path, 'wb') as f:
        pickle.dump(intents, f)
    print(f"Model saved to {model_path}")
    print(f"Intents data saved to {intents_path}")

def main():
    print("=== GymBuddy AI - Model Training ===\n")
    
    intents = load_intents('intents.json')
    print(f"Loaded {len(intents)} intent categories")
    
    X, y = prepare_data(intents)
    print(f"Training on {len(X)} patterns\n")
    
    pipeline = train_model(X, y)
    save_model(pipeline, intents)
    
    print("\nTraining complete! ✅")
    print("Now run: python app.py")

if __name__ == '__main__':
    main()