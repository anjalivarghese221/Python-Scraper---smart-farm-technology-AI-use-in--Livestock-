"""
Sentiment Analysis Model Training Script
Step 2: Build and train a sentiment classification model

This script:
1. Loads training data with labeled sentiments
2. Preprocesses text data
3. Trains a sentiment classification model
4. Saves the trained model for later use
"""

import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class SentimentModelTrainer:
    """Train sentiment analysis model on labeled data"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = None
        self.model_type = None
        
    def load_training_data(self, filepath):
        """Load training data from JSON file"""
        print(f"\nLoading training data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} training samples")
        print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        return df
    
    def preprocess_data(self, df):
        """Prepare data for training"""
        print("\nPreprocessing data...")
        # Remove any null values
        df = df.dropna(subset=['text', 'sentiment'])
        
        # Ensure text is string type
        df['text'] = df['text'].astype(str)
        
        # Map sentiment labels to standardized format
        sentiment_map = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'pos': 'positive',
            'neg': 'negative',
            'neu': 'neutral'
        }
        df['sentiment'] = df['sentiment'].str.lower().map(sentiment_map)
        
        print(f"After preprocessing: {len(df)} samples")
        return df
    
    def train_naive_bayes(self, X_train, y_train):
        """Train Naive Bayes classifier"""
        print("\nTraining Naive Bayes model...")
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train, y_train)
        self.model_type = 'naive_bayes'
        return model
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression classifier"""
        print("\nTraining Logistic Regression model...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        self.model_type = 'logistic_regression'
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def train(self, training_data_path, model_type='logistic_regression'):
        """Complete training pipeline"""
        print("=" * 60)
        print("SENTIMENT MODEL TRAINING")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_training_data(training_data_path)
        df = self.preprocess_data(df)
        
        # Split data
        print("\nSplitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
        )
        
        # Vectorize text
        print("\nVectorizing text with TF-IDF...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print(f"Feature dimensions: {X_train_vec.shape[1]} features")
        
        # Train model
        if model_type == 'naive_bayes':
            self.model = self.train_naive_bayes(X_train_vec, y_train)
        else:
            self.model = self.train_logistic_regression(X_train_vec, y_train)
        
        # Evaluate
        accuracy = self.evaluate_model(self.model, X_test_vec, y_test)
        
        print("\n" + "=" * 60)
        print(f"Training complete! Final accuracy: {accuracy:.4f}")
        print("=" * 60)
        
        return accuracy
    
    def save_model(self, model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Save trained model and vectorizer"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        print(f"\nSaving model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Saving vectorizer to {vectorizer_path}...")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print("Model and vectorizer saved successfully!")
    
    def predict(self, texts):
        """Predict sentiment for new texts"""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities


def main():
    """Main execution function"""
    trainer = SentimentModelTrainer()
    
    # Train the model
    training_file = 'sentiment_training_data.json'
    model_type = 'logistic_regression'  # or 'naive_bayes'
    
    try:
        accuracy = trainer.train(training_file, model_type=model_type)
        
        # Save the model
        trainer.save_model()
        
        # Test with sample predictions
        print("\n" + "=" * 60)
        print("TESTING MODEL WITH SAMPLE PREDICTIONS")
        print("=" * 60)
        
        test_samples = [
            "This smart farming technology is amazing and increases productivity",
            "The AI system failed and caused major problems on our farm",
            "The sensors are working as expected, nothing special to report"
        ]
        
        for text in test_samples:
            predictions, probs = trainer.predict(text)
            print(f"\nText: {text}")
            print(f"Predicted sentiment: {predictions[0]}")
            print(f"Confidence: {max(probs[0]):.4f}")
        
    except FileNotFoundError:
        print(f"\nERROR: Training data file '{training_file}' not found!")
        print("Please create the training dataset first.")
        print("Run the create_training_data.py script to generate sample data.")


if __name__ == "__main__":
    main()
