"""
Experiment 1: Sentiment Analysis of Fitness Reviews
Problem Statement: Fitness applications and workout platforms receive user reviews
and feedback about exercises, workout plans, trainers, and fitness programs.
This experiment develops an NLP system to analyze fitness-related reviews and
classify user opinions as positive, negative, or neutral.
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class FitnessSentimentAnalyzer:
    def __init__(self, data_path=None, model_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = data_path or os.path.join(base_dir, "data", "fitness_reviews.csv")
        self.model_dir = model_dir or os.path.join(base_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.vectorizer_path = os.path.join(self.model_dir, "sentiment_vectorizer.joblib")
        self.model_path = os.path.join(self.model_dir, "sentiment_model.joblib")
        
        self.vader = SentimentIntensityAnalyzer()
        self.vectorizer = None
        self.model = None
        self.metrics = None
        self._initialize_or_train()

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def train_and_evaluate(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Review dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        df["cleaned_text"] = df["review_text"].apply(self._clean_text)

        X = df["cleaned_text"]
        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model = LogisticRegression(max_iter=1000, C=1.5, class_weight='balanced', random_state=42)
        self.model.fit(X_train_vec, y_train)

        y_pred = self.model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        labels = sorted(list(set(y)))
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        self.metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        # Save artifacts
        joblib.dump(self.vectorizer, self.vectorizer_path)
        joblib.dump(self.model, self.model_path)

        return self.metrics

    def _initialize_or_train(self):
        if os.path.exists(self.vectorizer_path) and os.path.exists(self.model_path):
            try:
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.model = joblib.load(self.model_path)
            except Exception:
                self.train_and_evaluate()
        else:
            self.train_and_evaluate()

    def predict(self, text):
        clean_text = self._clean_text(text)
        vader_scores = self.vader.polarity_scores(text)

        if not clean_text:
            return {
                "sentiment": "Neutral",
                "confidence": 0.5,
                "probabilities": {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34},
                "vader_scores": vader_scores,
                "key_keywords": []
            }

        # Hybrid prediction using ML model and VADER
        vec = self.vectorizer.transform([clean_text])
        ml_probs = self.model.predict_proba(vec)[0]
        classes = self.model.classes_
        prob_dict = {classes[i]: float(ml_probs[i]) for i in range(len(classes))}

        # Adjust ML probability with VADER compound score
        compound = vader_scores["compound"]
        if compound >= 0.35 and prob_dict.get("Positive", 0) > 0.3:
            final_sentiment = "Positive"
            confidence = max(prob_dict.get("Positive", 0.5), (compound + 1) / 2)
        elif compound <= -0.35 and prob_dict.get("Negative", 0) > 0.3:
            final_sentiment = "Negative"
            confidence = max(prob_dict.get("Negative", 0.5), abs(compound))
        else:
            final_sentiment = self.model.predict(vec)[0]
            confidence = float(max(ml_probs))

        # Extract top influential terms
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        row_indices = vec.indices
        top_words = [feature_names[i] for i in row_indices]

        return {
            "text": text,
            "sentiment": final_sentiment,
            "confidence": round(float(confidence), 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "vader_scores": vader_scores,
            "influential_terms": top_words[:8]
        }

    def analyze_aspects(self, text):
        """Identifies specific fitness aspects in reviews and evaluates sentiment on them."""
        aspect_keywords = {
            "Trainer & Instruction": ["trainer", "coach", "instruction", "cues", "explanation", "video"],
            "Intensity & Difficulty": ["intense", "intensity", "hard", "easy", "tempo", "sweat", "burn", "exhausting", "brutal"],
            "Joint Safety & Form": ["joint", "knee", "back", "wrist", "pain", "strain", "injury", "form", "posture"],
            "Equipment & Setup": ["dumbbell", "barbell", "kettlebell", "mat", "machine", "weights", "equipment", "space"],
            "Progress & Results": ["results", "pr", "growth", "fat loss", "stamina", "pump", "stronger", "toned"]
        }
        
        detected_aspects = {}
        text_lower = text.lower()
        for aspect, keywords in aspect_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                sub_sentiment = self.predict(text)["sentiment"]
                detected_aspects[aspect] = {
                    "matched_keywords": matches,
                    "sentiment": sub_sentiment
                }
        return detected_aspects


sentiment_analyzer = FitnessSentimentAnalyzer()

