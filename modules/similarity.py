"""
Experiment 9: Text Similarity for Intelligent Workout Recommendation
Problem Statement: Users have different fitness goals, experience levels, and
exercise preferences, making it difficult to select suitable workout plans manually.
This experiment calculates text similarity between user requests and available workout
descriptions to identify and recommend workouts that best match the user's stated
goals and preferences.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess_text
from .ner import extract_named_entities


class WorkoutRecommender:
    def __init__(self, data_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = data_path or os.path.join(base_dir, "data", "workout_plans.csv")
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Workout database not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        
        # Create rich composite representation for NLP matching
        corpus = []
        for _, row in self.df.iterrows():
            composite = (
                f"{row.get('title', '')} "
                f"{row.get('category', '')} "
                f"{row.get('difficulty_level', '')} "
                f"{row.get('target_muscles', '')} "
                f"{row.get('equipment_needed', '')} "
                f"{row.get('description', '')} "
                f"{row.get('exercises', '')} "
                f"{row.get('tags', '')}"
            )
            prep = preprocess_text(composite)
            corpus.append(" ".join(prep["lemmatized_tokens"]))

        self.df["processed_composite"] = corpus
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def calculate_jaccard_similarity(self, tokens1, tokens2):
        s1 = set(tokens1)
        s2 = set(tokens2)
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def recommend_workouts(self, user_query, top_k=5, difficulty_filter=None, max_duration=None):
        """
        Calculates TF-IDF Cosine Similarity and Multi-Factor Hybrid Scores
        to recommend workouts matching user preferences.
        """
        prep_query = preprocess_text(user_query)
        query_lemmas = prep_query["lemmatized_tokens"]
        query_str = " ".join(query_lemmas)

        if not query_str:
            return {"query": user_query, "recommendations": []}

        # NER slot extraction
        ner_res = extract_named_entities(user_query)
        slots = ner_res["slots"]

        # TF-IDF Cosine similarity
        query_vec = self.vectorizer.transform([query_str])
        cos_sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        scored_plans = []
        for idx, row in self.df.iterrows():
            cos_score = float(cos_sims[idx])
            
            # Optional hard filters
            if difficulty_filter and difficulty_filter.lower() != "all":
                if row.get("difficulty_level", "").lower() != difficulty_filter.lower():
                    continue

            duration = int(row.get("duration_minutes", 30))
            if max_duration and duration > max_duration:
                continue

            # Jaccard similarity on tokens
            plan_tokens = row["processed_composite"].split()
            jaccard_score = self.calculate_jaccard_similarity(query_lemmas, plan_tokens)

            # Slot bonus: Target muscles match
            muscle_bonus = 0.0
            row_muscles = str(row.get("target_muscles", "")).lower()
            for m in slots.get("muscle_groups", []):
                if m in row_muscles:
                    muscle_bonus += 0.15

            # Slot bonus: Equipment match
            equip_bonus = 0.0
            row_equip = str(row.get("equipment_needed", "")).lower()
            for eq in slots.get("equipment", []):
                if eq in row_equip:
                    equip_bonus += 0.10

            # Slot bonus: Goal / Workout type match
            type_bonus = 0.0
            row_category = str(row.get("category", "")).lower()
            row_tags = str(row.get("tags", "")).lower()
            for wt in slots.get("workout_types", []) + slots.get("goals", []):
                if wt in row_category or wt in row_tags:
                    type_bonus += 0.15

            # Composite hybrid score
            hybrid_score = (
                0.50 * cos_score +
                0.15 * jaccard_score +
                min(0.20, type_bonus) +
                min(0.15, muscle_bonus) +
                min(0.10, equip_bonus)
            )

            # Identify matched keywords
            matched_keywords = [
                tok for tok in set(query_lemmas)
                if tok in plan_tokens and len(tok) > 2
            ]

            scored_plans.append({
                "workout_id": int(row["workout_id"]),
                "title": row["title"],
                "category": row["category"],
                "difficulty_level": row["difficulty_level"],
                "duration_minutes": duration,
                "target_muscles": row["target_muscles"],
                "equipment_needed": row["equipment_needed"],
                "calories_burned_est": int(row["calories_burned_est"]),
                "description": row["description"],
                "exercises": row["exercises"].split("; ") if isinstance(row["exercises"], str) else [],
                "cosine_similarity": round(cos_score, 4),
                "jaccard_similarity": round(jaccard_score, 4),
                "hybrid_score": round(hybrid_score, 4),
                "match_percentage": f"{min(99.8, round(hybrid_score * 100, 1))}%",
                "matched_keywords": matched_keywords[:6]
            })

        # Rank by hybrid score
        scored_plans.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_results = scored_plans[:top_k]

        return {
            "query": user_query,
            "extracted_slots": slots,
            "total_matches_evaluated": len(scored_plans),
            "recommendations": top_results
        }


workout_recommender = WorkoutRecommender()

