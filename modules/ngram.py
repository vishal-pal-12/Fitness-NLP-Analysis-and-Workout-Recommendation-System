"""
Experiment 5: Unigram, Bigram, and Trigram Models for Fitness Text
Problem Statement: Users describe their fitness goals using phrases such as
weight loss workout, muscle building exercises, and home workout plan.
This experiment implements Unigram, Bigram, and Trigram models on fitness-related
text to identify frequently occurring words and meaningful word sequences that help
understand user workout preferences.
"""

import os
import math
from collections import Counter, defaultdict
import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures, TrigramCollocationFinder, TrigramAssocMeasures

from .preprocessing import tokenize_and_filter, preprocess_text


class FitnessNgramLanguageModel:
    def __init__(self, data_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.path.join(base_dir, "data")
        
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.bigram_transitions = defaultdict(Counter)
        self.trigram_transitions = defaultdict(Counter)
        self.vocab = set()
        self.total_words = 0
        self.corpus_tokens = []
        
        self._build_corpus_models()

    def _build_corpus_models(self):
        corpus_texts = []

        # 1. Workout descriptions and tags
        workouts_file = os.path.join(self.data_dir, "workout_plans.csv")
        if os.path.exists(workouts_file):
            df_w = pd.read_csv(workouts_file)
            for _, row in df_w.iterrows():
                corpus_texts.append(str(row.get("title", "")))
                corpus_texts.append(str(row.get("description", "")))
                corpus_texts.append(str(row.get("exercises", "")).replace(";", " "))
                corpus_texts.append(str(row.get("tags", "")).replace(";", " "))

        # 2. Reviews
        reviews_file = os.path.join(self.data_dir, "fitness_reviews.csv")
        if os.path.exists(reviews_file):
            df_r = pd.read_csv(reviews_file)
            for _, row in df_r.iterrows():
                corpus_texts.append(str(row.get("review_text", "")))

        full_text = " . ".join(corpus_texts)
        tokens_data = tokenize_and_filter(full_text)
        tokens = [t.lower() for t in tokens_data["filtered_tokens"] if t.isalpha()]
        self.corpus_tokens = tokens
        self.total_words = len(tokens)
        self.vocab = set(tokens)

        # Unigrams
        self.unigrams.update(tokens)

        # Bigrams
        bg_list = list(ngrams(tokens, 2))
        self.bigrams.update(bg_list)
        for w1, w2 in bg_list:
            self.bigram_transitions[w1][w2] += 1

        # Trigrams
        tg_list = list(ngrams(tokens, 3))
        self.trigrams.update(tg_list)
        for w1, w2, w3 in tg_list:
            self.trigram_transitions[(w1, w2)][w3] += 1

    def analyze_query(self, query):
        """Extracts unigrams, bigrams, and trigrams from input query."""
        tokens_data = tokenize_and_filter(query)
        raw_tokens = [t.lower() for t in tokens_data["filtered_tokens"] if t.isalpha()]
        
        uni = raw_tokens
        bi = [" ".join(bg) for bg in ngrams(raw_tokens, 2)]
        tri = [" ".join(tg) for tg in ngrams(raw_tokens, 3)]

        return {
            "query": query,
            "tokens": raw_tokens,
            "unigrams_raw": uni,
            "bigrams_raw": bi,
            "trigrams_raw": tri,
            "unigram_frequencies": {u: self.unigrams.get(u, 0) for u in uni},
            "bigram_frequencies": {b: self.bigrams.get(tuple(b.split()), 0) for b in bi},
            "trigram_frequencies": {t: self.trigrams.get(tuple(t.split()), 0) for t in tri}
        }

    def suggest_next_word(self, context_words, top_n=5):
        """
        Suggests the most likely next word given preceding words.
        Uses Trigram transitions if 2 words given, falls back to Bigram.
        """
        if not context_words:
            return [{"word": w, "probability": count / self.total_words} for w, count in self.unigrams.most_common(top_n)]

        words = [w.lower() for w in context_words]
        suggestions = []

        # Try trigram first
        if len(words) >= 2:
            key = (words[-2], words[-1])
            if key in self.trigram_transitions:
                counter = self.trigram_transitions[key]
                total = sum(counter.values())
                for w, cnt in counter.most_common(top_n):
                    suggestions.append({
                        "word": w,
                        "source": "Trigram",
                        "count": cnt,
                        "probability": round(cnt / total, 4)
                    })

        # Fill or fallback with bigram
        if len(suggestions) < top_n and len(words) >= 1:
            key1 = words[-1]
            if key1 in self.bigram_transitions:
                counter = self.bigram_transitions[key1]
                total = sum(counter.values())
                existing_words = {s["word"] for s in suggestions}
                for w, cnt in counter.most_common(top_n * 2):
                    if w not in existing_words:
                        suggestions.append({
                            "word": w,
                            "source": "Bigram",
                            "count": cnt,
                            "probability": round(cnt / total, 4)
                        })
                    if len(suggestions) >= top_n:
                        break

        # Fallback to unigram top words if none found
        if not suggestions:
            for w, cnt in self.unigrams.most_common(top_n):
                suggestions.append({
                    "word": w,
                    "source": "Unigram (Global)",
                    "count": cnt,
                    "probability": round(cnt / self.total_words, 4)
                })

        return suggestions[:top_n]

    def get_top_fitness_collocations(self, top_n=10):
        """Calculates top bigram and trigram collocations using PMI."""
        finder_bi = BigramCollocationFinder.from_words(self.corpus_tokens)
        finder_bi.apply_freq_filter(2)
        measures_bi = BigramAssocMeasures()
        top_bigrams = finder_bi.nbest(measures_bi.pmi, top_n)

        finder_tri = TrigramCollocationFinder.from_words(self.corpus_tokens)
        finder_tri.apply_freq_filter(2)
        measures_tri = TrigramAssocMeasures()
        top_trigrams = finder_tri.nbest(measures_tri.pmi, top_n)

        return {
            "top_bigram_collocations": [" ".join(bg) for bg in top_bigrams],
            "top_trigram_collocations": [" ".join(tg) for tg in top_trigrams]
        }

    def compute_query_probability(self, query):
        """Computes query log likelihood and perplexity using Laplace Add-1 smoothed Bigram model."""
        tokens = [t.lower() for t in tokenize_and_filter(query)["filtered_tokens"] if t.isalpha()]
        if len(tokens) < 2:
            return {"tokens": tokens, "log_prob": 0.0, "perplexity": 0.0}

        vocab_size = len(self.vocab)
        log_prob = 0.0
        details = []

        for w1, w2 in ngrams(tokens, 2):
            count_w1_w2 = self.bigrams.get((w1, w2), 0)
            count_w1 = self.unigrams.get(w1, 0)
            # Laplace Add-1 smoothing
            prob = (count_w1_w2 + 1) / (count_w1 + vocab_size)
            log_prob += math.log(prob)
            details.append({"bigram": f"{w1} {w2}", "smoothed_prob": round(prob, 6)})

        n = len(tokens) - 1
        perplexity = math.exp(-log_prob / n) if n > 0 else 0.0

        return {
            "tokens": tokens,
            "log_probability": round(log_prob, 4),
            "perplexity": round(perplexity, 2),
            "bigram_steps": details
        }


ngram_model = FitnessNgramLanguageModel()

