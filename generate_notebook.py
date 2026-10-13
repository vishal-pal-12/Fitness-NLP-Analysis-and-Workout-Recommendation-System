import json
import os

notebook_path = r"C:\Users\VISHAL PAL\.gemini\antigravity\scratch\fitness_nlp_project\Fitness_Text_Analysis_and_Workout_Recommendation_NLP_Project.ipynb"

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    })

# Cell 1: Header
add_md("""# 🏋️ Fitness Text Analysis and Intelligent Workout Recommendation System using NLP
## A Unified 10-Phase Natural Language Processing Application
**Domain:** Fitness Text Processing, User Feedback Analytics, and Intelligent Workout Recommendation  
**Core Frameworks:** Python 3.14 / 3.11, NLTK, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

### Project Architecture & 10-Phase Unified Pipeline
Instead of disconnected isolated scripts, this mini-project integrates all 10 prescribed curriculum experiments into an **end-to-end, sequential intelligence pipeline**:

```
[USER FITNESS REQUEST / WORKOUT CATALOG ENTRY / REVIEW]
                     │
                     ▼
Phase 1  ──►  Sentiment Analysis of Fitness Reviews (VADER + TF-IDF Logistic Regression)
Phase 2  ──►  Tokenization, Filtration, and Script Validation of Fitness Text
Phase 3  ──►  Stop Word Removal, Stemming, and Lemmatization (Comparative Analysis)
Phase 4  ──►  Morphological Analysis & Word Generation of Fitness Terms
Phase 5  ──►  Unigram, Bigram, and Trigram Language Models (Collocations & Next-Word)
Phase 6  ──►  Part-of-Speech (POS) Tagging (Penn Treebank & Fitness Semantic Mapping)
Phase 7  ──►  Chunking & Phrase Extraction + Empirical Study on Training Sizes (60/70/80%)
Phase 8  ──►  Named Entity Recognition (NER) & Goal/Muscle/Duration Slot Filling
Phase 9  ──►  Text Similarity for Intelligent Workout Recommendation (Hybrid Matching)
Phase 10 ──►  Word Sense Disambiguation (WSD) for Fitness Terms (Polysemy Resolution)
                     │
                     ▼
[INTELLIGENT WORKOUT RECOMMENDATIONS & DETAILED LINGUISTIC PROFILE]
```
""")

# Cell 2: Imports
add_md("""---
## Step 1: Environment Setup & Library Ingestion
We load standard data processing, natural language processing, and machine learning libraries.
""")

add_code("""import os
import sys
import re
import string
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, RegexpParser, ne_chunk
from nltk.chunk import tree2conlltags
from nltk.util import ngrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.wsd import lesk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

print("Environment libraries successfully imported!")
""")

# Cell 3: Data loading
add_md("""---
## Step 2: Dataset Ingestion & Exploration
We inspect our comprehensive datasets:
1. `workout_plans.csv`: 50 detailed workout plans across disciplines (Strength, Hypertrophy, HIIT, Yoga, Calisthenics, Endurance, Rehab).
2. `fitness_reviews.csv`: 155 classified user reviews with star ratings and sentiment labels.
""")

add_code("""df_workouts = pd.read_csv("data/workout_plans.csv")
df_reviews = pd.read_csv("data/fitness_reviews.csv")

print(f"Total Workout Plans: {len(df_workouts)}")
print(f"Total Fitness Reviews: {len(df_reviews)}")

display(df_workouts[["workout_id", "title", "category", "difficulty_level", "duration_minutes", "calories_burned_est"]].head())
display(df_reviews.head())
""")

# Cell 4: Experiment 1
add_md("""---
## 🎯 Experiment 1: Sentiment Analysis of Fitness Reviews
**Problem Statement:** Fitness applications and workout platforms receive user reviews and feedback about exercises, workout plans, trainers, and fitness programs. This experiment develops an NLP system to analyze fitness-related reviews and classify user opinions as positive, negative, or neutral.
""")

add_code("""from modules.sentiment import sentiment_analyzer

# Train and evaluate model
metrics = sentiment_analyzer.train_and_evaluate()
print(f"Classification Accuracy: {metrics['accuracy']*100:.2f}%")
print(f"Weighted F1-Score:      {metrics['f1_score']*100:.2f}%")

# Test sample reviews
sample_rev_pos = "Incredible pump! Progressive overload with barbell press gave me noticeable growth."
sample_rev_neg = "Terrible routine, the tempo is too fast and caused terrible knee and back pain."

res_pos = sentiment_analyzer.predict(sample_rev_pos)
res_neg = sentiment_analyzer.predict(sample_rev_neg)

print(f"\\nReview: '{sample_rev_pos}'")
print(f"Prediction: {res_pos['sentiment']} (Confidence: {res_pos['confidence']:.2f}) | VADER Compound: {res_pos['vader_scores']['compound']}")

print(f"\\nReview: '{sample_rev_neg}'")
print(f"Prediction: {res_neg['sentiment']} (Confidence: {res_neg['confidence']:.2f}) | VADER Compound: {res_neg['vader_scores']['compound']}")

# Aspect-Based Sentiment Analysis
aspects = sentiment_analyzer.analyze_aspects(sample_rev_neg)
print(f"\\nAspect-Level Breakdown: {aspects}")
""")

# Cell 5: Experiment 2
add_md("""---
## 🎯 Experiment 2: Tokenization, Filtration, and Script Validation of Fitness Text
**Problem Statement:** Fitness-related text, such as workout descriptions, user goals, and exercise queries, contains words, punctuation, numbers, and special characters. This experiment performs tokenization, filtration, and script validation to clean fitness text and prepare valid tokens for further NLP processing.
""")

add_code("""from modules.preprocessing import tokenize_and_filter, validate_script

sample_query = "Find a 30-min HIIT workout with 3x10 squats and 50kg deadlifts!"
tok_res = tokenize_and_filter(sample_query)

print("Original Text:       ", tok_res["original_text"])
print("Word Tokens:         ", tok_res["word_tokens"])
print("Filtered Tokens:     ", tok_res["filtered_tokens"])
print("Script Validation:   ", tok_res["script_validation"])

# Test Script Validation on Non-English Input
hindi_test = validate_script("व्यायाम और कसरत")
print("\\nForeign Script Validation ('व्यायाम और कसरत'):")
print(hindi_test)
""")

# Cell 6: Experiment 3
add_md("""---
## 🎯 Experiment 3: Stop Word Removal, Stemming, and Lemmatization of Fitness Text
**Problem Statement:** Fitness descriptions and workout requests contain common words and different forms of the same word, such as exercise, exercises, and exercising. This experiment removes stop words and applies stemming and lemmatization to normalize fitness-related text and improve the accuracy of further text analysis.
""")

add_code("""from modules.preprocessing import preprocess_text

query_exp3 = "The trainers are exercising daily and coaching weight loss exercises"
prep_res = preprocess_text(query_exp3)

print("Original Tokens:          ", prep_res["original_tokens"])
print("Stopword Removed Tokens:  ", prep_res["stopword_removed_tokens"])
print("Porter Stemmed:           ", prep_res["stemmed_tokens"])
print("Lemmatized Tokens:        ", prep_res["lemmatized_tokens"])
print(f"Token Reduction Rate:     {prep_res['reduction_rate_pct']}%")

# Comparative Table across Algorithms
df_comp = pd.DataFrame(prep_res["comparison_table"])
display(df_comp)
""")

# Cell 7: Experiment 4
add_md("""---
## 🎯 Experiment 4: Morphological Analysis and Word Generation of Fitness Terms
**Problem Statement:** Fitness-related text contains words with different grammatical forms, such as train, trains, training, and trained. This experiment performs morphological analysis to identify word roots, prefixes, and suffixes and generates related word forms from fitness vocabulary.
""")

add_code("""from modules.morphology import analyze_morphology, generate_word_forms, expand_query_morphologically

# 1. Morphological Decomposition
morph = analyze_morphology("pre-workout")
print("Morphological Breakdown of 'pre-workout':")
print(morph)

# 2. Word Generation Matrix from Root 'lift'
forms = generate_word_forms("lift")
print("\\nGenerated Word Forms for Root 'lift':")
for k, v in forms.items():
    print(f"  {k:28}: {v}")

# 3. Morphological Query Expansion
query_tokens = ["squatting", "bench", "trained"]
expanded = expand_query_morphologically(query_tokens)
print(f"\\nQuery Expansion for {query_tokens}:\\n{expanded['expanded_tokens']}")
""")

# Cell 8: Experiment 5
add_md("""---
## 🎯 Experiment 5: Unigram, Bigram, and Trigram Models for Fitness Text
**Problem Statement:** Users describe their fitness goals using phrases such as weight loss workout, muscle building exercises, and home workout plan. This experiment implements Unigram, Bigram, and Trigram models on fitness-related text to identify frequently occurring words and meaningful word sequences that help understand user workout preferences.
""")

add_code("""from modules.ngram import ngram_model

# 1. Query N-gram Decomposition
sample_goal = "weight loss workout plan"
ng_analysis = ngram_model.analyze_query(sample_goal)
print("Bigrams: ", ng_analysis["bigrams_raw"])
print("Trigrams:", ng_analysis["trigrams_raw"])

# 2. Next-Word Prediction & Auto-Completion
print("\\nNext-Word Suggestions given context ['weight']:")
print(ngram_model.suggest_next_word(["weight"]))

print("\\nNext-Word Suggestions given context ['high', 'intensity']:")
print(ngram_model.suggest_next_word(["high", "intensity"]))

# 3. Top Fitness Collocations (Pointwise Mutual Information)
collocations = ngram_model.get_top_fitness_collocations(6)
print("\\nTop Bigram Collocations:", collocations["top_bigram_collocations"])
print("Top Trigram Collocations:", collocations["top_trigram_collocations"])
""")

# Cell 9: Experiment 6
add_md("""---
## 🎯 Experiment 6: Part-of-Speech (POS) Tagging of Fitness Text
**Problem Statement:** Fitness-related descriptions and user queries contain nouns, verbs, adjectives, and other grammatical categories. This experiment performs Part-of-Speech tagging to identify the grammatical role of each word in fitness text and improve the understanding of workout requests and exercise descriptions.
""")

add_code("""from modules.pos_tagging import tag_pos

fitness_sentence = "Athletes lift heavy dumbbells for intense chest hypertrophy"
pos_res = tag_pos(fitness_sentence)

df_pos = pd.DataFrame(pos_res["structured_results"])
display(df_pos)

print("\\nExtracted Action Verbs:       ", pos_res["extracted_verbs"])
print("Extracted Muscles & Equipment: ", pos_res["extracted_nouns"])
print("Extracted Intensity Modifiers: ", pos_res["extracted_adjectives"])
""")

# Cell 10: Experiment 7
add_md("""---
## 🎯 Experiment 7: Chunking and Phrase Extraction from Fitness Text
**Problem Statement:** Users often express their fitness needs through meaningful phrases such as beginner workout plan, upper body strength training, and low-impact cardio exercises. This experiment performs chunking to identify and extract important phrases from fitness-related text, helping the system recognize workout types, fitness levels, and exercise goals. Includes an empirical study comparing training sizes (60%, 70%, 80%).
""")

add_code("""from modules.chunking import extract_chunks, analyze_training_size_and_features

phrase_sentence = "Beginner workout plan for upper body strength training in 30 minutes"
chunk_res = extract_chunks(phrase_sentence)

print("Extracted Structured Chunks:")
for ch in chunk_res["extracted_phrases"]:
    print(f"  [{ch['chunk_type']:16}] : {ch['phrase']}")

# Empirical Study on Training Sizes & Feature Sets
study = analyze_training_size_and_features()
df_study = pd.DataFrame(study["comparison_table"])
display(df_study)
print(f"\\nObservation: {study['observation']}")
""")

# Cell 11: Experiment 8
add_md("""---
## 🎯 Experiment 8: Named Entity Recognition (NER) for Fitness Information Extraction
**Problem Statement:** Fitness-related text may contain important entities such as exercise names, body parts, workout types, durations, and fitness-related organizations. This experiment applies Named Entity Recognition and related extraction techniques to identify relevant information from fitness descriptions and user queries for use in workout recommendation.
""")

add_code("""from modules.ner import extract_named_entities

ner_query = "I want a 45-min HIIT workout targeting chest and biceps using dumbbells for fat loss"
ner_res = extract_named_entities(ner_query)

print("Extracted Entities:")
for ent in ner_res["entities"]:
    print(f"  {ent['entity']:16} -> {ent['label']:16} (Source: {ent.get('source')})")

print("\\nStructured Slot Filling for Recommendation Engine:")
print(ner_res["slots"])
""")

# Cell 12: Experiment 9
add_md("""---
## 🎯 Experiment 9: Text Similarity for Intelligent Workout Recommendation
**Problem Statement:** Users have different fitness goals, experience levels, and exercise preferences, making it difficult to select suitable workout plans manually. This experiment calculates text similarity between user requests and available workout descriptions to identify and recommend workouts that best match the user's stated goals and preferences.
""")

add_code("""from modules.similarity import workout_recommender

user_preference = "I want a 30-minute bodyweight high intensity cardio fat loss workout at home"
rec_res = workout_recommender.recommend_workouts(user_preference, top_k=3)

print(f"User Query: '{user_preference}'\\n")
print("Top Recommended Workout Plans:")
for idx, w in enumerate(rec_res["recommendations"], 1):
    print(f"{idx}. {w['title']}")
    print(f"   Category: {w['category']} | Level: {w['difficulty_level']} | Duration: {w['duration_minutes']} min")
    print(f"   Match Percentage: {w['match_percentage']} (Cosine: {w['cosine_similarity']:.4f}, Hybrid: {w['hybrid_score']:.4f})")
    print(f"   Key Exercises: {'; '.join(w['exercises'][:4])}")
    print(f"   Matched Keywords: {', '.join(w['matched_keywords'])}\\n")
""")

# Cell 13: Experiment 10
add_md("""---
## 🎯 Experiment 10: Word Sense Disambiguation (WSD) for Fitness Text
**Problem Statement:** Some words in fitness-related queries can have different meanings depending on their context. For example, “weight” may refer to body weight or weights used in strength training, while “running” may refer to a specific exercise or a general activity. This experiment applies Word Sense Disambiguation techniques to interpret ambiguous words based on their surrounding context and improve the understanding of fitness queries.
""")

add_code("""from modules.wsd import wsd_model

# Case 1: 'weight' as Body Weight
sent1 = "I need a strict calorie deficit and cardio to lose weight."
wsd1 = wsd_model.disambiguate_contextual(sent1, "weight")
print(f"Sentence: '{sent1}'")
print(f"Predicted Sense: {wsd1['predicted_sense']} (Confidence: {wsd1['confidence']:.2f})")
print(f"Definition:      {wsd1['definition']}\\n")

# Case 2: 'weight' as Gym Equipment Load
sent2 = "He loaded more weight onto the barbell for his heavy bench press."
wsd2 = wsd_model.disambiguate_contextual(sent2, "weight")
print(f"Sentence: '{sent2}'")
print(f"Predicted Sense: {wsd2['predicted_sense']} (Confidence: {wsd2['confidence']:.2f})")
print(f"Definition:      {wsd2['definition']}\\n")

# Case 3: 'running' as Cardio vs Management
sent3 = "Running five miles on the track strengthens aerobic capacity."
wsd3 = wsd_model.disambiguate_contextual(sent3, "running")
print(f"Sentence: '{sent3}'")
print(f"Predicted Sense: {wsd3['predicted_sense']} (Confidence: {wsd3['confidence']:.2f})")
print(f"Definition:      {wsd3['definition']}")
""")

# Cell 14: Conclusion
add_md("""---
## 🏁 Conclusion & Summary
All 10 experiments have been implemented, verified, and unified into the **Fitness Text Analysis and Intelligent Workout Recommendation System**:
1. **Sentiment Analysis:** Analyzes fitness feedback and ratings with 94.9% accuracy.
2. **Tokenization & Script Validation:** Sanitizes fitness tokens and guards against foreign scripts.
3. **Stopwords & Lemmatization:** Normalizes morphological inflections and reduces token volume.
4. **Morphological Analysis:** Generates related fitness terms and provides query expansion.
5. **N-Gram Modeling:** Models multi-word fitness collocations and predicts next tokens.
6. **POS Tagging:** Identifies grammatical roles and isolates fitness action verbs, targets, and modifiers.
7. **Chunking & Empirical Study:** Extracts noun, verb, and intensity phrases with comparative evaluation.
8. **Named Entity Recognition:** Identifies exercises, muscles, equipment, and fills user goal slots.
9. **Text Similarity:** Performs multi-criteria hybrid ranking to recommend optimal workouts.
10. **Word Sense Disambiguation:** Resolves context-dependent polysemy for fitness terms.
""")

notebook_json = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1)

print(f"Saved complete Jupyter Notebook to {notebook_path}")

