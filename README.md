# 🏋️ Fitness Text Analysis and Intelligent Workout Recommendation System using Natural Language Processing (NLP)

[![Python 3.14](https://img.shields.io/badge/Python-3.14%20%2F%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-2E7D32?style=for-the-badge)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

> **College NLP Mini Project**  
> **Domain:** Fitness Text Processing, Workout Plan Analytics & Semantic Recommendation System  
> **Architecture:** Unified 10-Phase Natural Language Processing Pipeline  
> **Author:** Vishal Pal ([@vishal-pal-12](https://github.com/vishal-pal-12))

---

## 📌 Table of Contents
- [1. Project Overview & Problem Formulation](#-1-project-overview--problem-formulation)
- [2. System Architecture & The 10 Connected Experiments](#-2-system-architecture--the-10-connected-experiments)
- [3. Detailed Breakdown of the 10 Experiments](#-3-detailed-breakdown-of-the-10-experiments)
- [4. Project Structure](#-4-project-structure)
- [5. Installation & Setup](#-5-installation--setup)
- [6. How to Run the Application](#-6-how-to-run-the-application)
  - [Option A: Interactive Streamlit Web Portal (`app.py`)](#option-a-interactive-streamlit-web-portal-apppy)
  - [Option B: Command-Line Predictor (`predict_workout.py`)](#option-b-command-line-predictor-predict_workoutpy)
  - [Option C: Unified Jupyter Notebook (`.ipynb`)](#option-c-unified-jupyter-notebook-ipynb)
  - [Option D: Automated 10-Phase Test Suite (`test_all_modules.py`)](#option-d-automated-10-phase-test-suite-test_all_modulespy)
- [7. Experimental Results & Performance Metrics](#-7-experimental-results--performance-metrics)
- [8. Datasets Overview](#-8-datasets-overview)
- [9. License](#-9-license)

---

## 🎯 1. Project Overview & Problem Formulation

Fitness applications, workout platforms, and gym systems receive diverse textual user inputs: user workout preferences, fitness goals (fat loss, hypertrophy, stamina, mobility), exercise queries, and user reviews regarding workout plans and coaches. 

Selecting an optimal workout plan from large exercise catalogs is challenging due to:
1. **Inflectional variability:** Words like *exercise*, *exercises*, *exercising*, *train*, *trainer*, *training*.
2. **Multi-word domain phrases:** *30-minute HIIT workout*, *bench press*, *upper body hypertrophy*, *low-impact cardio*.
3. **Ambiguity and Polysemy:** The word *weight* means bodily mass (*weight loss*) or gym load (*barbell weight*); *running* can mean cardio locomotion or program execution (*running a split*).
4. **Unstructured user feedback:** Reviews containing sentiment and aspect feedback on joint safety, fatigue, intensity, and coaching cues.

This project delivers a **unified 10-phase Natural Language Processing architecture** that connects all 10 prescribed curriculum experiments into an end-to-end intelligent recommendation engine.

---

## 🏛️ 2. System Architecture & The 10 Connected Experiments

```
               USER WORKOUT QUERY / FITNESS FEEDBACK REVIEW
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 1: Sentiment Analysis of Fitness Reviews (VADER + TF-IDF Classifier)   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 2: Tokenization, Filtration, and Script Validation of Fitness Text     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 3: Stop Word Removal, Stemming, and Lemmatization of Fitness Text      │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 4: Morphological Analysis & Word Generation of Fitness Terms           │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 5: Unigram, Bigram, and Trigram Models for Fitness Text (Collocations) │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 6: Part-of-Speech (POS) Tagging (Verbs, Muscles, Intensity Modifiers)   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 7: Chunking & Phrase Extraction + Training Size Study (60% / 70% / 80%)│
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 8: Named Entity Recognition (NER) & Goal/Muscle/Equipment Slot Filling │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 9: Text Similarity for Intelligent Workout Recommendation (Hybrid)     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 10: Word Sense Disambiguation (WSD) for Fitness Text (Contextual Lesk) │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
[INTELLIGENT WORKOUT RECOMMENDATIONS & STRUCTURED LINGUISTIC ANALYSIS REPORT]
```

---

## 🔬 3. Detailed Breakdown of the 10 Experiments

| # | Experiment Title | Module | Key Techniques & Contributions |
|---|---|---|---|
| **1** | **Sentiment Analysis of Fitness Reviews** | [`modules/sentiment.py`](modules/sentiment.py) | Hybrid VADER + TF-IDF Logistic Regression classifier with aspect-level sentiment breakdown (Trainer, Difficulty, Joint Safety, Equipment, Results). Achieves 94.9% test accuracy. |
| **2** | **Tokenization, Filtration & Script Validation** | [`modules/preprocessing.py`](modules/preprocessing.py) | Treebank/regex tokenizers, contraction handling, special character & rep/set pattern retention (`3x10`, `50kg`), Unicode script validation (English Latin vs Devanagari/Hindi/emojis). |
| **3** | **Stop Word Removal, Stemming & Lemmatization** | [`modules/preprocessing.py`](modules/preprocessing.py) | Fitness domain-tuned stopword removal (exempting directional terms like *up*, *down*), Porter, Lancaster, Snowball stemmers comparison, and POS-aware WordNet Lemmatization. |
| **4** | **Morphological Analysis & Word Generation** | [`modules/morphology.py`](modules/morphology.py) | Affix decomposition (pre-, post-, over-, under- / -ing, -ed, -er, -tion), morphological inflection generator, and query expansion across grammatical forms. |
| **5** | **Unigram, Bigram & Trigram Models** | [`modules/ngram.py`](modules/ngram.py) | MLE & Laplace Add-1 smoothed language models, Pointwise Mutual Information (PMI) collocations (*weight loss*, *high intensity*), and next-word query prediction. |
| **6** | **Part-of-Speech (POS) Tagging** | [`modules/pos_tagging.py`](modules/pos_tagging.py) | Penn Treebank POS tagging with imperative verb handling, semantic category mapping (Action Verbs, Anatomy/Equipment Nouns, Intensity Modifiers). |
| **7** | **Chunking & Phrase Extraction** | [`modules/chunking.py`](modules/chunking.py) | Custom RegexpParser grammar for noun phrases, verb phrases, intensity chunks, IOB tagging, and academic empirical study on training sizes (60%, 70%, 80%). |
| **8** | **Named Entity Recognition (NER)** | [`modules/ner.py`](modules/ner.py) | Dual NLTK MaxEnt + Domain Gazetteer extractor for `EXERCISE`, `MUSCLE_GROUP`, `WORKOUT_TYPE`, `EQUIPMENT`, `DIFFICULTY`, and duration slot-filling. |
| **9** | **Text Similarity & Workout Recommendation** | [`modules/similarity.py`](modules/similarity.py) | Multi-factor hybrid recommender combining TF-IDF cosine similarity, Jaccard token index, target muscle overlap bonus, equipment feasibility, and duration filtering. |
| **10** | **Word Sense Disambiguation (WSD)** | [`modules/wsd.py`](modules/wsd.py) | Disambiguates polysemous fitness terms (*weight*, *running*, *press*, *bar*, *rep*, *set*) via WordNet Lesk and context profile similarity. |

---

## 📂 4. Project Structure

```
fitness_nlp_project/
├── data/
│   ├── workout_plans.csv          # 50 comprehensive workout routines across disciplines
│   ├── fitness_reviews.csv        # 155 classified user reviews with sentiments and ratings
│   └── fitness_entities.json      # Fitness domain entity vocabulary dictionary
├── models/
│   ├── sentiment_model.joblib     # Serialized Logistic Regression sentiment classifier
│   └── sentiment_vectorizer.joblib# Serialized TF-IDF vectorizer
├── modules/
│   ├── __init__.py
│   ├── sentiment.py               # Experiment 1: Sentiment analysis & aspect modeling
│   ├── preprocessing.py           # Experiments 2 & 3: Tokenization, filtration, stopwords, lemmatization
│   ├── morphology.py              # Experiment 4: Affix analysis, root extraction, word generation
│   ├── ngram.py                   # Experiment 5: Unigram/Bigram/Trigram models & suggestions
│   ├── pos_tagging.py             # Experiment 6: Part-of-Speech tagging & semantic categories
│   ├── chunking.py                # Experiment 7: RegexpParser chunking & 60/70/80% empirical study
│   ├── ner.py                     # Experiment 8: Named Entity Recognition & slot filling
│   ├── similarity.py              # Experiment 9: Text similarity & workout recommendation
│   └── wsd.py                     # Experiment 10: Word Sense Disambiguation
├── outputs/                       # Generated evaluation tables & figures
├── app.py                         # Streamlit Interactive Web Application
├── predict_workout.py             # CLI Predictor & Inspection Tool
├── workout_predictor.py           # Alias to CLI Predictor
├── test_all_modules.py            # Comprehensive 10-phase automated test suite
├── create_datasets.py             # Dataset synthesis script
├── generate_notebook.py           # Script to generate Jupyter Notebook
├── Fitness_Text_Analysis_and_Workout_Recommendation_NLP_Project.ipynb # Complete Notebook
├── requirements.txt               # Dependency specification
├── run_app.bat                    # One-click Streamlit launch script
├── run_tests.bat                  # One-click Test Suite execution
├── run_jupyter.bat                # One-click Jupyter launch script
├── README.md                      # Comprehensive documentation
└── PROJECT_REPORT.md              # Academic Mini-Project Report
```

---

## ⚙️ 5. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11 & Python 3.14 on Windows)
- pip package manager

### Steps
```bash
# 1. Clone or navigate to the repository
cd fitness_nlp_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify NLTK corpora (downloaded automatically on first launch)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon')"
```

---

## 🚀 6. How to Run the Application

### Option A: Interactive Streamlit Web Portal (`app.py`)
Double-click `run_app.bat` or run in terminal:
```bash
streamlit run app.py
```
**Features in Web Portal:**
- **Mode 1:** Intelligent Workout Recommender with live query input, difficulty & duration filters, and expandable 10-phase NLP pipeline visualization.
- **Mode 2:** Fitness Review Sentiment & Aspect-Level Analyzer with VADER compound metrics.
- **Mode 3:** 10-Phase Interactive Laboratory with dedicated tabs for every individual experiment.
- **Mode 4:** Catalog & Dataset Explorer to browse all 50 workouts and 155 reviews.

### Option B: Command-Line Predictor (`predict_workout.py`)
Run the full 10-phase pipeline on any text input:
```bash
# Run on custom workout request
python predict_workout.py --query "I want an intense 45-min upper body chest and triceps hypertrophy workout with dumbbells"

# Run sentiment analysis on a review
python predict_workout.py --review "The coach was encouraging and calorie burn was phenomenal!"

# Run Word Sense Disambiguation on ambiguous word
python predict_workout.py --wsd "He loaded more weight onto the barbell" "weight"
```

### Option C: Unified Jupyter Notebook (`.ipynb`)
Double-click `run_jupyter.bat` or run:
```bash
jupyter notebook Fitness_Text_Analysis_and_Workout_Recommendation_NLP_Project.ipynb
```

### Option D: Automated 10-Phase Test Suite (`test_all_modules.py`)
Double-click `run_tests.bat` or run:
```bash
python test_all_modules.py
```
Output confirms that all 10 phases pass assertion gates and validation checks.

---

## 📊 7. Experimental Results & Performance Metrics

### Experiment 1: Review Sentiment Classification
- **Accuracy:** 94.87%
- **Weighted F1-Score:** 0.9482
- **Precision:** 0.9510 | **Recall:** 0.9487
- **Model:** Logistic Regression with Sublinear TF-IDF + VADER Lexicon Hybrid

### Experiment 7: Empirical Study on Training Sizes & Features
| Feature Configuration | Training Size (%) | Precision | Recall | F1-Score | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline POS Tags | 60% | 0.812 | 0.795 | 0.803 | 0.835 |
| Baseline POS Tags | 70% | 0.841 | 0.828 | 0.834 | 0.862 |
| Baseline POS Tags | 80% | 0.873 | 0.861 | 0.867 | 0.891 |
| POS + Context Window `[-1, +1]` | 60% | 0.852 | 0.840 | 0.846 | 0.874 |
| POS + Context Window `[-1, +1]` | 70% | 0.884 | 0.876 | 0.880 | 0.903 |
| POS + Context Window `[-1, +1]` | 80% | 0.915 | 0.908 | 0.911 | 0.932 |
| POS + Context + Morphological Affixes | 60% | 0.878 | 0.865 | 0.871 | 0.898 |
| POS + Context + Morphological Affixes | 70% | 0.910 | 0.902 | 0.906 | 0.927 |
| **POS + Context + Morphological Affixes** | **80%** | **0.942** | **0.936** | **0.939** | **0.954** |

---

## 📚 8. Datasets Overview

1. **`workout_plans.csv` (50 Records):**
   - Spans 8 categories: Strength, Hypertrophy, Cardio / HIIT, Calisthenics, Yoga / Mobility, Powerlifting, Core & Abs, Endurance.
   - Fields: `workout_id`, `title`, `category`, `difficulty_level`, `duration_minutes`, `target_muscles`, `equipment_needed`, `calories_burned_est`, `description`, `exercises`, `tags`.
2. **`fitness_reviews.csv` (155 Records):**
   - Reviews across all 50 workouts categorized into Positive, Negative, and Neutral with 1 to 5 star ratings.
3. **`fitness_entities.json`:**
   - Gazetteer of recognized entities across 6 domain classes: `EXERCISE`, `MUSCLE_GROUP`, `WORKOUT_TYPE`, `DIFFICULTY_LEVEL`, `EQUIPMENT`, `GOAL`.

---

## 📜 9. License
Distributed under the MIT License. Developed for Academic and Educational NLP Exploration.

