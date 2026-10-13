# 📄 College Mini-Project Technical Report

## Fitness Text Analysis and Intelligent Workout Recommendation System using Natural Language Processing (NLP)

**Academic Level:** Mini-Project / Final Course Laboratory Submission  
**Course:** Natural Language Processing (NLP)  
**Author:** Vishal Pal  
**Status:** Completed, Verified & Operational  

---

### Executive Abstract
Digital fitness platforms, mobile workout apps, and gym management systems handle enormous volumes of unstructured natural language text, ranging from user workout inquiries and training goals to workout plan descriptions and patron feedback. Traditional keyword lookup systems struggle to interpret grammatical inflections, polysemous terms (such as *weight* or *running*), compound fitness expressions (such as *high-intensity interval training*), and user sentiment.

This project develops an end-to-end **Fitness Text Analysis and Intelligent Workout Recommendation System** integrating 10 prescribed curriculum experiments into a unified pipeline. The system combines:
1. Lexicon and Machine Learning Sentiment Analysis on fitness reviews.
2. Tokenization, filtration, and multi-script Unicode validation.
3. Stopword elimination and comparative stemming/lemmatization.
4. Morphological decomposition and inflectional word generation.
5. Unigram, Bigram, and Trigram language modeling with Pointwise Mutual Information (PMI) collocations.
6. Part-of-Speech (POS) tagging with imperative verb handling and fitness semantic mapping.
7. Phrase chunking and an empirical study on varying training sizes (60%, 70%, 80%).
8. Named Entity Recognition (NER) with domain gazetteers for goal and muscle slot-filling.
9. Multi-factor hybrid text similarity ranking (TF-IDF Cosine + Jaccard + entity bonuses).
10. Word Sense Disambiguation (WSD) resolving context-dependent polysemous fitness vocabulary.

The system is delivered with an interactive Streamlit web dashboard, a command-line interface, a verified test suite, and an annotated Jupyter Notebook.

---

### 1. Introduction & Problem Statement
Finding an appropriate workout plan from a large collection of exercise descriptions is difficult for users without an intelligent recommendation engine. Users express their preferences in natural language (e.g., *"I want a 30-minute cardio workout at home to burn fat without weights"*). 

To solve this, the 10 experiments are structured into an interconnected sequence where the output of earlier linguistic stages enriches downstream recommendations:

```
[Raw User Query / Review]
         │
         ▼
[Script Validation & Tokenization] (Exp 2)
         │
         ▼
[Stopword Filtration & Lemmatization] (Exp 3)
         │
         ▼
[Morphological Analysis & Query Expansion] (Exp 4)
         │
         ▼
[POS Tagging & Semantic Mapping] (Exp 6)
         │
         ▼
[Chunking & Phrase Extraction] (Exp 7)
         │
         ▼
[Named Entity Recognition & Slot Filling] (Exp 8)
         │
         ▼
[Word Sense Disambiguation] (Exp 10)
         │
         ▼
[N-Gram Language Model & Query Autocomplete] (Exp 5)
         │
         ▼
[Multi-Factor Semantic Similarity & Ranking] (Exp 9)
         │
         ▼
[Final Ranked Recommendations + Review Sentiment Analysis] (Exp 1)
```

---

### 2. Experiment-by-Experiment Technical Implementation

#### Experiment 1: Sentiment Analysis of Fitness Reviews
- **Problem Statement:** Fitness applications receive user reviews and feedback about exercises, workout plans, trainers, and fitness programs. This experiment classifies user opinions into Positive, Negative, or Neutral.
- **Approach:** Hybrid classification leveraging TF-IDF unigram and bigram representation trained on Logistic Regression combined with VADER (Valence Aware Dictionary and sEntiment Reasoner) polarity scoring. Aspect-based sentiment analysis is implemented to detect user sentiment across five key categories: Trainer/Coach, Intensity/Difficulty, Joint Safety/Form, Equipment, and Progress/Results.
- **Results:** 94.87% test accuracy and 0.9482 weighted F1-score across 155 reviews.

#### Experiment 2: Tokenization, Filtration, and Script Validation of Fitness Text
- **Problem Statement:** Fitness text contains words, punctuation, numbers, and special characters. This experiment performs tokenization, filtration, and script validation to clean fitness text and prepare valid tokens.
- **Approach:** Multi-strategy tokenization (NLTK Treebank, WordPunct, and Regex tokenizers), contraction expansion (*can't* -> *cannot*), filtration of punctuation and non-informative symbols while preserving fitness patterns like *3x10*, *50kg*, *45min*. Script validation inspects Unicode character categories and confirms Latin/English integrity, flagging foreign scripts (e.g. Devanagari) or emoji noise.
- **Results:** Clean token streams extracted with robust non-English rejection.

#### Experiment 3: Stop Word Removal, Stemming, and Lemmatization of Fitness Text
- **Problem Statement:** Common words and morphological variants (*exercise*, *exercises*, *exercising*) clutter raw text. This experiment normalizes text through stopword removal, stemming, and lemmatization.
- **Approach:** Fitness domain-customized stopword removal (preserving directional and intensity terms like *up*, *down*, *more*, *not*). Comparative analysis between PorterStemmer, LancasterStemmer, and SnowballStemmer against POS-guided WordNet Lemmatization.
- **Results:** 27.3% to 45% token reduction rate while preserving critical fitness semantics.

#### Experiment 4: Morphological Analysis and Word Generation of Fitness Terms
- **Problem Statement:** Fitness terms contain inflectional and derivational variants (*train*, *trains*, *training*, *trained*). This experiment identifies roots, prefixes, suffixes, and generates related word forms.
- **Approach:** Prefix detection (*pre-*, *post-*, *over-*, *under-*, *re-*), suffix analysis (*-ing*, *-ed*, *-er*, *-tion*), morphological root extraction, and programmatic word generation for third-person singular, present participle, past tense, and agent nouns. Morphological query expansion broadens search terms so queries like *squatting* match *squat* and *squats*.
- **Results:** Complete inflectional paradigms generated for fitness verbs and nouns.

#### Experiment 5: Unigram, Bigram, and Trigram Models for Fitness Text
- **Problem Statement:** Users describe goals using multi-word phrases (*weight loss workout*, *muscle building*). This experiment implements N-gram models to identify frequent collocations and word sequences.
- **Approach:** Unigram, Bigram, and Trigram language modeling over the fitness corpus. Pointwise Mutual Information (PMI) identifies domain collocations (*weight loss*, *bench press*, *interval training*). Next-word prediction suggests continuations, and Laplace Add-1 smoothing evaluates sequence perplexity.
- **Results:** Successfully identified top multi-word fitness phrases and enabled auto-completion.

#### Experiment 6: Part-of-Speech (POS) Tagging of Fitness Text
- **Problem Statement:** Grammatical roles indicate whether a word represents an action, muscle, or difficulty. This experiment tags words with grammatical categories.
- **Approach:** Penn Treebank tagging with specialized handling for sentence-initial imperative verbs (*Lift*, *Run*, *Squat*). Tags are mapped to fitness semantic categories: Action Verbs (`VB*`), Target Muscles & Equipment (`NN*`), and Intensity Modifiers (`JJ*`).
- **Results:** Accurate grammatical distribution and extraction of key action verbs and target muscles.

#### Experiment 7: Chunking and Phrase Extraction from Fitness Text
- **Problem Statement:** Extracting meaningful phrases (*beginner workout plan*, *upper body strength training*) helps identify workout goals and levels.
- **Approach:** RegexpParser grammar targeting Noun Phrases (`FITNESS_NP`), Verb Phrases (`WORKOUT_VP`), Intensity Chunks, and Duration Chunks. Produces IOB tags. Includes an academic empirical study evaluating Chunking F1-score across 60%, 70%, and 80% training splits.
- **Results:** Chunking F1-score improved from 80.3% (at 60% with baseline POS) to 93.9% (at 80% with context windows and morphological affixes).

#### Experiment 8: Named Entity Recognition (NER) for Fitness Information Extraction
- **Problem Statement:** Fitness text contains entities such as exercise names, body parts, workout types, and equipment.
- **Approach:** Hybrid NER combining NLTK Maximum Entropy chunking with a comprehensive domain gazetteer covering 6 entity classes: `EXERCISE`, `MUSCLE_GROUP`, `WORKOUT_TYPE`, `EQUIPMENT`, `DIFFICULTY_LEVEL`, and `DURATION`. Extracted entities populate structured recommendation slots.
- **Results:** High-precision entity extraction and slot filling for complex queries.

#### Experiment 9: Text Similarity for Intelligent Workout Recommendation
- **Problem Statement:** Selecting suitable workout plans manually is difficult. This experiment calculates text similarity between user requests and available workout descriptions.
- **Approach:** Multi-factor hybrid scoring function:
  $$\text{Score} = 0.50 \cdot \text{TF-IDF Cosine} + 0.15 \cdot \text{Jaccard} + \text{Bonus}_{\text{Goal}} + \text{Bonus}_{\text{Muscles}} + \text{Bonus}_{\text{Equipment}}$$
  Evaluated against 50 diverse workout plans with duration and difficulty filters.
- **Results:** Accurately ranks the most relevant workouts with transparent explanation of matched features.

#### Experiment 10: Word Sense Disambiguation (WSD) for Fitness Text
- **Problem Statement:** Words like *weight* (body weight vs gym load) or *running* (cardio vs program execution) have multiple meanings. This experiment resolves context-dependent word senses.
- **Approach:** WordNet Lesk algorithm compared against Contextual Cosine Similarity using curated domain sense profiles for ambiguous terms (*weight*, *running*, *press*, *bar*, *rep*, *set*).
- **Results:** Correct sense prediction with >90% confidence across test cases.

---

### 3. Empirical Results & Performance Evaluation

#### Sentiment Classification Metrics (Exp 1)
- **Accuracy:** 94.87%
- **F1-Score (Weighted):** 0.9482
- **Precision:** 0.9510
- **Recall:** 0.9487

#### Chunking Empirical Study (Exp 7)
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

### 4. Software Architecture & Deliverables

1. **Modular Codebase (`modules/`):** 10 modular Python files containing reusable classes and helper functions.
2. **Interactive Streamlit Web Dashboard (`app.py`):** Multi-mode user interface with full interactive testing for every experiment, review analysis, and recommendation generation.
3. **Command-Line Predictor (`predict_workout.py`):** Terminal utility for pipeline execution and debugging.
4. **Automated Verification Suite (`test_all_modules.py`):** Rigorous assertion tests ensuring 10/10 experiments pass cleanly.
5. **Jupyter Notebook (`.ipynb`):** Complete documented notebook with code cells, theory, and execution traces.
6. **Execution Scripts (`.bat`):** Single-click launchers for web app, tests, and notebook.

---

### 5. Conclusion & Future Directions
This mini-project successfully bridges theoretical NLP concepts with practical domain-specific engineering. Every experiment fulfills its academic problem statement while serving as an active component in the recommendation system. Future extensions could incorporate dense neural embeddings (Sentence-BERT), user collaborative filtering, and multilingual translation for global fitness apps.

