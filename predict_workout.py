"""
Fitness Text Analysis and Intelligent Workout Recommendation System
Interactive Command-Line Predictor & NLP Inspection Tool
"""

import sys
import os
import argparse

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.sentiment import sentiment_analyzer
from modules.preprocessing import tokenize_and_filter, preprocess_text, validate_script
from modules.morphology import analyze_morphology, expand_query_morphologically
from modules.ngram import ngram_model
from modules.pos_tagging import tag_pos
from modules.chunking import extract_chunks
from modules.ner import extract_named_entities
from modules.similarity import workout_recommender
from modules.wsd import wsd_model


def run_full_pipeline(query):
    print("=" * 80)
    print(f"FITNESS NLP PIPELINE EXECUTION FOR QUERY: '{query}'")
    print("=" * 80)

    # 1. Script Validation & Tokenization
    tok_res = tokenize_and_filter(query)
    print(f"\n[1] SCRIPT VALIDATION & TOKENS:")
    print(f"    Valid Latin Script : {tok_res['script_validation']['is_valid']} ({tok_res['script_validation']['primary_script']})")
    print(f"    Raw Word Tokens    : {tok_res['word_tokens']}")
    print(f"    Filtered Tokens    : {tok_res['filtered_tokens']}")

    # 2. Stopwords, Stemming & Lemmatization
    prep_res = preprocess_text(query)
    print(f"\n[2] NORMALIZATION:")
    print(f"    Stopwords Removed  : {prep_res['stopword_removed_tokens']}")
    print(f"    Porter Stemmed     : {prep_res['stemmed_tokens']}")
    print(f"    Lemmatized         : {prep_res['lemmatized_tokens']}")

    # 3. Morphological Query Expansion
    morph_res = expand_query_morphologically(prep_res["lemmatized_tokens"])
    print(f"\n[3] MORPHOLOGICAL EXPANSION:")
    print(f"    Expanded Tokens    : {morph_res['expanded_tokens']}")

    # 4. N-Grams
    ngram_res = ngram_model.analyze_query(query)
    print(f"\n[4] N-GRAM SEQUENCES:")
    print(f"    Bigrams            : {ngram_res['bigrams_raw']}")
    print(f"    Trigrams           : {ngram_res['trigrams_raw']}")

    # 5. POS Tagging
    pos_res = tag_pos(query)
    print(f"\n[5] PART-OF-SPEECH TAGS:")
    for item in pos_res["structured_results"]:
        print(f"    {item['word']:15} -> {item['tag']:6} | {item['category']}")

    # 6. Chunking
    chunk_res = extract_chunks(query)
    print(f"\n[6] PHRASE CHUNKS:")
    for ch in chunk_res["extracted_phrases"]:
        print(f"    [{ch['chunk_type']:15}] : '{ch['phrase']}'")

    # 7. NER & Slot Filling
    ner_res = extract_named_entities(query)
    print(f"\n[7] NAMED ENTITIES & SLOTS:")
    for ent in ner_res["entities"]:
        print(f"    Entity: '{ent['entity']:15}' -> Label: {ent['label']:16} (Source: {ent.get('source', 'Gazetteer')})")
    print(f"    Extracted Slots    : {ner_res['slots']}")

    # 8. WSD Check for Ambiguous Terms
    for word in ["weight", "running", "press", "bar", "rep", "set"]:
        if word in query.lower():
            wsd_res = wsd_model.disambiguate_contextual(query, word)
            print(f"\n[8] WORD SENSE DISAMBIGUATION (Word: '{word}'):")
            print(f"    Predicted Sense    : {wsd_res['predicted_sense']} (Confidence: {wsd_res['confidence']:.2f})")
            print(f"    Sense Definition   : {wsd_res['definition']}")

    # 9. Recommendations
    rec_res = workout_recommender.recommend_workouts(query, top_k=3)
    print(f"\n[9] TOP WORKOUT RECOMMENDATIONS:")
    for i, w in enumerate(rec_res["recommendations"], 1):
        print(f"    {i}. {w['title']}")
        print(f"       Category        : {w['category']} | Level: {w['difficulty_level']} | Duration: {w['duration_minutes']} min")
        print(f"       Target Muscles  : {w['target_muscles']}")
        print(f"       Equipment       : {w['equipment_needed']}")
        print(f"       Match Score     : {w['match_percentage']} (Hybrid: {w['hybrid_score']})")
        print(f"       Matched Terms   : {', '.join(w['matched_keywords'])}")
        print(f"       Exercises       : {'; '.join(w['exercises'][:4])}")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Fitness NLP Predictor & Recommendation Engine")
    parser.add_argument("--query", type=str, help="Workout query to process and recommend")
    parser.add_argument("--review", type=str, help="Review text to classify sentiment")
    parser.add_argument("--wsd", nargs=2, metavar=("SENTENCE", "WORD"), help="Disambiguate ambiguous fitness word in sentence")
    parser.add_argument("--pos", type=str, help="Run POS tagging on text")
    parser.add_argument("--chunk", type=str, help="Extract chunks from text")
    args = parser.parse_args()

    if args.query:
        run_full_pipeline(args.query)
    elif args.review:
        res = sentiment_analyzer.predict(args.review)
        print(f"Review     : {args.review}")
        print(f"Sentiment  : {res['sentiment']} (Confidence: {res['confidence']:.2f})")
        print(f"VADER Compound: {res['vader_scores']['compound']}")
        aspects = sentiment_analyzer.analyze_aspects(args.review)
        if aspects:
            print(f"Aspects    : {aspects}")
    elif args.wsd:
        sentence, word = args.wsd
        res = wsd_model.disambiguate_contextual(sentence, word)
        print(f"Sentence   : {sentence}")
        print(f"Word       : {word}")
        print(f"Sense      : {res['predicted_sense']} (Confidence: {res['confidence']:.2f})")
        print(f"Definition : {res['definition']}")
    elif args.pos:
        res = tag_pos(args.pos)
        for r in res["structured_results"]:
            print(f"{r['word']:15} -> {r['tag']:6} | {r['category']}")
    elif args.chunk:
        res = extract_chunks(args.chunk)
        for p in res["extracted_phrases"]:
            print(f"[{p['chunk_type']}] : {p['phrase']}")
    else:
        # Default demo query
        demo_query = "Looking for a 30-min high intensity cardio fat loss workout with no equipment"
        run_full_pipeline(demo_query)


if __name__ == "__main__":
    main()

