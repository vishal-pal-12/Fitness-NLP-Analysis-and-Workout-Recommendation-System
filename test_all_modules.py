"""
Comprehensive Test Suite for Fitness NLP Mini Project
Verifies all 10 NLP phases end-to-end.
"""

import sys
import os

# Ensure UTF-8 output in Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_pipeline():
    print("=" * 80)
    print("VERIFYING ALL 10 NLP PHASES FOR FITNESS TEXT ANALYSIS & RECOMMENDATION SYSTEM")
    print("=" * 80)

    # -------------------------------------------------------------
    # PHASE 1: Sentiment Analysis
    # -------------------------------------------------------------
    print("\n[TEST 1/10] Phase 1 — Sentiment Analysis of Fitness Reviews...")
    from modules.sentiment import sentiment_analyzer
    metrics = sentiment_analyzer.train_and_evaluate()
    print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

    pos_review = "This HIIT session completely energized me and the trainer was fantastic!"
    pred_pos = sentiment_analyzer.predict(pos_review)
    print(f"  Review: '{pos_review}' -> Sentiment: {pred_pos['sentiment']} (Conf: {pred_pos['confidence']:.2f})")
    assert pred_pos["sentiment"] == "Positive", f"Expected Positive, got {pred_pos['sentiment']}"

    neg_review = "Terrible routine, the tempo is too fast and caused terrible knee and back pain."
    pred_neg = sentiment_analyzer.predict(neg_review)
    print(f"  Review: '{neg_review}' -> Sentiment: {pred_neg['sentiment']} (Conf: {pred_neg['confidence']:.2f})")
    assert pred_neg["sentiment"] == "Negative", f"Expected Negative, got {pred_neg['sentiment']}"

    aspects = sentiment_analyzer.analyze_aspects(neg_review)
    print(f"  Aspects detected: {list(aspects.keys())}")
    assert "Joint Safety & Form" in aspects
    print("  -> Phase 1 PASSED.")

    # -------------------------------------------------------------
    # PHASE 2: Tokenization & Script Validation
    # -------------------------------------------------------------
    print("\n[TEST 2/10] Phase 2 — Tokenization, Filtration & Script Validation...")
    from modules.preprocessing import tokenize_and_filter, validate_script
    sample_query = "Find a 30-min HIIT workout with 3x10 squats and 50kg deadlifts!"
    tok_res = tokenize_and_filter(sample_query)
    print(f"  Filtered Tokens: {tok_res['filtered_tokens']}")
    print(f"  Script Valid: {tok_res['script_validation']['is_valid']} ({tok_res['script_validation']['primary_script']})")
    assert tok_res["script_validation"]["is_valid"] is True
    assert "3x10" in tok_res["filtered_tokens"] or "squats" in tok_res["filtered_tokens"]

    # Foreign script test (Devanagari / Hindi)
    hindi_check = validate_script("व्यायाम और कसरत")
    print(f"  Foreign Script Test ('व्यायाम और कसरत') -> Valid Latin: {hindi_check['is_valid']}, Script: {hindi_check['primary_script']}")
    assert hindi_check["is_valid"] is False
    print("  -> Phase 2 PASSED.")

    # -------------------------------------------------------------
    # PHASE 3: Stopwords, Stemming & Lemmatization
    # -------------------------------------------------------------
    print("\n[TEST 3/10] Phase 3 — Stop Word Removal, Stemming & Lemmatization...")
    from modules.preprocessing import preprocess_text
    prep_res = preprocess_text(sample_query)
    print(f"  Original Tokens: {prep_res['original_tokens']}")
    print(f"  Stopwords Removed: {prep_res['stopword_removed_tokens']}")
    print(f"  Porter Stemmed: {prep_res['stemmed_tokens']}")
    print(f"  Lemmatized: {prep_res['lemmatized_tokens']}")
    print(f"  Reduction Rate: {prep_res['reduction_rate_pct']}%")
    assert len(prep_res["lemmatized_tokens"]) > 0
    print("  -> Phase 3 PASSED.")

    # -------------------------------------------------------------
    # PHASE 4: Morphological Analysis & Word Generation
    # -------------------------------------------------------------
    print("\n[TEST 4/10] Phase 4 — Morphological Analysis and Word Generation...")
    from modules.morphology import analyze_morphology, expand_query_morphologically
    morph1 = analyze_morphology("training")
    print(f"  Word: 'training' -> Root: {morph1['root']}, Suffix: {morph1['suffix']}, Forms: {morph1['generated_forms']['present_3rd_person']}, {morph1['generated_forms']['past_tense']}")
    assert morph1["root"] == "train"

    expansion = expand_query_morphologically(["squatting", "press"])
    print(f"  Morphological Query Expansion: {expansion['expanded_tokens']}")
    assert "squats" in expansion["expanded_tokens"] or "squat" in expansion["expanded_tokens"]
    print("  -> Phase 4 PASSED.")

    # -------------------------------------------------------------
    # PHASE 5: N-Gram Language Model
    # -------------------------------------------------------------
    print("\n[TEST 5/10] Phase 5 — Unigram, Bigram, and Trigram Models...")
    from modules.ngram import ngram_model
    query_ngram = "weight loss workout plan"
    ngram_res = ngram_model.analyze_query(query_ngram)
    print(f"  Bigrams: {ngram_res['bigrams_raw']}")
    print(f"  Trigrams: {ngram_res['trigrams_raw']}")

    suggestions = ngram_model.suggest_next_word(["weight"])
    print(f"  Next word after 'weight': {[s['word'] for s in suggestions]}")
    assert len(ngram_res["bigrams_raw"]) == 3
    print("  -> Phase 5 PASSED.")

    # -------------------------------------------------------------
    # PHASE 6: POS Tagging
    # -------------------------------------------------------------
    print("\n[TEST 6/10] Phase 6 — Part-of-Speech (POS) Tagging...")
    from modules.pos_tagging import tag_pos
    pos_res = tag_pos("Lift heavy dumbbells for intense chest hypertrophy")
    for item in pos_res["structured_results"][:4]:
        print(f"  {item['word']:12} -> {item['tag']:5} ({item['category']})")
    assert any(item["category"].startswith("Noun") for item in pos_res["structured_results"])
    assert any(item["category"].startswith("Verb") for item in pos_res["structured_results"])
    print("  -> Phase 6 PASSED.")

    # -------------------------------------------------------------
    # PHASE 7: Chunking & Empirical Study
    # -------------------------------------------------------------
    print("\n[TEST 7/10] Phase 7 — Chunking & Empirical Study on Training Sizes...")
    from modules.chunking import extract_chunks, analyze_training_size_and_features
    chunk_res = extract_chunks("Beginner workout plan for upper body strength training")
    print(f"  Extracted Phrases: {[p['phrase'] for p in chunk_res['extracted_phrases']]}")
    assert len(chunk_res["extracted_phrases"]) > 0

    study = analyze_training_size_and_features()
    print("  Training Size Study Results:")
    for row in study["comparison_table"]:
        print(f"    {row['Feature Set'][:20]:20} | Size: {row['Training Size (%)']:3}% | Acc: {row['Accuracy']:.3f} | F1: {row['F1 Score']:.3f}")
    assert study["best_configuration"]["Training Size (%)"] == 80
    print("  -> Phase 7 PASSED.")

    # -------------------------------------------------------------
    # PHASE 8: Named Entity Recognition
    # -------------------------------------------------------------
    print("\n[TEST 8/10] Phase 8 — Named Entity Recognition (NER)...")
    from modules.ner import extract_named_entities
    ner_query = "I want a 45-min HIIT workout targeting chest and biceps using dumbbells for fat loss"
    ner_res = extract_named_entities(ner_query)
    for ent in ner_res["entities"]:
        print(f"  Entity: '{ent['entity']}' -> Label: {ent['label']} (Source: {ent.get('source')})")
    slots = ner_res["slots"]
    print(f"  Extracted Slots: {slots}")
    assert len(slots["muscle_groups"]) > 0 or len(slots["workout_types"]) > 0
    assert slots["duration_minutes"] == 45
    print("  -> Phase 8 PASSED.")

    # -------------------------------------------------------------
    # PHASE 9: Text Similarity & Workout Recommendation
    # -------------------------------------------------------------
    print("\n[TEST 9/10] Phase 9 — Text Similarity & Intelligent Recommendation...")
    from modules.similarity import workout_recommender
    rec_res = workout_recommender.recommend_workouts(
        "30 minute bodyweight high intensity cardio fat loss at home",
        top_k=3
    )
    print("  Top Recommendations:")
    for idx, w in enumerate(rec_res["recommendations"], 1):
        print(f"    {idx}. {w['title']} ({w['category']}) - Match: {w['match_percentage']} (Score: {w['hybrid_score']})")
    assert len(rec_res["recommendations"]) > 0
    print("  -> Phase 9 PASSED.")

    # -------------------------------------------------------------
    # PHASE 10: Word Sense Disambiguation
    # -------------------------------------------------------------
    print("\n[TEST 10/10] Phase 10 — Word Sense Disambiguation (WSD)...")
    from modules.wsd import wsd_model

    wsd_test1 = "I need a strict calorie diet and cardio to lose weight."
    res1 = wsd_model.disambiguate_contextual(wsd_test1, "weight")
    print(f"  Sentence: '{wsd_test1}' -> Predicted Sense: {res1['predicted_sense']} (Conf: {res1['confidence']:.2f})")
    assert res1["predicted_sense"] == "BODY_WEIGHT", f"Expected BODY_WEIGHT, got {res1['predicted_sense']}"

    wsd_test2 = "He added more weight to the barbell for the heavy bench press."
    res2 = wsd_model.disambiguate_contextual(wsd_test2, "weight")
    print(f"  Sentence: '{wsd_test2}' -> Predicted Sense: {res2['predicted_sense']} (Conf: {res2['confidence']:.2f})")
    assert res2["predicted_sense"] == "GYM_LOAD", f"Expected GYM_LOAD, got {res2['predicted_sense']}"

    wsd_test3 = "Running five miles on the track boosts stamina and burns calories."
    res3 = wsd_model.disambiguate_contextual(wsd_test3, "running")
    print(f"  Sentence: '{wsd_test3}' -> Predicted Sense: {res3['predicted_sense']} (Conf: {res3['confidence']:.2f})")
    assert res3["predicted_sense"] == "CARDIO_EXERCISE", f"Expected CARDIO_EXERCISE, got {res3['predicted_sense']}"
    print("  -> Phase 10 PASSED.")

    print("\n" + "=" * 80)
    print("ALL 10 NLP PHASES VERIFIED AND FULLY OPERATIONAL!")
    print("=" * 80)


if __name__ == "__main__":
    test_pipeline()

