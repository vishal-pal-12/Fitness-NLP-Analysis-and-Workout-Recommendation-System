"""
Fitness Text Analysis and Intelligent Workout Recommendation System Using NLP
College NLP Mini Project integrating Experiments 1 through 10.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 and project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.sentiment import sentiment_analyzer
from modules.preprocessing import tokenize_and_filter, preprocess_text, validate_script
from modules.morphology import analyze_morphology, expand_query_morphologically, generate_word_forms
from modules.ngram import ngram_model
from modules.pos_tagging import tag_pos
from modules.chunking import extract_chunks, analyze_training_size_and_features
from modules.ner import extract_named_entities
from modules.similarity import workout_recommender
from modules.wsd import wsd_model, AMBIGUOUS_FITNESS_WORDS

# Streamlit Page Config
st.set_page_config(
    page_title="Fitness NLP & Intelligent Workout Recommendation System",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #047857 100%);
        color: #FFFFFF;
        padding: 2rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 6px 20px rgba(4, 120, 87, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.92;
        margin-bottom: 0;
        line-height: 1.45;
    }
    
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        color: #047857;
        font-weight: 800;
    }
    .metric-card p {
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
    }

    .workout-card {
        border: 1px solid #E2E8F0;
        border-left: 6px solid #059669;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        background-color: #FFFFFF;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .workout-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    
    .match-pill {
        display: inline-block;
        background: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid #BBF7D0;
    }
    
    .phase-badge {
        display: inline-block;
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 0.3rem 0.75rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    
    .tag-chip {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        margin-right: 4px;
        margin-bottom: 4px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
df_workouts = workout_recommender.df
df_reviews = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "fitness_reviews.csv"))

# Sidebar Navigation
st.sidebar.markdown("### 🏋️ Fitness NLP Portal")
st.sidebar.caption("Intelligent Workout Recommendation System")

app_mode = st.sidebar.radio(
    "Navigation Mode:",
    [
        "💪 Workout Recommender & 10-Phase Pipeline",
        "💬 Review Sentiment & Aspect Analyzer",
        "🔬 NLP 10-Phase Interactive Laboratory",
        "📋 Catalog & Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Syllabus Experiments Covered:**
1. **Exp 1:** Sentiment Analysis of Reviews
2. **Exp 2:** Tokenization, Filtration & Script Validation
3. **Exp 3:** Stopwords, Stemming & Lemmatization
4. **Exp 4:** Morphological Analysis & Word Generation
5. **Exp 5:** Unigram, Bigram, Trigram Models
6. **Exp 6:** Part-of-Speech (POS) Tagging
7. **Exp 7:** Chunking & Empirical Study (60/70/80%)
8. **Exp 8:** Named Entity Recognition (NER) & Slots
9. **Exp 9:** Text Similarity & Content-Based Recommendation
10. **Exp 10:** Word Sense Disambiguation (WSD)
""")

# ==============================================================================
# MODE 1: WORKOUT RECOMMENDER & 10-PHASE PIPELINE
# ==============================================================================
if app_mode == "💪 Workout Recommender & 10-Phase Pipeline":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">💪 Intelligent Workout Recommendation System</div>
        <div class="hero-subtitle">Input your natural language fitness goal or pick a routine to execute all 10 NLP phases—from script validation and morphology to chunking, NER slot-filling, and hybrid semantic similarity.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        preset_queries = [
            "I want a 30-min high intensity cardio fat loss workout at home with no equipment",
            "Heavy barbell bench press and overhead press for upper body chest hypertrophy",
            "Looking for beginner low-impact mobility exercises to relieve lower back pain",
            "45-minute intense pull workout focusing on lats and bicep curls with dumbbells",
            "Starting strength 5x5 compound powerlifting routine for squats and deadlifts",
            "Runner stamina builder with sprint intervals and aerobic conditioning"
        ]
        user_query = st.selectbox("Select a sample fitness query or type below:", preset_queries)
        custom_query = st.text_input("Or enter your custom workout goal / request:", value="")
        active_query = custom_query.strip() if custom_query.strip() else user_query

    with col2:
        top_k = st.slider("Top Recommendations:", 1, 10, 5)
        diff_filter = st.selectbox("Filter Difficulty:", ["All", "Beginner", "Intermediate", "Advanced"])
        duration_filter = st.slider("Max Duration (mins):", 10, 90, 90, step=5)

    if st.button("🚀 Analyze Query & Generate Recommendations", type="primary", use_container_width=True):
        st.session_state["active_query"] = active_query

    current_query = st.session_state.get("active_query", active_query)

    # Execute Recommendation
    rec_results = workout_recommender.recommend_workouts(
        current_query,
        top_k=top_k,
        difficulty_filter=diff_filter if diff_filter != "All" else None,
        max_duration=duration_filter if duration_filter < 90 else None
    )

    st.markdown("### 🏆 Top Recommended Workout Plans")
    if not rec_results["recommendations"]:
        st.warning("No workouts matched the specific filter criteria. Try relaxing difficulty or duration filters.")
    else:
        for idx, w in enumerate(rec_results["recommendations"], 1):
            with st.container():
                st.markdown(f"""
                <div class="workout-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:1.25rem; font-weight:700; color:#0F172A;">#{idx}. {w['title']}</span>
                        <span class="match-pill">Match: {w['match_percentage']}</span>
                    </div>
                    <div style="margin-bottom:8px;">
                        <span class="tag-chip"><b>Category:</b> {w['category']}</span>
                        <span class="tag-chip"><b>Level:</b> {w['difficulty_level']}</span>
                        <span class="tag-chip"><b>Duration:</b> {w['duration_minutes']} min</span>
                        <span class="tag-chip"><b>Calories:</b> ~{w['calories_burned_est']} kcal</span>
                        <span class="tag-chip"><b>Equipment:</b> {w['equipment_needed']}</span>
                    </div>
                    <p style="color:#334155; margin-bottom:8px; font-size:0.95rem;">{w['description']}</p>
                    <div style="font-size:0.88rem; color:#475569; margin-bottom:6px;">
                        <b>Target Muscles:</b> {w['target_muscles']}
                    </div>
                    <div style="font-size:0.88rem; color:#475569; margin-bottom:6px;">
                        <b>Key Exercises:</b> {', '.join(w['exercises'][:5])}
                    </div>
                    <div style="font-size:0.82rem; color:#059669; font-weight:600;">
                        <b>Matched Features:</b> {', '.join(w['matched_keywords']) if w['matched_keywords'] else 'Contextual TF-IDF match'} (Hybrid Score: {w['hybrid_score']:.4f})
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Deep Linguistic Pipeline Inspection Accordion
    st.markdown("---")
    st.markdown("### 🔍 10-Phase NLP Linguistic Execution Breakdown")
    with st.expander("Click to view full 10-Phase NLP Execution Trace on this Query", expanded=True):
        t1, t2, t3, t4, t5 = st.tabs(["1-3: Preprocessing", "4-5: Morphology & N-Grams", "6-7: POS & Chunks", "8: NER & Slots", "10: WSD"])

        with t1:
            st.markdown("#### Phase 2: Tokenization & Script Validation")
            tok_info = tokenize_and_filter(current_query)
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Script Validation:**", tok_info["script_validation"])
            with c2:
                st.write("**Filtered Tokens:**", tok_info["filtered_tokens"])

            st.markdown("#### Phase 3: Stopword Removal, Stemming & Lemmatization")
            prep_info = preprocess_text(current_query)
            st.dataframe(pd.DataFrame(prep_info["comparison_table"]), use_container_width=True)
            st.caption(f"Token Reduction Rate: {prep_info['reduction_rate_pct']}%")

        with t2:
            st.markdown("#### Phase 4: Morphological Analysis & Query Expansion")
            morph_exp = expand_query_morphologically(prep_info["lemmatized_tokens"])
            st.write("**Expanded Tokens for Multi-Inflection Search:**", morph_exp["expanded_tokens"])

            st.markdown("#### Phase 5: N-Gram Language Model")
            ng_info = ngram_model.analyze_query(current_query)
            st.write("**Bigrams:**", ng_info["bigrams_raw"])
            st.write("**Trigrams:**", ng_info["trigrams_raw"])
            suggestions = ngram_model.suggest_next_word(prep_info["lemmatized_tokens"][-1:] if prep_info["lemmatized_tokens"] else [])
            st.write("**Next-Token Suggestions:**", suggestions)

        with t3:
            st.markdown("#### Phase 6: Part-of-Speech Tagging")
            pos_info = tag_pos(current_query)
            st.dataframe(pd.DataFrame(pos_info["structured_results"]), use_container_width=True)

            st.markdown("#### Phase 7: Chunking & Phrase Extraction")
            chk_info = extract_chunks(current_query)
            for p in chk_info["extracted_phrases"]:
                st.info(f"**[{p['chunk_type']}]** : {p['phrase']}")

        with t4:
            st.markdown("#### Phase 8: Named Entity Recognition & Slot Filling")
            ner_info = extract_named_entities(current_query)
            st.json(ner_info["slots"])
            st.write("**All Extracted Entities:**", ner_info["entities"])

        with t5:
            st.markdown("#### Phase 10: Word Sense Disambiguation (WSD)")
            detected_wsd = False
            for w in AMBIGUOUS_FITNESS_WORDS.keys():
                if w in current_query.lower():
                    detected_wsd = True
                    wsd_res = wsd_model.disambiguate_contextual(current_query, w)
                    st.success(f"**Word:** '{w}' | **Predicted Sense:** `{wsd_res['predicted_sense']}` (Confidence: {wsd_res['confidence']:.2f})")
                    st.write(f"**Definition:** {wsd_res['definition']}")
                    st.write(f"**Context Keywords Matched:** {wsd_res['matched_context_keywords']}")
            if not detected_wsd:
                st.info("No standard ambiguous words (weight, running, press, bar, rep, set) detected in this specific query.")

# ==============================================================================
# MODE 2: REVIEW SENTIMENT & ASPECT ANALYZER (EXP 1)
# ==============================================================================
elif app_mode == "💬 Review Sentiment & Aspect Analyzer":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">💬 Sentiment Analysis of Fitness Reviews (Exp 1)</div>
        <div class="hero-subtitle">Classify user feedback on workout plans, exercise difficulty, and trainers into Positive, Negative, or Neutral with aspect-level breakdown and VADER polarity scoring.</div>
    </div>
    """, unsafe_allow_html=True)

    metrics = sentiment_analyzer.metrics or sentiment_analyzer.train_and_evaluate()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><h3>{metrics['accuracy']*100:.1f}%</h3><p>MODEL ACCURACY</p></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><h3>{metrics['f1_score']*100:.1f}%</h3><p>WEIGHTED F1-SCORE</p></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><h3>{metrics['train_samples']}</h3><p>TRAINING REVIEWS</p></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><h3>{metrics['test_samples']}</h3><p>TESTING REVIEWS</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Test Any Fitness Review")

    sample_reviews = [
        "Incredible chest pump! The progressive overload scheme with barbell press gave me noticeable growth.",
        "Terrible routine. Way too fast, poor cues, and caused awful knee inflammation and lower back pain.",
        "Decent workout with average intensity. Good if you have only 30 minutes, but nothing special.",
        "The coach was very encouraging and the core intervals kept my heart rate in the fat-burning zone!"
    ]
    selected_sample = st.selectbox("Pick a sample review or type below:", sample_reviews)
    custom_review = st.text_area("Review text:", value=selected_sample, height=100)

    if st.button("Evaluate Sentiment", type="primary"):
        res = sentiment_analyzer.predict(custom_review)
        aspects = sentiment_analyzer.analyze_aspects(custom_review)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            sent = res["sentiment"]
            color = "#10B981" if sent == "Positive" else ("#EF4444" if sent == "Negative" else "#F59E0B")
            st.markdown(f"""
            <div style="background-color:#F8FAFC; border:2px solid {color}; border-radius:10px; padding:1.2rem; margin-top:1rem;">
                <h3 style="color:{color}; margin:0;">Predicted Sentiment: {sent.upper()}</h3>
                <p style="margin:5px 0; color:#475569;"><b>Confidence:</b> {res['confidence']*100:.1f}%</p>
                <p style="margin:5px 0; color:#475569;"><b>VADER Compound Score:</b> {res['vader_scores']['compound']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("**Class Probability Distribution:**")
            st.progress(res["probabilities"].get("Positive", 0.0), text=f"Positive: {res['probabilities'].get('Positive', 0.0)*100:.1f}%")
            st.progress(res["probabilities"].get("Negative", 0.0), text=f"Negative: {res['probabilities'].get('Negative', 0.0)*100:.1f}%")
            st.progress(res["probabilities"].get("Neutral", 0.0), text=f"Neutral: {res['probabilities'].get('Neutral', 0.0)*100:.1f}%")

        with col_b:
            st.write("**Aspect Sentiment Analysis:**")
            if aspects:
                for asp, data in aspects.items():
                    st.write(f"- **{asp}:** `{data['sentiment']}` (keywords: {', '.join(data['matched_keywords'])})")
            else:
                st.caption("No specific domain aspect keywords detected.")

            st.write("**Influential Terms:**", res["influential_terms"])

# ==============================================================================
# MODE 3: NLP 10-PHASE INTERACTIVE LABORATORY
# ==============================================================================
elif app_mode == "🔬 NLP 10-Phase Interactive Laboratory":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔬 NLP 10-Phase Interactive Laboratory</div>
        <div class="hero-subtitle">Dedicated interactive sandboxes and rigorous empirical evaluations for all 10 syllabus experiments.</div>
    </div>
    """, unsafe_allow_html=True)

    tab_names = [
        "Exp 1: Sentiment",
        "Exp 2: Tokenization & Script",
        "Exp 3: Stopwords & Stems",
        "Exp 4: Morphology",
        "Exp 5: N-Grams",
        "Exp 6: POS Tagging",
        "Exp 7: Chunking Study",
        "Exp 8: NER Slots",
        "Exp 9: Similarity",
        "Exp 10: WSD"
    ]
    tabs = st.tabs(tab_names)

    # Exp 1
    with tabs[0]:
        st.subheader("Experiment 1: Sentiment Analysis of Fitness Reviews")
        st.write("Confusion Matrix on Test Split:")
        m = sentiment_analyzer.metrics
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap="Greens", xticklabels=m["labels"], yticklabels=m["labels"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # Exp 2
    with tabs[1]:
        st.subheader("Experiment 2: Tokenization, Filtration & Script Validation")
        test_txt = st.text_input("Enter text (try English, numbers like 3x10, or Hindi like व्यायाम):", "Find a 30-min HIIT workout with 3x10 squats")
        res2 = tokenize_and_filter(test_txt)
        st.write("**Script Validation:**", res2["script_validation"])
        st.write("**Filtered Tokens:**", res2["filtered_tokens"])
        st.write("**Regex Tokens:**", res2["regex_tokens"])

    # Exp 3
    with tabs[2]:
        st.subheader("Experiment 3: Stop Word Removal, Stemming & Lemmatization")
        test_txt3 = st.text_input("Text for normalization:", "The trainers are exercising and teaching weight loss exercises")
        res3 = preprocess_text(test_txt3)
        st.dataframe(pd.DataFrame(res3["comparison_table"]), use_container_width=True)

    # Exp 4
    with tabs[3]:
        st.subheader("Experiment 4: Morphological Analysis & Word Generation")
        test_word = st.selectbox("Select fitness root or term:", ["train", "lift", "squat", "press", "stretch", "recover", "burn", "build"])
        res4 = analyze_morphology(test_word)
        st.json(res4)

    # Exp 5
    with tabs[4]:
        st.subheader("Experiment 5: Unigram, Bigram, and Trigram Models")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("**Top Fitness Bigram Collocations (PMI):**")
            colls = ngram_model.get_top_fitness_collocations(8)
            st.write(colls["top_bigram_collocations"])
        with col_c2:
            st.write("**Top Trigram Collocations:**")
            st.write(colls["top_trigram_collocations"])
        prompt_words = st.text_input("Context words for next-token suggestion:", "weight")
        st.write(ngram_model.suggest_next_word(prompt_words.split()))

    # Exp 6
    with tabs[5]:
        st.subheader("Experiment 6: Part-of-Speech (POS) Tagging")
        test_pos_txt = st.text_input("Sentence for POS analysis:", "Athletes lift heavy dumbbells for intense chest hypertrophy")
        res6 = tag_pos(test_pos_txt)
        st.dataframe(pd.DataFrame(res6["structured_results"]), use_container_width=True)
        st.write("**Category Distribution:**", res6["pos_distribution"])

    # Exp 7
    with tabs[6]:
        st.subheader("Experiment 7: Chunking & Empirical Study on Training Sizes")
        study7 = analyze_training_size_and_features()
        st.write(f"**Experiment Title:** {study7['experiment_title']}")
        st.dataframe(pd.DataFrame(study7["comparison_table"]), use_container_width=True)
        st.success(f"**Key Finding:** {study7['observation']}")

    # Exp 8
    with tabs[7]:
        st.subheader("Experiment 8: Named Entity Recognition (NER)")
        ner_input = st.text_input("Fitness text with entities:", "I want a 45-min HIIT workout targeting chest and biceps using dumbbells for fat loss")
        res8 = extract_named_entities(ner_input)
        st.json(res8["slots"])
        st.write("**Entities Extracted:**", res8["entities"])

    # Exp 9
    with tabs[8]:
        st.subheader("Experiment 9: Text Similarity & Workout Recommendation")
        st.write("Compares TF-IDF Cosine Similarity, Jaccard Index, and Hybrid Ranking Multi-Factor Score.")
        sim_input = st.text_input("User preference query:", "chest and triceps hypertrophy dumbbells")
        res9 = workout_recommender.recommend_workouts(sim_input, top_k=3)
        st.dataframe(pd.DataFrame(res9["recommendations"])[["title", "category", "match_percentage", "cosine_similarity", "jaccard_similarity", "hybrid_score"]], use_container_width=True)

    # Exp 10
    with tabs[9]:
        st.subheader("Experiment 10: Word Sense Disambiguation (WSD)")
        st.write("Resolves fitness-specific polysemy ('weight', 'running', 'press', 'bar', 'rep', 'set').")
        sample_wsd_cases = [
            ("I need to lose weight with cardio", "weight"),
            ("He lifted heavy weight on the barbell", "weight"),
            ("Running five miles improves stamina", "running"),
            ("I am running a 3-day workout split", "running"),
            ("He ordered a chocolate protein bar", "bar"),
            ("Grab the pull-up bar firmly", "bar")
        ]
        selected_case = st.selectbox("Select sample ambiguous sentence:", [f"{c[0]} (Target: {c[1]})" for c in sample_wsd_cases])
        target_sent = selected_case.split(" (Target: ")[0]
        target_w = selected_case.split(" (Target: ")[1].rstrip(")")
        res10 = wsd_model.disambiguate_contextual(target_sent, target_w)
        st.success(f"**Predicted Sense:** `{res10['predicted_sense']}` (Confidence: {res10['confidence']:.2f})")
        st.write(f"**Definition:** {res10['definition']}")
        st.write(f"**Context Scores:** {res10['all_sense_scores']}")
        st.write(f"**WordNet Lesk Result:**", res10["wordnet_lesk_result"])

# ==============================================================================
# MODE 4: CATALOG & DATASET EXPLORER
# ==============================================================================
elif app_mode == "📋 Catalog & Dataset Explorer":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📋 Workout Catalog & Dataset Explorer</div>
        <div class="hero-subtitle">Browse and filter all 50 workout plans and 155 user review entries.</div>
    </div>
    """, unsafe_allow_html=True)

    cat_tab, rev_tab = st.tabs(["🏋️ Workout Plans (50 Items)", "⭐ Fitness Reviews (155 Items)"])
    with cat_tab:
        st.dataframe(df_workouts[["workout_id", "title", "category", "difficulty_level", "duration_minutes", "target_muscles", "equipment_needed", "calories_burned_est"]], use_container_width=True)
    with rev_tab:
        st.dataframe(df_reviews[["review_id", "workout_name", "sentiment", "rating", "review_text"]], use_container_width=True)

