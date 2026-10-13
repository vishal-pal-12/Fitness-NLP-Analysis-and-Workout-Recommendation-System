"""
Experiment 2: Tokenization, Filtration, and Script Validation of Fitness Text
Experiment 3: Stop Word Removal, Stemming, and Lemmatization of Fitness Text

Problem Statements:
- Exp 2: Fitness-related text contains words, punctuation, numbers, and special
  characters. Performs tokenization, filtration, and script validation to clean
  fitness text and prepare valid tokens.
- Exp 3: Fitness descriptions contain common words and different forms of the same
  word (exercise, exercises, exercising). Removes stop words and applies stemming
  and lemmatization to normalize fitness text.
"""

import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer

# Ensure required NLTK resources
for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res, quiet=True)

# Common contraction mapping
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'t": " not", "'ve": " have",
    "'m": " am", "let's": "let us", "gains'll": "gains will"
}

# Domain-specific stop words adjustment: keep directional and goal terms
FITNESS_EXEMPT_STOPWORDS = {
    "up", "down", "out", "in", "off", "on", "no", "not", "more", "most", "under", "over"
}

def expand_contractions(text):
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = re.sub(re.escape(contraction), expansion, text_lower)
    return text_lower

def validate_script(text):
    """
    Validates script of fitness text.
    Checks whether the input primarily consists of English/Latin script
    and identifies foreign scripts (Devanagari, Cyrillic, Arabic, CJK) or emojis.
    """
    scripts_detected = {}
    total_chars = 0
    latin_chars = 0
    non_latin_chars = 0
    special_or_emojis = 0

    for ch in text:
        if ch.isspace():
            continue
        total_chars += 1
        name = unicodedata.name(ch, "UNKNOWN")
        cat = unicodedata.category(ch)

        if "LATIN" in name:
            latin_chars += 1
            scripts_detected["Latin (English)"] = scripts_detected.get("Latin (English)", 0) + 1
        elif "DEVANAGARI" in name:
            non_latin_chars += 1
            scripts_detected["Devanagari (Hindi)"] = scripts_detected.get("Devanagari (Hindi)", 0) + 1
        elif "ARABIC" in name:
            non_latin_chars += 1
            scripts_detected["Arabic"] = scripts_detected.get("Arabic", 0) + 1
        elif "CYRILLIC" in name:
            non_latin_chars += 1
            scripts_detected["Cyrillic"] = scripts_detected.get("Cyrillic", 0) + 1
        elif "CJK" in name:
            non_latin_chars += 1
            scripts_detected["CJK (Chinese/Japanese)"] = scripts_detected.get("CJK", 0) + 1
        elif cat.startswith("S") or cat.startswith("P"):
            special_or_emojis += 1
            scripts_detected["Symbols/Punctuation"] = scripts_detected.get("Symbols/Punctuation", 0) + 1
        else:
            scripts_detected["Other"] = scripts_detected.get("Other", 0) + 1

    is_valid_latin = total_chars > 0 and (latin_chars / max(1, total_chars - special_or_emojis)) >= 0.8
    primary_script = max(scripts_detected.items(), key=lambda x: x[1])[0] if scripts_detected else "Empty"

    return {
        "is_valid": is_valid_latin,
        "primary_script": primary_script,
        "total_characters": total_chars,
        "latin_count": latin_chars,
        "non_latin_count": non_latin_chars,
        "symbol_or_emoji_count": special_or_emojis,
        "script_distribution": scripts_detected
    }

def tokenize_and_filter(text):
    """
    Experiment 2:
    Performs sentence tokenization, word tokenization, filtration,
    preserves fitness patterns like '3x10', '50kg', '30min', and validates script.
    """
    script_info = validate_script(text)
    expanded = expand_contractions(text)
    sentences = sent_tokenize(text)

    # Word tokenizers
    treebank_tokens = word_tokenize(expanded)
    punct_tokens = wordpunct_tokenize(expanded)
    regex_tokens = re.findall(r"\b\w+(?:[-']\w+)*\b|\d+x\d+|\d+kg|\d+lbs|\d+min", expanded)

    # Filtration: remove pure punctuation, symbols, non-informative tokens
    filtered_tokens = []
    for tok in regex_tokens:
        tok_clean = tok.strip()
        if len(tok_clean) > 0 and not re.match(r"^[\W_]+$", tok_clean):
            filtered_tokens.append(tok_clean.lower())

    return {
        "original_text": text,
        "sentence_tokens": sentences,
        "sentence_count": len(sentences),
        "word_tokens": treebank_tokens,
        "wordpunct_tokens": punct_tokens,
        "regex_tokens": regex_tokens,
        "filtered_tokens": filtered_tokens,
        "total_raw_tokens": len(treebank_tokens),
        "total_filtered_tokens": len(filtered_tokens),
        "script_validation": script_info
    }

def _get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text, use_domain_stopwords=True):
    """
    Experiment 3:
    Removes stop words, applies Porter, Lancaster, and Snowball stemmers,
    and applies WordNet POS-aware Lemmatizer.
    """
    tok_data = tokenize_and_filter(text)
    tokens = tok_data["filtered_tokens"]

    # Stop words
    stop_words_set = set(stopwords.words("english"))
    if use_domain_stopwords:
        stop_words_set = stop_words_set - FITNESS_EXEMPT_STOPWORDS

    non_stopword_tokens = [t for t in tokens if t not in stop_words_set]

    # Stemming algorithms
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")

    porter_stemmed = [porter.stem(t) for t in non_stopword_tokens]
    lancaster_stemmed = [lancaster.stem(t) for t in non_stopword_tokens]
    snowball_stemmed = [snowball.stem(t) for t in non_stopword_tokens]

    # Lemmatization with POS tagging
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(non_stopword_tokens) if non_stopword_tokens else []
    pos_lemmatized = [
        lemmatizer.lemmatize(word, _get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]

    # Detailed token mapping comparison
    comparison_table = []
    for i, tok in enumerate(non_stopword_tokens):
        pos_tag = pos_tags[i][1] if i < len(pos_tags) else "NN"
        comparison_table.append({
            "Original": tok,
            "Porter": porter_stemmed[i],
            "Lancaster": lancaster_stemmed[i],
            "Snowball": snowball_stemmed[i],
            "POS": pos_tag,
            "Lemmatized": pos_lemmatized[i]
        })

    reduction_rate = 0.0
    if len(tokens) > 0:
        reduction_rate = round((1 - len(non_stopword_tokens) / len(tokens)) * 100, 2)

    return {
        "original_tokens": tokens,
        "stopword_removed_tokens": non_stopword_tokens,
        "stemmed_tokens": porter_stemmed,
        "lancaster_tokens": lancaster_stemmed,
        "snowball_tokens": snowball_stemmed,
        "lemmatized_tokens": pos_lemmatized,
        "comparison_table": comparison_table,
        "script_validation": tok_data["script_validation"],
        "reduction_rate_pct": reduction_rate
    }

