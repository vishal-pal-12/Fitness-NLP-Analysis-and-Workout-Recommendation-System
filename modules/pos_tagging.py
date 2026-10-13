"""
Experiment 6: Part-of-Speech (POS) Tagging of Fitness Text
Problem Statement: Fitness-related descriptions and user queries contain nouns,
verbs, adjectives, and other grammatical categories. This experiment performs
Part-of-Speech tagging to identify the grammatical role of each word in fitness
text and improve the understanding of workout requests and exercise descriptions.
"""

from collections import Counter
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

PENN_TAG_DESCRIPTIONS = {
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition / subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "MD": "Modal auxiliary",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund / present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb"
}

def map_to_fitness_category(tag):
    if tag.startswith("NN"):
        return "Noun (Exercise / Muscle / Equipment)"
    elif tag.startswith("VB"):
        return "Verb (Workout Action / Movement)"
    elif tag.startswith("JJ"):
        return "Adjective (Intensity / Difficulty / Quality)"
    elif tag.startswith("RB"):
        return "Adverb (Frequency / Velocity / Manner)"
    elif tag == "CD":
        return "Number (Reps / Sets / Duration / Weight)"
    elif tag in ["IN", "TO"]:
        return "Preposition / Direction"
    elif tag in ["DT", "PDT"]:
        return "Determiner"
    else:
        return "Other Function Word"

FITNESS_ACTION_VERBS = {
    "lift", "squat", "press", "run", "stretch", "burn", "jump", "train",
    "curl", "row", "pull", "push", "build", "breathe", "hold", "target", "drop"
}

def tag_pos(text):
    """
    Tags fitness text with Penn Treebank POS tags, maps to high-level
    fitness semantic categories, and calculates POS distribution.
    Handles sentence-initial capitalization and imperative fitness verbs.
    """
    tokens = word_tokenize(text)
    raw_tags = pos_tag(tokens)

    # Check for sentence-initial words or fitness action verbs
    adjusted_tags = []
    for idx, (word, tag) in enumerate(raw_tags):
        word_l = word.lower()
        if word_l in FITNESS_ACTION_VERBS and tag in ["NNP", "NN"]:
            adjusted_tags.append((word, "VB"))
        else:
            adjusted_tags.append((word, tag))

    structured_results = []
    category_counts = Counter()
    tag_counts = Counter()

    for word, tag in adjusted_tags:
        category = map_to_fitness_category(tag)
        desc = PENN_TAG_DESCRIPTIONS.get(tag, "Punctuation / Symbol")
        category_counts[category] += 1
        tag_counts[tag] += 1

        structured_results.append({
            "word": word,
            "tag": tag,
            "tag_description": desc,
            "category": category
        })

    total_words = len(tokens)
    distribution = [
        {
            "Category": cat,
            "Count": count,
            "Percentage": round((count / max(1, total_words)) * 100, 1)
        }
        for cat, count in category_counts.most_common()
    ]

    # Extract key fitness terms based on POS
    nouns = [item["word"] for item in structured_results if item["tag"].startswith("NN")]
    verbs = [item["word"] for item in structured_results if item["tag"].startswith("VB")]
    adjectives = [item["word"] for item in structured_results if item["tag"].startswith("JJ")]

    return {
        "text": text,
        "total_tokens": total_words,
        "structured_results": structured_results,
        "pos_distribution": distribution,
        "tag_counts": dict(tag_counts),
        "extracted_nouns": nouns,
        "extracted_verbs": verbs,
        "extracted_adjectives": adjectives
    }
