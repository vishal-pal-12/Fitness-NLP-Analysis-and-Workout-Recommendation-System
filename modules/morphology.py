"""
Experiment 4: Morphological Analysis and Word Generation of Fitness Terms
Problem Statement: Fitness-related text contains words with different grammatical
forms, such as train, trains, training, and trained. This experiment performs
morphological analysis to identify word roots, prefixes, and suffixes and generates
related word forms from fitness vocabulary.
"""

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Common fitness affixes
PREFIXES = {
    "pre": "before / prior (e.g., pre-workout)",
    "post": "after / subsequent (e.g., post-workout, post-injury)",
    "over": "excessive / beyond normal (e.g., overtrain, overload)",
    "under": "insufficient / below normal (e.g., undertrain, recover)",
    "re": "again / restore (e.g., recover, rebuild, reset)",
    "de": "reduce / remove (e.g., dehydrate, de-load)",
    "hyper": "elevated / enlarged (e.g., hypertrophy, hypertension)",
    "un": "not / reversal (e.g., unstable, unrestricted)"
}

SUFFIXES = {
    "ing": "present participle / continuous action (e.g., training, lifting)",
    "ed": "past tense / completed action (e.g., trained, lifted)",
    "s": "third person singular or plural noun (e.g., squats, lifts)",
    "es": "third person singular or plural noun (e.g., presses, stretches)",
    "er": "agent noun / performer of action (e.g., trainer, runner, lifter)",
    "or": "agent noun / device (e.g., rotator, extensor)",
    "tion": "nominalization / state of action (e.g., conditioning, exertion)",
    "ment": "action or resulting state (e.g., development, alignment)",
    "able": "capable of being (e.g., adjustable, scalable)",
    "ive": "expressing tendency (e.g., progressive, active)",
    "ness": "quality or condition (e.g., fitness, tightness)"
}

# Core fitness root patterns and known irregular forms
FITNESS_BASE_LEMMAS = {
    "train": {"agent": "trainer", "noun": "training", "adjective": "trainable"},
    "lift": {"agent": "lifter", "noun": "lifting", "adjective": "liftable"},
    "squat": {"agent": "squatter", "noun": "squatting", "adjective": "squatting"},
    "press": {"agent": "presser", "noun": "pressing", "adjective": "pressed"},
    "stretch": {"agent": "stretcher", "noun": "stretching", "adjective": "stretchable"},
    "recover": {"agent": "recoverer", "noun": "recovery", "adjective": "recoverable"},
    "run": {"agent": "runner", "noun": "running", "adjective": "running"},
    "flex": {"agent": "flexor", "noun": "flexion", "adjective": "flexible"},
    "extend": {"agent": "extensor", "noun": "extension", "adjective": "extensible"},
    "burn": {"agent": "burner", "noun": "burning", "adjective": "burned"},
    "build": {"agent": "builder", "noun": "building", "adjective": "built"},
    "condition": {"agent": "conditioner", "noun": "conditioning", "adjective": "conditioned"},
    "hydrate": {"agent": "hydrator", "noun": "hydration", "adjective": "hydrated"},
    "load": {"agent": "loader", "noun": "loading", "adjective": "loaded"}
}

lemmatizer = WordNetLemmatizer()


def analyze_morphology(word):
    """
    Decomposes a fitness term into prefix, root/stem, suffix,
    and analyzes grammatical inflection.
    """
    word_clean = word.lower().strip()
    detected_prefix = None
    detected_prefix_desc = None
    core = word_clean

    # 1. Prefix detection
    for pfx, desc in PREFIXES.items():
        if word_clean.startswith(pfx + "-"):
            detected_prefix = pfx + "-"
            detected_prefix_desc = desc
            core = word_clean[len(pfx) + 1:]
            break
        elif word_clean.startswith(pfx) and len(word_clean) > len(pfx) + 3:
            detected_prefix = pfx
            detected_prefix_desc = desc
            core = word_clean[len(pfx):]
            break

    # 2. Suffix detection
    detected_suffix = None
    detected_suffix_desc = None
    stem_candidate = core

    for sfx in sorted(SUFFIXES.keys(), key=lambda s: len(s), reverse=True):
        if core.endswith(sfx) and len(core) > len(sfx) + 2:
            detected_suffix = sfx
            detected_suffix_desc = SUFFIXES[sfx]
            stem_candidate = core[:-len(sfx)]
            break

    # 3. Lemmatization & root identification
    verb_root = lemmatizer.lemmatize(core, pos="v")
    noun_root = lemmatizer.lemmatize(core, pos="n")
    
    if verb_root in FITNESS_BASE_LEMMAS:
        final_root = verb_root
    elif noun_root in FITNESS_BASE_LEMMAS:
        final_root = noun_root
    elif verb_root != core:
        final_root = verb_root
    elif noun_root != core:
        final_root = noun_root
    else:
        final_root = stem_candidate if detected_suffix else core

    # 4. Generate related morphological variants
    generated = generate_word_forms(final_root)

    return {
        "original_word": word,
        "root": final_root,
        "prefix": detected_prefix,
        "prefix_description": detected_prefix_desc,
        "suffix": detected_suffix,
        "suffix_description": detected_suffix_desc,
        "base_part_of_speech": "Verb" if verb_root in FITNESS_BASE_LEMMAS else "Noun/General",
        "generated_forms": generated
    }


def generate_word_forms(root):
    """
    Generates grammatical and derivational forms from a fitness root word:
    - 3rd person singular present
    - Present participle / Gerund (-ing)
    - Past tense / Participle (-ed)
    - Agent noun (-er / -or)
    - Prefixed derivatives (over-, under-, re-, pre-)
    """
    root = root.lower().strip()
    
    # Check if known custom dictionary exists
    custom_info = FITNESS_BASE_LEMMAS.get(root, {})
    agent = custom_info.get("agent")
    action_noun = custom_info.get("noun")
    adjective = custom_info.get("adjective")

    # Regular inflection rules
    # 1. 3rd person singular present (-s / -es)
    if root.endswith(("s", "sh", "ch", "x", "z")):
        third_sg = root + "es"
    elif root.endswith("y") and len(root) > 1 and root[-2] not in "aeiou":
        third_sg = root[:-1] + "ies"
    else:
        third_sg = root + "s"

    # 2. Present participle (-ing)
    if root == "run":
        pres_part = "running"
    elif root == "squat":
        pres_part = "squatting"
    elif root.endswith("e") and not root.endswith("ee"):
        pres_part = root[:-1] + "ing"
    elif len(root) >= 3 and root[-1] not in "aeiouwxy" and root[-2] in "aeiou" and root[-3] not in "aeiou":
        pres_part = root + root[-1] + "ing"
    else:
        pres_part = root + "ing"

    # 3. Past tense (-ed)
    if root == "run":
        past = "ran"
    elif root == "build":
        past = "built"
    elif root == "squat":
        past = "squatted"
    elif root.endswith("e"):
        past = root + "d"
    elif len(root) >= 3 and root[-1] not in "aeiouwxy" and root[-2] in "aeiou" and root[-3] not in "aeiou":
        past = root + root[-1] + "ed"
    else:
        past = root + "ed"

    # 4. Agent noun (-er / -or)
    if not agent:
        if root.endswith("e"):
            agent = root + "r"
        elif len(root) >= 3 and root[-1] not in "aeiouwxy" and root[-2] in "aeiou" and root[-3] not in "aeiou":
            agent = root + root[-1] + "er"
        else:
            agent = root + "er"

    # 5. Prefixed derivatives
    derivatives = [
        f"over{root}",
        f"under{root}",
        f"re{root}",
        f"pre-{root}",
        f"post-{root}"
    ]

    forms = {
        "root": root,
        "present_3rd_person": third_sg,
        "present_participle_gerund": pres_part,
        "past_tense": past,
        "agent_noun": agent,
        "nominalization": action_noun or pres_part,
        "adjective_form": adjective or past,
        "prefixed_derivatives": derivatives
    }
    return forms


def expand_query_morphologically(tokens):
    """
    Expands query tokens using morphological analysis so that search
    matches all grammatical inflections (e.g. 'squat' expands to 'squats', 'squatting').
    """
    expanded = set(tokens)
    explanations = []

    for tok in tokens:
        analysis = analyze_morphology(tok)
        root = analysis["root"]
        expanded.add(root)
        forms = analysis["generated_forms"]
        for key in ["present_3rd_person", "present_participle_gerund", "past_tense", "agent_noun"]:
            form_val = forms.get(key)
            if form_val:
                expanded.add(form_val)
        explanations.append({
            "token": tok,
            "root": root,
            "forms_added": [forms["present_3rd_person"], forms["present_participle_gerund"], forms["past_tense"]]
        })

    return {
        "original_tokens": tokens,
        "expanded_tokens": sorted(list(expanded)),
        "expansion_details": explanations
    }

