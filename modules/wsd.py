"""
Experiment 10: Word Sense Disambiguation (WSD) for Fitness Text
Problem Statement: Some words in fitness-related queries can have different meanings
depending on their context. For example, "weight" may refer to body weight or weights
used in strength training, while "running" may refer to a specific exercise or a
general activity. This experiment applies Word Sense Disambiguation techniques to
interpret ambiguous words based on their surrounding context and improve the
understanding of fitness queries.
"""

import re
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Detailed fitness domain sense definitions & context profiles
AMBIGUOUS_FITNESS_WORDS = {
    "weight": {
        "BODY_WEIGHT": {
            "definition": "Body mass / anatomical weight (e.g., body fat, weighing scale, weight loss, obesity)",
            "context_keywords": ["loss", "lose", "losing", "fat", "scale", "diet", "calorie", "obese", "pounds", "kilos", "reduction", "drop", "bodyweight", "gain"],
            "example": "I need a cardio plan to lose weight and burn calories."
        },
        "GYM_LOAD": {
            "definition": "External resistance / barbell plates / dumbbells used for resistance training",
            "context_keywords": ["lift", "lifting", "heavy", "bar", "plates", "dumbbells", "bench", "press", "squat", "rack", "reps", "kg", "lbs", "overload", "increase"],
            "example": "I lift heavy weight on the barbell bench press."
        }
    },
    "running": {
        "CARDIO_EXERCISE": {
            "definition": "Cardiovascular running / jogging locomotion exercise",
            "context_keywords": ["miles", "km", "jogging", "treadmill", "sprint", "shoes", "pace", "track", "heart", "stamina", "endurance", "aerobic", "legs", "5k", "marathon"],
            "example": "Running five miles on the treadmill builds great stamina."
        },
        "EXECUTION_GENERAL": {
            "definition": "Managing, operating, or executing an activity or program",
            "context_keywords": ["split", "program", "routine", "schedule", "timer", "clock", "out of", "breath", "low", "smoothly", "system", "app"],
            "example": "I am running a 4-day workout split this month."
        }
    },
    "press": {
        "EXERCISE_MOVEMENT": {
            "definition": "Weight training pushing movement (e.g., bench press, overhead press)",
            "context_keywords": ["bench", "chest", "overhead", "shoulder", "dumbbells", "barbell", "incline", "military", "push", "sets", "reps", "lockout"],
            "example": "Barbell overhead press strengthens the anterior deltoids."
        },
        "MEDIA_OR_ACTION": {
            "definition": "Journalistic media, news release, or physical button action",
            "context_keywords": ["news", "media", "release", "journal", "button", "screen", "conference", "article", "publish"],
            "example": "The sports company published a press release regarding the new gym shoes."
        }
    },
    "bar": {
        "GYM_EQUIPMENT": {
            "definition": "Barbell or pull-up apparatus used for gym training",
            "context_keywords": ["barbell", "pullup", "pull-up", "olympic", "rack", "grip", "hang", "collar", "chalk", "plates", "knurling"],
            "example": "Grip the pull-up bar firmly and pull your chin over the bar."
        },
        "NUTRITION_FOOD": {
            "definition": "Nutritional dietary supplement bar (e.g. protein bar, energy bar)",
            "context_keywords": ["protein", "snack", "energy", "eat", "flavor", "chocolate", "calories", "grams", "chewy", "meal", "diet"],
            "example": "I ate a 20g whey protein bar right after my workout."
        },
        "SOCIAL_VENUE": {
            "definition": "Drinking establishment or tavern",
            "context_keywords": ["drinks", "beer", "cocktail", "alcohol", "bartender", "nightclub", "pub", "friends", "party"],
            "example": "We met friends at the juice bar after work."
        }
    },
    "rep": {
        "EXERCISE_REPETITION": {
            "definition": "Single completion of an exercise movement (repetition)",
            "context_keywords": ["repetition", "sets", "squat", "pushup", "failure", "count", "cadence", "tempo", "form", "10", "12", "maximum"],
            "example": "Perform 3 sets of 10 reps of Romanian deadlifts."
        },
        "REPRESENTATIVE": {
            "definition": "Spokesperson or sales / customer agent",
            "context_keywords": ["sales", "customer", "agent", "support", "company", "call", "contact", "spokesperson", "membership"],
            "example": "I talked to the gym sales rep about canceling my annual membership."
        }
    },
    "set": {
        "EXERCISE_SET": {
            "definition": "A group of consecutive repetitions in a workout",
            "context_keywords": ["reps", "rest", "working", "warmup", "superset", "dropset", "failure", "interval", "exhaustion", "3", "4"],
            "example": "Complete four sets of eight reps with ninety seconds rest."
        },
        "COLLECTION_GENERAL": {
            "definition": "A collection, group of items, or setting an adjustment",
            "context_keywords": ["rules", "instructions", "timer", "clock", "target", "goal", "adjust", "collection", "equipment"],
            "example": "She established a new set of training guidelines."
        }
    }
}


class FitnessWSDModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def disambiguate_with_lesk(self, sentence, target_word):
        """Standard NLTK Lesk algorithm using WordNet synsets."""
        tokens = nltk.word_tokenize(sentence.lower())
        synset = lesk(tokens, target_word)
        if synset:
            return {
                "synset_name": synset.name(),
                "definition": synset.definition(),
                "examples": synset.examples()
            }
        return {
            "synset_name": "None",
            "definition": "No synset found via Lesk",
            "examples": []
        }

    def disambiguate_contextual(self, sentence, target_word=None):
        """
        Disambiguates target ambiguous word using context similarity against
        fitness domain senses. If target_word is not specified, auto-detects
        ambiguous terms present in the sentence.
        """
        sentence_clean = sentence.lower()
        
        # If target word not supplied, detect first matching ambiguous term
        if not target_word:
            for term in AMBIGUOUS_FITNESS_WORDS.keys():
                if re.search(rf"\b{term}\b", sentence_clean):
                    target_word = term
                    break

        if not target_word or target_word not in AMBIGUOUS_FITNESS_WORDS:
            return {
                "sentence": sentence,
                "target_word": target_word or "None",
                "detected": False,
                "error": f"Target word '{target_word}' is not in the ambiguous fitness vocabulary."
            }

        senses_dict = AMBIGUOUS_FITNESS_WORDS[target_word]
        words_in_sentence = set(re.findall(r"\b\w+\b", sentence_clean))

        sense_scores = {}
        matched_cues = {}

        for sense_name, sense_data in senses_dict.items():
            keywords = sense_data["context_keywords"]
            overlap = words_in_sentence.intersection(set(keywords))
            score = len(overlap) * 2.0

            # Text similarity between sentence and sense profile
            profile_text = f"{sense_data['definition']} {' '.join(keywords)} {sense_data['example']}"
            try:
                vec = TfidfVectorizer().fit_transform([sentence_clean, profile_text])
                cos_sim = float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
            except Exception:
                cos_sim = 0.0

            final_score = score + (cos_sim * 3.0)
            sense_scores[sense_name] = final_score
            matched_cues[sense_name] = list(overlap)

        # Determine winner
        best_sense = max(sense_scores.items(), key=lambda x: x[1])[0]
        total_score = sum(sense_scores.values()) or 1.0
        confidence = min(0.99, max(0.55, sense_scores[best_sense] / total_score))

        # Also get Lesk result for comparison
        lesk_res = self.disambiguate_with_lesk(sentence, target_word)

        return {
            "sentence": sentence,
            "target_word": target_word,
            "detected": True,
            "predicted_sense": best_sense,
            "definition": senses_dict[best_sense]["definition"],
            "confidence": round(float(confidence), 3),
            "matched_context_keywords": matched_cues.get(best_sense, []),
            "all_sense_scores": {k: round(v, 3) for k, v in sense_scores.items()},
            "wordnet_lesk_result": lesk_res
        }


wsd_model = FitnessWSDModel()

