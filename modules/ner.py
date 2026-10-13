"""
Experiment 8: Named Entity Recognition (NER) for Fitness Information Extraction
Problem Statement: Fitness-related text may contain important entities such as
exercise names, body parts, workout types, durations, and fitness-related organizations.
This experiment applies Named Entity Recognition and related extraction techniques to
identify relevant information from fitness descriptions and user queries for use in
workout recommendation.
"""

import os
import json
import re
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk


class FitnessNERExtractor:
    def __init__(self, entities_file=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.entities_file = entities_file or os.path.join(base_dir, "data", "fitness_entities.json")
        self.domain_entities = self._load_entities()

    def _load_entities(self):
        if os.path.exists(self.entities_file):
            try:
                with open(self.entities_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def extract_named_entities(self, text):
        """
        Extracts both fitness domain entities (EXERCISE, MUSCLE_GROUP, WORKOUT_TYPE,
        EQUIPMENT, DIFFICULTY, DURATION, GOAL) and standard NLTK entities (PERSON, GPE, ORG).
        Also performs structured slot-filling for intelligent recommendations.
        """
        text_lower = text.lower()
        extracted = []
        slots = {
            "goals": [],
            "workout_types": [],
            "muscle_groups": [],
            "exercises": [],
            "equipment": [],
            "difficulty": None,
            "duration_minutes": None
        }

        # 1. Regex duration extraction (e.g. '30 minutes', '45 min', '45-min', '1 hour', '1-hr')
        dur_match = re.search(r"\b(\d+)\s*[-]?\s*(?:min|minute|minutes|mins)\b", text_lower)
        if dur_match:
            mins = int(dur_match.group(1))
            extracted.append({
                "entity": dur_match.group(0),
                "label": "DURATION",
                "source": "Regex Pattern",
                "value": mins
            })
            slots["duration_minutes"] = mins
        else:
            hr_match = re.search(r"\b(\d+)\s*[-]?\s*(?:hour|hours|hr|hrs)\b", text_lower)
            if hr_match:
                mins = int(hr_match.group(1)) * 60
                extracted.append({
                    "entity": hr_match.group(0),
                    "label": "DURATION",
                    "source": "Regex Pattern",
                    "value": mins
                })
                slots["duration_minutes"] = mins

        # 2. Domain gazetteer entity matching
        for label, term_list in self.domain_entities.items():
            # Sort by length descending to match multi-word phrases first
            sorted_terms = sorted(term_list, key=lambda t: len(t), reverse=True)
            for term in sorted_terms:
                pattern = rf"\b{re.escape(term)}\b"
                for match in re.finditer(pattern, text_lower):
                    # Check if already covered
                    if not any(e["entity"].lower() == term for e in extracted):
                        extracted.append({
                            "entity": text[match.start():match.end()],
                            "label": label,
                            "source": "Domain Gazetteer",
                            "start": match.start(),
                            "end": match.end()
                        })

                        # Slot filling
                        if label == "GOAL":
                            slots["goals"].append(term)
                        elif label == "WORKOUT_TYPE":
                            slots["workout_types"].append(term)
                        elif label == "MUSCLE_GROUP":
                            slots["muscle_groups"].append(term)
                        elif label == "EXERCISE":
                            slots["exercises"].append(term)
                        elif label == "EQUIPMENT":
                            slots["equipment"].append(term)
                        elif label == "DIFFICULTY_LEVEL" and not slots["difficulty"]:
                            slots["difficulty"] = term.capitalize()

        # 3. Standard NLTK Named Entity Chunker (Persons, Brands/Orgs, Locations)
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        try:
            tree = ne_chunk(tagged)
            for subtree in tree:
                if hasattr(subtree, 'label'):
                    entity_text = " ".join(c[0] for c in subtree)
                    label = subtree.label()
                    # Filter out overlaps
                    if not any(e["entity"].lower() == entity_text.lower() for e in extracted):
                        extracted.append({
                            "entity": entity_text,
                            "label": label,
                            "source": "NLTK MaxEnt Chunker"
                        })
        except Exception:
            pass

        return {
            "text": text,
            "entities": extracted,
            "entity_count": len(extracted),
            "slots": slots
        }


ner_extractor = FitnessNERExtractor()
extract_named_entities = ner_extractor.extract_named_entities
