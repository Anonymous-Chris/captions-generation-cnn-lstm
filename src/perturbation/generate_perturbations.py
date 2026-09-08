import random
import re

import pandas as pd

# Settings
input_file = "results/perturbation/perturbation_sample.csv"
output_file = "results/perturbation/perturbed_captions.csv"
random.seed(42)

# Simple replacement dictionaries
object_replacements = {
    "dog": "cat",
    "cat": "dog",
    "bicycle": "motorcycle",
    "bike": "motorcycle",
    "car": "truck",
    "truck": "car",
    "horse": "cow",
    "cow": "horse",
    "ball": "book",
    "boat": "car",
}

person_replacements = {"man": "woman", "woman": "man", "boy": "girl", "girl": "boy"}

color_replacements = {
    "red": "blue",
    "blue": "red",
    "green": "yellow",
    "yellow": "green",
    "pink": "purple",
    "purple": "pink",
}

action_phrase_replacements = {
    # Running
    "is running on": "is standing on",
    "are running on": "are standing on",
    "is running beside": "is standing beside",
    "are running beside": "are standing beside",
    "is running through": "is standing in",
    "are running through": "are standing in",
    # Sitting
    "is sitting beside": "is standing beside",
    "are sitting beside": "are standing beside",
    # Riding
    "is riding a scooter": "is carrying a scooter",
    "are riding a scooter": "are carrying a scooter",
    "is riding a bicycle": "is carrying a bicycle",
    "are riding a bicycle": "are carrying a bicycle",
    "is riding a bike": "is carrying a bike",
    "are riding a bike": "are carrying a bike",
    # Walking
    "is walking in front of": "is standing in front of",
    "are walking in front of": "are standing in front of",
    "is walking on": "is standing on",
    "are walking on": "are standing on",
    # Jumping
    "is jumping in": "is standing in",
    "are jumping in": "are standing in",
    "is jumping on": "is standing on",
    "are jumping on": "are standing on",
    "is jumping over": "is standing beside",
    "are jumping over": "are standing beside",
    # Playing
    "is playing a guitar": "is holding a guitar",
    "are playing a guitar": "are holding a guitar",
    "is playing volleyball": "is watching volleyball",
    "are playing volleyball": "are watching volleyball",
    # Looking
    "is looking at": "is pointing at",
    "are looking at": "are pointing at",
    # Climbing
    "is climbing up": "is standing beside",
    "are climbing up": "are standing beside",
    # Catching
    "is catching a frisbee": "is holding a frisbee",
    "are catching a frisbee": "are holding a frisbee",
    # Holding
    "is holding onto": "is standing beside",
    "are holding onto": "are standing beside",
    # Rowing
    "is rowing in": "is floating in",
    "are rowing in": "are floating in",
    # Surfing
    "is surfing in": "is swimming in",
    "are surfing in": "are swimming in",
    # Chasing
    "is chasing": "is watching",
    "are chasing": "are watching",
    # Sniffing
    "is sniffing": "is walking on",
    "are sniffing": "are walking on",
    # Recording
    "is recording": "is watching",
    "are recording": "are watching",
}

hallucinated_objects = [
    "a helicopter",
    "an umbrella",
    "a suitcase",
    "a traffic cone",
    "a television",
]


# Helper function
def replace_word(caption, replacements):
    words = caption.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r"[^a-zA-Z]", "", word.lower())
        if clean_word in replacements:
            replacement = replacements[clean_word]
            punctuation = word[len(clean_word) :] if len(word) > len(clean_word) else ""
            words[i] = replacement + punctuation
            return " ".join(words)
    return None


# Skips blue and white format
def replace_color_safely(caption):
    words = caption.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r"[^a-zA-Z]", "", word.lower())
        if clean_word not in color_replacements:
            continue

        # Skip constructions such as:
        # "black and white"
        # "red and green"
        previous_word = words[i - 1].lower() if i > 0 else ""
        next_word = words[i + 1].lower() if i + 1 < len(words) else ""
        if previous_word == "and" or next_word == "and":
            continue

        replacement = color_replacements[clean_word]
        return " ".join(words[:i] + [replacement] + words[i + 1 :])
    return None


def replace_action_phrase(caption):
    lower_caption = caption.lower()
    for phrase, replacement in action_phrase_replacements.items():
        if phrase in lower_caption:
            return re.sub(
                re.escape(phrase), replacement, caption, count=1, flags=re.IGNORECASE
            )
    return None


def negate_action(caption):
    """
    Convert:
        is running -> is not running
        are playing -> are not playing
        is sitting -> is not sitting
    """
    patterns = [
        r"\bis ([a-z]+ing)\b",
        r"\bare ([a-z]+ing)\b",
        r"\bwas ([a-z]+ing)\b",
        r"\bwere ([a-z]+ing)\b",
    ]

    replacements = [r"is not \1", r"are not \1", r"was not \1", r"were not \1"]
    for pattern, replacement in zip(patterns, replacements):
        new_caption, count = re.subn(
            pattern, replacement, caption, count=1, flags=re.IGNORECASE
        )

        if count > 0:
            return new_caption
    return None


def create_hallucination(caption):
    lower_caption = caption.lower()
    available = []
    for obj in hallucinated_objects:
        object_word = obj.split()[-1]
        if object_word not in lower_caption:
            available.append(obj)

    if not available:
        return None

    hallucination = random.choice(available)
    return f"{caption} with {hallucination}"


# Load sample
df = pd.read_csv(input_file)
rows = []

# Generate perturbations
for _, row in df.iterrows():
    image = row["image"]
    original = str(row["original_caption"]).strip()

    # Original
    rows.append(
        {
            "image": image,
            "perturbation_type": "original",
            "original_caption": original,
            "perturbed_caption": original,
        }
    )

    # Object replacement
    object_caption = replace_word(original, object_replacements)
    if object_caption:
        rows.append(
            {
                "image": image,
                "perturbation_type": "object_replacement",
                "original_caption": original,
                "perturbed_caption": object_caption,
            }
        )

    # Person replacement
    person_caption = replace_word(original, person_replacements)
    if person_caption:
        rows.append(
            {
                "image": image,
                "perturbation_type": "person_replacement",
                "original_caption": original,
                "perturbed_caption": person_caption,
            }
        )

    # Color replacement
    color_caption = replace_color_safely(original)
    if color_caption:
        rows.append(
            {
                "image": image,
                "perturbation_type": "color_replacement",
                "original_caption": original,
                "perturbed_caption": color_caption,
            }
        )

    # Action replacement
    action_caption = replace_action_phrase(original)
    if action_caption:
        rows.append(
            {
                "image": image,
                "perturbation_type": "action_replacement",
                "original_caption": original,
                "perturbed_caption": action_caption,
            }
        )

    # Negation
    negated_caption = negate_action(original)
    if negated_caption:
        rows.append(
            {
                "image": image,
                "perturbation_type": "negation",
                "original_caption": original,
                "perturbed_caption": negated_caption,
            }
        )

    # Hallucination
    hallucinated_caption = create_hallucination(original)
    if hallucinated_caption:
        rows.append(
            {
                "image": image,
                "perturbation_type": "hallucination",
                "original_caption": original,
                "perturbed_caption": hallucinated_caption,
            }
        )

# Save
result_df = pd.DataFrame(rows)
result_df.to_csv(output_file, index=False)

# Summary
print("\nPerturbation dataset created")
print("-" * 60)
print(f"Images: {df['image'].nunique()}")
print(f"Total captions: {len(result_df)}")
print("\nPerturbation counts")
print("-" * 60)
print(result_df["perturbation_type"].value_counts())
print("\nSaved to:")
print(output_file)
