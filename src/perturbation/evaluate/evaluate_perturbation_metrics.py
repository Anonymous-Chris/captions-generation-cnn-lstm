import os

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer

# Files
input_file = "results/perturbation/" "perturbed_captions_with_references.csv"
output_file = "results/perturbation/" "perturbation_metrics.csv"

# Load data
df = pd.read_csv(input_file)
print("Total rows:", len(df))

# Metric setup
smoothing = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# Bleu
def calculate_bleu(candidate, references):
    candidate_tokens = candidate.split()
    reference_tokens = [ref.split() for ref in references]
    bleu1 = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing,
    )

    bleu2 = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing,
    )

    bleu3 = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smoothing,
    )

    bleu4 = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    return bleu1, bleu2, bleu3, bleu4


# Meteor
def calculate_meteor(candidate, references):
    reference_tokens = [ref.split() for ref in references]
    candidate_tokens = candidate.split()
    return meteor_score(reference_tokens, candidate_tokens)


# ROUGE-L
def calculate_rouge(candidate, references):
    scores = []
    for reference in references:
        score = rouge.score(reference, candidate)["rougeL"].fmeasure
        scores.append(score)

    # Use best matching reference
    return max(scores)


# Calculate BLEU, METEOR, ROUGE
bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []
meteor_scores = []
rouge_scores = []

for i, row in df.iterrows():
    candidate = str(row["perturbed_caption"]).strip()

    references = [
        ref.strip() for ref in str(row["references"]).split(" ||| ") if ref.strip()
    ]

    b1, b2, b3, b4 = calculate_bleu(candidate, references)
    meteor = calculate_meteor(candidate, references)
    rouge_l = calculate_rouge(candidate, references)

    bleu1_scores.append(b1)
    bleu2_scores.append(b2)
    bleu3_scores.append(b3)
    bleu4_scores.append(b4)
    meteor_scores.append(meteor)
    rouge_scores.append(rouge_l)

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(df)}")

df["BLEU-1"] = bleu1_scores
df["BLEU-2"] = bleu2_scores
df["BLEU-3"] = bleu3_scores
df["BLEU-4"] = bleu4_scores
df["METEOR"] = meteor_scores
df["ROUGE-L"] = rouge_scores

# CIDEr
print("\nCalculating CIDEr...")

gts = {}
res = {}

for i, row in df.iterrows():
    references = [
        ref.strip() for ref in str(row["references"]).split(" ||| ") if ref.strip()
    ]

    candidate = str(row["perturbed_caption"]).strip()
    gts[i] = references
    res[i] = [candidate]

cider = Cider()
cider_score, cider_scores = cider.compute_score(gts, res)
df["CIDEr"] = cider_scores

# Save
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)

# Summary
print("\nEvaluation complete")
print("-" * 60)
print("Rows:", len(df))
print("\nAverage scores across all rows:")
metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
print(df[metrics].mean().round(4))
print("\nSaved to:")
print(output_file)
