import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Files
# Per image clip score
# Per image cider score
predictions_file = "results/test_predictions.csv"
cider_file = "results/test_predictions_with_cider.csv"
clip_file = "results/test_predictions_with_clipscore.csv"
output_file = "results/per_image_metrics.csv"

# Load predictions + CLIPScore
df = pd.read_csv(predictions_file)
cider_df = pd.read_csv(cider_file)[["image", "CIDEr"]]
clip_df = pd.read_csv(clip_file)[["image", "CLIPScore"]]

# Merge CIDEr and CLIPScore
df = df.merge(cider_df, on="image", how="left")
df = df.merge(clip_df, on="image", how="left")

print("Images:", len(df))

# Metric setup
smoothing = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []
meteor_scores = []
rouge_scores = []

# Calculate per-image metrics
for i, row in df.iterrows():
    references_text = row["references"].split(" ||| ")
    generated_text = str(row["generated_caption"])

    # Tokenize
    references_tokens = [reference.split() for reference in references_text]
    generated_tokens = generated_text.split()

    # BLEU-1
    bleu1 = sentence_bleu(
        references_tokens,
        generated_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing,
    )

    # BLEU-2
    bleu2 = sentence_bleu(
        references_tokens,
        generated_tokens,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing,
    )

    # BLEU-3
    bleu3 = sentence_bleu(
        references_tokens,
        generated_tokens,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smoothing,
    )

    # BLEU-4
    bleu4 = sentence_bleu(
        references_tokens,
        generated_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    # METEOR
    meteor = meteor_score(references_tokens, generated_tokens)

    # ROUGE-L
    rouge_reference_scores = [
        rouge.score(reference, generated_text)["rougeL"].fmeasure
        for reference in references_text
    ]

    rouge_l = max(rouge_reference_scores)

    # Store scores
    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu3_scores.append(bleu3)
    bleu4_scores.append(bleu4)

    meteor_scores.append(meteor)
    rouge_scores.append(rouge_l)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {len(df)}")

# Add metrics to DataFrame

df["BLEU-1"] = bleu1_scores
df["BLEU-2"] = bleu2_scores
df["BLEU-3"] = bleu3_scores
df["BLEU-4"] = bleu4_scores
df["METEOR"] = meteor_scores
df["ROUGE-L"] = rouge_scores

# Save combined file
df.to_csv(output_file, index=False)

# Print averages
metric_columns = [
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "CIDEr",
    "CLIPScore",
]

print()
print("Per-image metrics saved to:")
print(output_file)
print()
print("Average per-image scores:")

print(df[metric_columns].mean())
