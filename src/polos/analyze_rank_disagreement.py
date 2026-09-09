from itertools import combinations

import pandas as pd

INPUT = "results/polos/per_image_metrics_with_polos.csv"
OUTPUT = "results/rank_disagreement.csv"
PAIR_OUTPUT = "results/rank_pair_disagreement.csv"

df = pd.read_csv(INPUT)

metrics = [
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "CIDEr",
    "CLIPScore",
    "polos",
]

# Convert every metric to percentile rank (0-1)
for metric in metrics:
    df[f"{metric}_rank"] = df[metric].rank(pct=True, method="average")

# ---------------------------------------------
# Pairwise rank disagreement
# ---------------------------------------------

pair_results = []

for metric1, metric2 in combinations(metrics, 2):

    difference = abs(df[f"{metric1}_rank"] - df[f"{metric2}_rank"])

    pair_results.append(
        {
            "metric_1": metric1,
            "metric_2": metric2,
            "mean_rank_disagreement": difference.mean(),
            "max_rank_disagreement": difference.max(),
        }
    )

pair_df = pd.DataFrame(pair_results)

pair_df = pair_df.sort_values("mean_rank_disagreement", ascending=False)

pair_df.to_csv(PAIR_OUTPUT, index=False)

print("\n=== LARGEST PAIRWISE RANK DISAGREEMENTS ===\n")

print(pair_df.head(15).round(3).to_string(index=False))

# ---------------------------------------------
# Traditional vs semantic rank
# ---------------------------------------------

traditional = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]

semantic = ["CLIPScore", "polos"]

df["traditional_rank"] = df[[f"{m}_rank" for m in traditional]].mean(axis=1)

df["semantic_rank"] = df[[f"{m}_rank" for m in semantic]].mean(axis=1)

df["rank_disagreement"] = df["semantic_rank"] - df["traditional_rank"]

# Largest absolute disagreements
df["absolute_rank_disagreement"] = abs(df["rank_disagreement"])

df = df.sort_values("absolute_rank_disagreement", ascending=False)

df.to_csv(OUTPUT, index=False)

columns = [
    "image",
    "generated_caption",
    "traditional_rank",
    "semantic_rank",
    "rank_disagreement",
]

print("\n=== STRONGEST INDIVIDUAL RANK DISAGREEMENTS ===\n")

print(df[columns].head(15).round(3).to_string(index=False))

print("\nSaved:")
print(OUTPUT)
print(PAIR_OUTPUT)
