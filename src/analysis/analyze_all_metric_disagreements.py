from itertools import combinations

import pandas as pd

INPUT = "results/metrics/per_image_metrics_with_polos.csv"
OUTPUT_PAIRS = "results/analysis/metric_pair_disagreements.csv"
OUTPUT_CASES = "results/analysis/metric_disagreement_cases.csv"

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

# 1. Normalize every metric to 0-1
normalized = pd.DataFrame()

for metric in metrics:
    minimum = df[metric].min()
    maximum = df[metric].max()
    normalized[metric] = (df[metric] - minimum) / (maximum - minimum)

# 2. Calculate disagreement for every metric pair
pair_results = []
for metric1, metric2 in combinations(metrics, 2):
    disagreement = abs(normalized[metric1] - normalized[metric2])
    pair_results.append(
        {
            "metric_1": metric1,
            "metric_2": metric2,
            "mean_disagreement": disagreement.mean(),
            "max_disagreement": disagreement.max(),
            "correlation": df[metric1].corr(df[metric2]),
        }
    )

pair_df = pd.DataFrame(pair_results)
pair_df = pair_df.sort_values("mean_disagreement", ascending=False)
pair_df.to_csv(OUTPUT_PAIRS, index=False)
print("\n=== METRIC PAIRS WITH MOST DISAGREEMENT ===\n")
print(pair_df.head(15).round(3).to_string(index=False))

# 3. Compare traditional vs semantic metrics
traditional = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
semantic = ["CLIPScore", "polos"]
normalized["traditional_score"] = normalized[traditional].mean(axis=1)
normalized["semantic_score"] = normalized[semantic].mean(axis=1)
normalized["group_disagreement"] = (
    normalized["semantic_score"] - normalized["traditional_score"]
)

# Add scores to original dataframe
cases = df.copy()
cases["traditional_score"] = normalized["traditional_score"]
cases["semantic_score"] = normalized["semantic_score"]
cases["group_disagreement"] = normalized["group_disagreement"]

# 4. Find semantic-high / traditional-low captions
semantic_high = cases.sort_values("group_disagreement", ascending=False)
print("\n=== SEMANTIC METRICS HIGHER THAN " "TRADITIONAL METRICS ===\n")
columns = [
    "image",
    "generated_caption",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "CIDEr",
    "CLIPScore",
    "polos",
    "group_disagreement",
]

print(semantic_high[columns].head(10).round(3).to_string(index=False))

# 5. Find traditional-high / semantic-low captions
traditional_high = cases.sort_values("group_disagreement", ascending=True)
print("\n=== TRADITIONAL METRICS HIGHER THAN " "SEMANTIC METRICS ===\n")
print(traditional_high[columns].head(10).round(3).to_string(index=False))

# Save all cases
cases = cases.sort_values("group_disagreement", key=abs, ascending=False)
cases.to_csv(OUTPUT_CASES, index=False)
print("\nSaved:")
print(OUTPUT_PAIRS)
print(OUTPUT_CASES)
